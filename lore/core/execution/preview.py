"""
Preview execution logic for Tasks. Previews are meant to be fast, responsive
checks that can be used in the UI to validate inputs and get a sense of how a
Task will execute and should be configured.
"""
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel
from typing import Any, TYPE_CHECKING

from .context import ExecutionContext
from .materializer import materialize_task_inputs
from lore.core.io import get_reader_for
from lore.core.tasks import AdapterStrategy, Task

if TYPE_CHECKING:
    from lore.core.runtime import Runtime


class PreviewOutput(BaseModel):
    """Structured output for a single previewed output slot."""
    data: Any
    data_type: str


class PreviewPayload(BaseModel):
    """
    Unified API response for all Task preview modes.
    """
    task_key: str
    is_dry_run: bool

    resolved_inputs: dict[str, Any]
    execution_config: dict[str, Any]

    output_previews: dict[str, PreviewOutput] = {}
    engine_meta: dict[str, Any] = {}

    logs: str | None = None
    error: str | None = None


@dataclass
class DummyPreviewArtifact:
    """A mock Artifact to prevent AttributeErrors in Handlers during preview."""
    id: str = "preview_artifact_id"
    name: str = "preview_data"
    data_type: str = "unknown"


class PreviewContext(ExecutionContext):
    """
    Specialized ExecutionContext for handling in-memory previews.
    Intercepts materialization to return UI-ready data payloads.
    Leaves the Manifest untouched.
    """
    def materialize_file(
        self,
        source_path: Path | str,
        name: str | None = None,
        output_key: str | None = None,
        data_type: str | None = None,
        metadata: dict | None = None,
        move: bool = False,  # Previews shouldn't move files
        **kwargs,
    ) -> DummyPreviewArtifact:
        """
        Intercepts file materialization to return preview payload in RAM
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # 1. Resolve output keys
        output_key = self._resolve_output_key(output_key)
        data_type = self._resolve_data_type(output_key, data_type)

        # 2. Delegate packaging
        payload = self._adapt_for_preview(source_path, data_type)

        # 3. Store in ephemeral results object
        self.results.add(output_key, payload)
        self.logger.debug("Preview intercepted file materialization for slot: %s", output_key)

        return DummyPreviewArtifact()  # Return a mock artifact for preview mode

    def _adapt_for_preview(self, source_path: Path, data_type: str) -> dict:
        """Logic to find adapter and prepare preview payload"""
        from lore.core.adapters import adapter_registry

        # 1. Resolve adapter
        extension = source_path.suffix.lstrip(".") or "*"
        adapters = adapter_registry.get_for_type(data_type, extension)
        adapter = adapters[0] if adapters else None

        if not adapter:
            self.logger.warning(
                "No adapter found for previewing data type '%s' with extension '%s'",
                data_type,
                source_path.suffix,
            )
            return {
                "is_preview": True,
                "view_mode": "raw",
                "adapter_name": "Raw file (no Adapter)",
                "data": f"No adapter found for {data_type}. Cannot preview.",
                "metadata": {"strategy_used": "system_fallback"},
            }

        # 2. Read raw data and apply adapter
        try:
            adapter_config = self.task.exec_config.get("adapter", {})
            strategy = adapter_config.get("strategy", "peek")

            reader = get_reader_for(source_path)

            # Only load all data for a preview if explicitly requested
            if strategy in ("full", "eager"):
                raw_data = reader.read_full()
                io_metadata = {"file_eof_reached": True}
            else:
                raw_data, io_metadata = reader.preview(limit=100)

            adapter_result = adapter.preview(
                raw_data,
                io_metadata,
                config=adapter_config,
                ext=extension,
            )

            return {
                "data": adapter_result.data,
                "metadata": adapter_result.metadata,
                "is_preview": True,
                "view_mode": adapter.view_mode,
                "adapter_name": adapter.name,
            }

        except Exception as e:
            self.logger.error(
                "Adapter preview failed for data type '%s': %s", data_type, str(e), exc_info=True
            )
            return {
                "is_preview": True,
                "view_mode": "raw",
                "adapter_name": adapter.name if adapter else "No adapter",
                "data": f"Error generating preview: {str(e)}",
                "metadata": {"error": str(e)},
            }


def run_preview_worker(
    rt: "Runtime",
    session_id: str,
    task_key: str,
    raw_inputs: dict,
    exec_config: dict | None = None,
) -> PreviewPayload:
    """
    Execute a Task purely in memory. Is synchronous and meant for quick previews
    in the UI. Does not modify the Manifest or create Artifacts.
    Errors raise or return, rather than sys.exit.
    """
    from lore.core.tasks import task_registry

    # 1. Guards
    task_def = task_registry.get(task_key)
    if not task_def:
        raise ValueError(f"Task key: '{task_key}' not found in Task Registry.")
    if not task_def.preview_mode.is_allowed:
        raise RuntimeError(
            f"Previews are disabled for Task '{task_key}' "
            f"(preview_mode={task_def.preview_mode})."
        )

    rt.logger.info("Running preview for '%s' in Session ID: '%s'", task_key, session_id)

    # 2. Create ephemeral Task and validate inputs/execution config
    ephemeral_task = Task(
        id=f"preview_{uuid4().hex[:8]}",
        registry_key=task_key,
        inputs=raw_inputs,
    )
    try:
        ephemeral_task.exec_config = ephemeral_task.validate_config(exec_config or {})
        clean_inputs = ephemeral_task.validate_and_serialize()
    except Exception as e:
        raise ValueError(f"Input validation failed: {str(e)}") from e

    adapter_config = ephemeral_task.exec_config.get("adapter", {})
    strategy = adapter_config.get("strategy", AdapterStrategy.PEEK)

    # 3. Resolve inputs
    with rt.open_session(session_id, read_only=True) as s:
        resolved_inputs, input_artifacts = materialize_task_inputs(
            s=s,
            task_def=task_def,
            bindings=ephemeral_task.inputs,
            strategy=AdapterStrategy(strategy),
        )

    # 4. Initialize preview payload
    preview = PreviewPayload(
        task_key=task_key,
        is_dry_run=not task_def.preview_mode.executes_handler,
        resolved_inputs=clean_inputs,
        execution_config=ephemeral_task.exec_config,
    )

    # 5. "Dry Run" logic: No handler execution, just return the validated config
    if not task_def.preview_mode.executes_handler:
        return preview

    # 6. Execute handler
    ctx = PreviewContext(
            runtime=rt,
            session_id=session_id,
            task=ephemeral_task,
            task_def=task_def,
            input_artifacts=input_artifacts,
        )
    try:
        task_def.handler(ctx, **resolved_inputs)

        for key in ctx.results.output_keys:
            data_list = ctx.results[key]

            if data_list:
                continue

            _, field_extra = task_def.field_meta(key, is_output=True)

            # For preview purposes, only adapt the first item if 
            # multiple are present in an output slot
            preview.output_previews[key] = PreviewOutput(
                data=data_list[0],
                data_type=field_extra.get("data_type", "unknown"),
            )

    except Exception as e:
        preview.error = str(e)
    finally:
        ctx.cleanup()

    return preview
