"""
Preview execution logic for Tasks. Previews are meant to be fast, responsive
checks that can be used in the UI to validate inputs and get a sense of how a
Task will execute and should be configured.
"""
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel
from pydantic.fields import Field
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
    io_metadata: dict[str, Any] = Field(default_factory=dict)


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


@dataclass
class PreviewContext(ExecutionContext):
    """
    Specialized ExecutionContext for handling in-memory previews.
    Intercepts materialization to return UI-ready data payloads.
    Leaves the Manifest untouched.

    `inputs_complete`: simple flag indicating whether the input to
    the handler is delivered in full. If any input was 'peeked' (impartial load),
    outputs derived from it are also partial previews.
    """
    inputs_complete: bool = True

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
        adapter_config = self.task.exec_config.get("adapter", {})
        strategy = AdapterStrategy(adapter_config.get("strategy", AdapterStrategy.PEEK))
        peek_limit = self.runtime.settings.preview_peek_limit
        reader = get_reader_for(source_path)

        if strategy in (AdapterStrategy.FULL, AdapterStrategy.EAGER):
            try:
                raw_data = reader.read_full(config=adapter_config)
                io_meta = {"file_eof_hit": True, "io_strategy": "Full load"}
            except NotImplementedError:
                raw_data, io_meta = reader.preview(peek_limit=peek_limit, config=adapter_config)
        else:
            raw_data, io_meta = reader.preview(peek_limit=peek_limit, config=adapter_config)
            # 'file_eof_hit' means 'is this the complete result", which is only true
            # when the the inputs were read to EOF (not peeked) and the output was
            # read to EOF (not truncated)
            output_eof_hit = io_meta.get("file_eof_hit", True)
            io_meta["file_eof_hit"] = output_eof_hit and self.inputs_complete
            if not io_meta["file_eof_hit"]:
                io_meta["total_rows"] = None

        io_meta["extension"] = source_path.suffix.lstrip(".")
        io_meta["data_type"] = data_type

        # 3. Store in ephemeral results object
        self.results.add(output_key, (raw_data, io_meta))
        self.logger.debug("Preview intercepted file materialization for slot: %s", output_key)

        return DummyPreviewArtifact()  # Return a mock artifact for preview mode


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
    strategy = AdapterStrategy.FULL

    # 3. Resolve inputs
    with rt.open_session(session_id, read_only=True) as s:
        materialized = materialize_task_inputs(
            s=s,
            task_def=task_def,
            bindings=ephemeral_task.inputs,
            strategy=AdapterStrategy(strategy),
        )

    # An input was peeked (delivered partially) if any of its reads did not hit
    # EOF. If so, every output derived from it is a partial preview.
    inputs_complete = all(
        meta.get("file_eof_hit", True)
        for slot_metas in materialized.io_metadata.values()
        for meta in slot_metas
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
            input_artifacts=materialized.artifacts,
            inputs_complete=inputs_complete,
        )
    try:
        task_def.handler(ctx, **materialized.data)

        # 7. Extract preview outputs from the ephemeral context
        for key in ctx.results.output_keys:
            data_list = ctx.results[key]
            if not data_list:
                continue

            # For previews, only adapt the first output per output slot
            raw_data = data_list[0]
            if isinstance(raw_data, tuple) and len(raw_data) == 2 and isinstance(raw_data[1], dict):
                data, io_meta = raw_data
            else:
                data, io_meta = raw_data, {}

            # PreviewContext should always populate data_type in io_meta, but fall back to
            # field definition just in case
            data_type = io_meta.get("data_type")
            if not data_type:
                _, field_extra = task_def.field_meta(key, is_output=True)
                data_type = field_extra.get("data_type", "unknown")

            preview.output_previews[key] = PreviewOutput(
                data=data,
                data_type=data_type,
                io_metadata=io_meta,
            )

    except Exception as e:
        preview.error = str(e)
    finally:
        ctx.cleanup()

    return preview
