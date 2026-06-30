"""
Preview execution logic for Tasks. Previews are meant to be fast, responsive
checks that can be used in the UI to validate inputs and get a sense of how a
Task will execute and should be configured.
"""
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel
from pydantic.fields import Field
from typing import Any, Literal, TYPE_CHECKING

from .context import ExecutionContext
from .materializer import materialize_task_inputs
from lore.core.readers import get_reader_for
from lore.core.tasks import AdapterStrategy, Task

if TYPE_CHECKING:
    from lore.core.runtime import Runtime


class PreviewOutput(BaseModel):
    """
    Structured output for a single previewed output slot.
    - data: the raw output data, whatever was returned by the handler
    - data_type: the semantic data type. TODO: Note to self, Is this actually the semantic type or the view type?
    - result_complete: were the inputs to the handler fully loaded? (false if 'peeked' or hit RAM ceiling)
    - display_complete: was the output itself fully loaded? (false if truncated for display or hit RAM ceiling)
    - truncation_reason: 'sampled' if intentional, 'capped' if RAM ceiling, null if complete.
    - io_metadata: any relevant metadata from the materialization layer (file format, row count, etc)
    If truncation_reason is 'sampled', the UI can offer an escalation: "Show full result"
    """
    data: Any
    data_type: str
    result_complete: bool = True
    display_complete: bool = True
    truncation_reason: Literal["sampled", "capped"] | None = None
    io_metadata: dict[str, Any] = Field(default_factory=dict)
    # Handler-declared metadata for this output (e.g. {"header": False}). The preview analog of Artifact.metadata
    source_metadata: dict[str, Any] = Field(default_factory=dict)


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
    # TODO(security): `traceback` exposes internal file paths and source structure.
    # Acceptable for now (local dev tool). Gate behind a debug setting before any
    # multi-user / HPC deployment.
    traceback: str | None = None


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

    For caching, checks if input_complete is true or not.
    """

    def materialize_file(
        self,
        source: Path | str | Mapping[str, Path | str],
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
        # 1. Normalize source to dict[str, Path] (shared with the real engine path)
        from lore.core.artifacts import normalize_sources
        sources = normalize_sources(source)

        # 2. Resolve output keys
        output_key = self._resolve_output_key(output_key)
        data_type = self._resolve_data_type(output_key, data_type)

        # 3. Delegate packaging
        adapter_config = self.task.exec_config.get("adapter", {})
        strategy = AdapterStrategy(adapter_config.get("strategy", AdapterStrategy.PEEK))
        peek_limit = self.runtime.settings.preview_peek_limit
        reader = get_reader_for(sources["main"])

        if strategy in (AdapterStrategy.FULL, AdapterStrategy.EAGER):
            try:
                raw_data = reader.read_full(config=adapter_config)
                io_meta = {"io_strategy": "Full load", "file_eof_hit": True}
            except NotImplementedError:
                raw_data, io_meta = reader.preview(peek_limit=peek_limit, config=adapter_config)
        else:
            raw_data, io_meta = reader.preview(peek_limit=peek_limit, config=adapter_config)

        # Can't know total row count of the output if the inputs were not fully loaded
        if not self.inputs_complete:
            io_meta["total_rows"] = None

        io_meta["extension"] = sources["main"].suffix.lstrip(".")
        io_meta["data_type"] = data_type

        # Carry the handler's declared metadata. Preview has no Artifact.metadata, so we stash it here
        source_metadata = dict(metadata or {})
        if kwargs:
            source_metadata.update(kwargs)
        io_meta["source_metadata"] = source_metadata

        # 4. Store in ephemeral results object
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
    strategy = AdapterStrategy(adapter_config.get("strategy", AdapterStrategy.PEEK))

    # Some tasks cannot function in PEEK mode, so escalate to FULL if the definition requires it.
    if task_def.preview_mode.loads_full_inputs:
        strategy = AdapterStrategy.FULL

    # 3. Resolve inputs
    with rt.open_session(session_id, read_only=True) as s:
        materialized = materialize_task_inputs(
            s=s,
            task_def=task_def,
            bindings=ephemeral_task.inputs,
            strategy=strategy,
        )

    # 4. Completeness check. Was every input delivered to the handler in full? If not, was
    # it an intentional 'peek' (escalation works) or a hard RAM ceiling (it won't)?
    input_metas = [
        meta
        for slot_metas in materialized.io_metadata.values()
        for meta in slot_metas
    ]
    inputs_complete = all(meta.get("file_eof_hit", True) for meta in input_metas)
    input_capped = any(meta.get("ram_limit_hit", False) for meta in input_metas)

    # 5. Initialize preview payload
    preview = PreviewPayload(
        task_key=task_key,
        is_dry_run=not task_def.preview_mode.executes_handler,
        resolved_inputs=clean_inputs,
        execution_config=ephemeral_task.exec_config,
    )

    # 6. "Check" logic: No handler execution, just return the validated config
    if not task_def.preview_mode.executes_handler:
        return preview

    # 7. Execute handler (and set data completeness in context for caching)
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

        # 8. Extract preview outputs from the ephemeral context
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

            # Handler metadata was stashed in io_meta purely for transport. Pop to store in PreviewOutput
            source_metadata = io_meta.pop("source_metadata", {})

            # PreviewContext should always populate data_type in io_meta, but fall back to
            # field definition just in case
            data_type = io_meta.get("data_type")
            if not data_type:
                _, field_extra = task_def.field_meta(key, is_output=True)
                data_type = field_extra.get("data_type", "unknown")

            # Full result is always returned from the worker, but the UI may truncate for display
            output_eof = io_meta.get("file_eof_hit", True)
            output_ram = io_meta.get("ram_limit_hit", False)
            display_complete = output_eof and not output_ram

            if not inputs_complete:
                truncation_reason = "capped" if input_capped else "sampled"
            elif not display_complete:
                truncation_reason = "capped"
            else:
                truncation_reason = None

            preview.output_previews[key] = PreviewOutput(
                data=data,
                data_type=data_type,
                result_complete=inputs_complete,
                display_complete=display_complete,
                truncation_reason=truncation_reason,
                io_metadata=io_meta,
                source_metadata=source_metadata,
            )

    except Exception as e:
        preview.error = str(e)
        preview.traceback = traceback.format_exc()
    finally:
        ctx.cleanup()

    return preview
