"""
Task execution logic. Is the engine behind Session and TaskRecord.
Uses Sandwich pattern (load -> run -> save) to manage task lifecycle within a session.
Allows long-running tasks to run without holding locks on the session manifest.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from pathlib import Path
import logging
import tempfile

from lore.core.io import get_reader_for
from lore.core.tasks import TaskResults

if TYPE_CHECKING:
    from lore.core.adapters import BaseAdapter
    from lore.core.artifacts import Artifact
    from lore.core.runtime import Runtime
    from lore.core.tasks import Task, TaskDefinition


@dataclass
class ExecutionContext:
    """
    Context object that doesn't hold a file lock. Holds rt and session_id
    in case it needs momentary access to session data (e.g. saving artifacts).
    Helpers safely open the session as needed.
    """
    runtime: "Runtime"
    session_id: str
    task: "Task"
    task_def: "TaskDefinition"
    input_artifacts: dict[str, list["Artifact"]] = field(default_factory=dict)
    results: TaskResults = field(init=False)
    _temp_dir: tempfile.TemporaryDirectory | None = field(default=None, init=False)

    def __post_init__(self):
        self.results = TaskResults(task_def=self.task_def)

    @property
    def logger(self) -> logging.Logger:
        """Namespaced logger for the task."""
        return self.runtime.logger.getChild(f"task.{self.task.registry_key}")

    def get_input_artifact(self, key: str) -> "list[Artifact] | None":
        """Get input Artifact record"""
        return self.input_artifacts.get(key)

    def get_input_adapter(self, input_key: str) -> "BaseAdapter | None":
        """Returns the primary adapter used to materialize a specific input."""
        artifact = self.input_artifacts.get(input_key, [])[0]
        if not artifact:
            return None

        adapters = artifact.get_adapters()
        return adapters[0] if adapters else None

    def get_temp_path(self, filename: str) -> Path:
        """Generates an isolated path in the task's scratch directory."""
        if self._temp_dir is None:
            cache_dir = self.runtime.settings.cache_root

            self._temp_dir = tempfile.TemporaryDirectory(
                prefix=f"task_{self.task.id[:8]}_",
                dir=cache_dir,
            )
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix=f"task_{self.task.id[:8]}_")

        return Path(self._temp_dir.name) / filename

    def memoize(self, key_prefix: str, func, **kwargs):
        """
        Use for heavy computations within a task that may benefit from caching.
        Makes use of Runtime's cache which is shared across Tasks and Sessions.
        Usage:
            my_obj = ctx.memoize("my_unique_key", expensive_function, arg1=val1)
        """
        def _thunk():
            return func(**kwargs)

        return self.runtime.cache.get_or_compute(
            session_id=self.session_id,
            prefix=key_prefix,
            compute_fn=_thunk,
            **kwargs
        )

    def cleanup(self):
        """Manually trigger the deletion of the scratch directory"""
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    # --- Materialization ---

    def _resolve_output_key(self, output_key: str | None) -> str:
        """Finds the provided key or assigns the next positional slot"""
        if output_key is not None:
            return output_key
        key = self.results.next_empty_key()
        if key is None:
            raise ValueError("Cannot assign TaskOutput: all slots are full.")
        return key

    def _resolve_data_type(self, output_key: str, data_type: str | None) -> str:
        """Finds the explicitly passed type or infers it from the TaskDefinition"""
        if data_type is not None:
            return data_type

        field_info = self.task_def.output_model.model_fields.get(output_key, None)
        if field_info is not None and field_info.json_schema_extra is not None:
            dt = field_info.json_schema_extra.get("data_type")
            if isinstance(dt, str):
                return dt

        return "unknown"

    def _get_inferred_parents(self) -> list[str]:
        """Helper to infer parents artifact_id lineage from Tasks's inputs"""
        return [a.id for artifacts in self.input_artifacts.values() for a in artifacts]

    def _generate_default_name(self, output_key: str | None, data_type: str) -> str:
        """
        Derives a human-readable name from the Task definition's Output label
        """
        label = None
        if output_key and self.task_def and self.task_def.output_model:
            field_info = self.task_def.output_model.model_fields.get(output_key)
            if field_info and hasattr(field_info.default, "label"):
                label = field_info.default.label

        fallback = label or output_key or data_type or "output"

        if self.task.name:
            return f"{str(self.task.name)[:12]}_{fallback}"
        return fallback

    def materialize_file(
        self,
        source_path: Path | str,
        name: str | None = None,
        output_key: str | None = None,
        data_type: str | None = None,
        metadata: dict | None = None,
        move: bool = True,
        **kwargs,
    ) -> "Artifact":
        """Adopts an existing file (e.g. temp file) in the Session"""
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file for materialization not found: {source_path}")

        # If not specified, fill outputs positionally
        output_key = self._resolve_output_key(output_key)
        data_type = self._resolve_data_type(output_key, data_type)
        name = name or self._generate_default_name(output_key, data_type)

        with self.runtime.open_session(self.session_id) as s:
            artifact = s.register_artifact(
                source=source_path,
                name=name,
                created_by_task_id=self.task.id,
                data_type=data_type,
                metadata=metadata,
                parent_artifact_ids=self._get_inferred_parents(),
                **kwargs,
            )

            try:
                self.results.add(output_key, artifact)
            except KeyError as e:
                self.logger.error("Error adding artifact to results: %s", str(e))
            except (AttributeError, ValueError) as e:
                self.logger.error("Error adding artifact to results: %s", str(e))
                raise

            self.logger.info("Task ID: %s emitted Artifact: %s to slot: %s",
                             self.task.id, artifact.id, output_key)
            return artifact

    def materialize_content(
        self,
        content: str | bytes,
        name: str | None = None,
        output_key: str | None = None,
        data_type: str | None = None,
        extension: str = "txt",
        metadata: dict | None = None,
        **kwargs,
    ) -> "Artifact":
        """
        Write in-memory content to tempfile of specified type then register it
        as an Artifact. Takes advantage of Session's atomic write and name
        collision logic.
        """
        output_key = self._resolve_output_key(output_key)
        data_type = self._resolve_data_type(output_key, data_type)
        ext = f".{extension.lstrip(".")}"

        tmp_path = self.get_temp_path(f"inline_content_{output_key}{ext}")

        byte_content = content.encode("utf-8") if isinstance(content, str) else content
        tmp_path.write_bytes(byte_content)

        try:
            return self.materialize_file(
                source_path=tmp_path,
                name=name,
                output_key=output_key,
                data_type=data_type,
                metadata=metadata,
                move=True,
                **kwargs,
            )
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()  # Clean up temp file on failure
                except OSError as e:
                    self.logger.warning("Failed to clean up temporary file: %s, (%s)", tmp_path, str(e))

# --- Preview execution for the Workbench ---

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
        setattr(self.results, output_key, payload)
        self.logger.debug("Preview intercepted file materialization for slot: %s", output_key)

        return DummyPreviewArtifact()  # Return a mock artifact for preview mode

    def _adapt_for_preview(self, source_path: Path, data_type: str) -> dict:
        """Logic to find adapter and prepare preview payload"""
        from lore.core.adapters import adapter_registry

        adapters = adapter_registry.get_adapters_by_type(data_type, source_path.suffix.lstrip("."))
        adapter = adapters[0] if adapters else None

        if not adapter:
            self.logger.warning("No adapter found for previewing data type '%s' with extension '%s'",
                                data_type, source_path.suffix)
            return {
                "is_preview": True,
                "view_mode": "raw",
                "adapter_name": "Raw file (no Adapter)",
                "data": f"No adapter found for {data_type}. Cannot preview.",
                "metadata": {"strategy_used": "system_fallback"},
            }

        try:
            reader = get_reader_for(source_path)
            raw_data, io_metadata = reader.preview(limit=100)

            adapter_config = self.task.exec_config.get("adapter", {})
            adapter_result = adapter.preview(raw_data, io_metadata, config=adapter_config)

            return {
                "data": adapter_result.data,
                "metadata": adapter_result.metadata,
                "is_preview": True,
                "view_mode": adapter.view_mode,
                "adapter_name": adapter.name,
            }

        except Exception as e:
            self.logger.error("Adapter preview failed for data type '%s': %s", data_type, str(e), exc_info=True)
            return {
                "is_preview": True,
                "view_mode": "raw",
                "adapter_name": adapter.name if adapter else "No adapter",
                "data": f"Error generating preview: {str(e)}",
                "metadata": {"error": str(e)},
            }
