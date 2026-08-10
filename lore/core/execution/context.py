"""
Task execution logic. Is the engine behind Session and TaskRecord.
Uses Sandwich pattern (load -> run -> save) to manage task lifecycle within a session.
Allows long-running tasks to run without holding locks on the session manifest.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Iterator, TYPE_CHECKING
from pathlib import Path
import logging
import tempfile

from lore.core.artifacts.manager import TransferMode
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
    # Inputs are always complete (for now) for a task, but not for a preview
    inputs_complete: bool = True
    results: TaskResults = field(init=False)
    _temp_dir: tempfile.TemporaryDirectory | None = field(default=None, init=False)

    def __post_init__(self):
        self.results = TaskResults(task_def=self.task_def)

    @property
    def logger(self) -> logging.Logger:
        """Namespaced logger for the task."""
        return self.runtime.logger.getChild(f"task.{self.task.registry_key}")

    def get_config(self, plugin_key: str) -> Any:
        """Pass-through to get plugin-specific config from Runtime settings."""
        config_obj = self.runtime.settings.get_plugin_config(plugin_key)
        if config_obj is None:
            self.logger.warning("No config found for plugin '%s'", plugin_key)
        return config_obj

    def get_input_artifact(self, key: str) -> "list[Artifact] | None":
        """Get input Artifact record"""
        return self.input_artifacts.get(key)

    def get_input_adapter(self, input_key: str) -> "BaseAdapter | None":
        """Returns the primary adapter used to materialize a specific input."""
        artifacts = self.input_artifacts.get(input_key, [])
        if not artifacts:
            return None

        adapters = artifacts[0].get_adapters()
        return adapters[0] if adapters else None

    def get_temp_path(self, filename: str) -> Path:
        """Generates an isolated path in the task's scratch directory."""
        base_dir = Path(self._temp_dir.name) if self._temp_dir else self._init_temp_dir()
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / filename

    def get_temp_dir(self, dirname: str | None = None) -> Path:
        """Generates an isolated directory in the task's scratch space."""
        base_dir = Path(self._temp_dir.name) if self._temp_dir else self._init_temp_dir()
        if dirname:
            subdir = base_dir / dirname
            subdir.mkdir(parents=True, exist_ok=True)
            return subdir
        return base_dir

    def _init_temp_dir(self) -> Path:
        """Helper to initialize an empty temp directory if it doesn't yet exist."""
        cache_dir = self.runtime.settings.cache_root
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix=f"task_{self.task.id[:8]}_",
            dir=cache_dir,
        )
        return Path(self._temp_dir.name)

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
            cache_kwargs=kwargs,
            persist=self.inputs_complete,
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

        with self.runtime.open_session(self.session_id, read_only=True) as s:
            resolved_extra = self.task.resolve_output_type(output_key, s)

        data_type = resolved_extra.get("data_type")
        if isinstance(data_type, str):
            return data_type

        self.logger.warning(
            "Failed to resolve data type for output '%s'. Defaulting to 'unknown',",
            output_key
        )
        return "unknown"

    def _get_inferred_parents(self) -> list[str]:
        """Helper to infer parents artifact_id lineage from Tasks's inputs"""
        return [a.id for artifacts in self.input_artifacts.values() for a in artifacts]

    def _generate_default_name(self, output_key: str | None, data_type: str) -> str:
        """
        Derives a human-readable name from the Task definition's Output label
        """
        fallback = output_key or data_type or "output"
        if self.task.name:
            return f"{str(self.task.name)[:12]}_{fallback}"
        return fallback

    def materialize_file(
        self,
        source: Path | str | Mapping[str, Path | str],
        output_key: str | None = None,
        name: str | None = None,
        data_type: str | None = None,
        metadata: dict | None = None,
        move: bool = True,
        **kwargs,
    ) -> "Artifact":
        """
        Adopts an existing file (e.g. temp file) in the Session.
        Accepts a single path (auto-assigned to 'main') or a dictionary mapping
        artifact bundle keys to paths (must explicitly include 'main').
        """
        from lore.core.artifacts import normalize_sources

        # 1. Normalize source to dict[str, Path]
        sources = normalize_sources(source)

        # 2. Validate physical existence
        for key, path in sources.items():
            if not path.exists():
                raise FileNotFoundError(f"Source file for key '{key}' does not exist: {path}")

        # 3. If not specified, fill outputs positionally
        output_key = self._resolve_output_key(output_key)
        data_type = self._resolve_data_type(output_key, data_type)
        name = name or self._generate_default_name(output_key, data_type)

        # Catch stray kwargs
        final_metadata = metadata or {}
        if kwargs:
            final_metadata.update(kwargs)

        # 4. Delegate to Session manager
        with self.runtime.open_session(self.session_id) as s:
            artifact = s.register_artifact(
                source=sources,
                transfer_mode=TransferMode.MOVE if move else TransferMode.COPY,
                name=name,
                data_type=data_type,
                created_by_task_id=self.task.id,
                created_by_output_key=output_key,
                parent_artifact_ids=self._get_inferred_parents(),
                metadata=final_metadata,
            )

            try:
                self.results.add(output_key, artifact)
            except (AttributeError, KeyError, ValueError) as e:
                self.logger.error("Error adding artifact to results: %s", str(e))
                raise

            self.logger.info(
                "Task ID: %s emitted Artifact: %s to slot: %s (Files: %s)",
                self.task.id,
                artifact.id,
                output_key,
                list(sources.keys()),
            )
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
        as an Artifact. Automatically becomes the 'main' file of the Artifact.
        """
        output_key = self._resolve_output_key(output_key)
        data_type = self._resolve_data_type(output_key, data_type)
        ext = f".{extension.lstrip('.')}"

        tmp_path = self.get_temp_path(f"inline_content_{output_key}{ext}")

        byte_content = content.encode("utf-8") if isinstance(content, str) else content
        tmp_path.write_bytes(byte_content)

        try:
            return self.materialize_file(
                source=tmp_path,  # single path -> 'main' file
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
                    self.logger.warning(
                        "Failed to clean up temporary file: %s, (%s)", tmp_path, str(e)
                    )

    def materialize_stream(
        self,
        stream: Iterator[str | bytes],
        name: str | None = None,
        output_key: str | None = None,
        data_type: str | None = None,
        extension: str = "txt",
        metadata: dict | None = None,
        **kwargs,
    ) -> "Artifact":
        """
        Consumes an iterator stream of content, writing chunk-by-chunk to a tempfile,
        then registers it as an Artifact. Useful for streaming very large outputs
        without holding them in memory.
        Note that this is an atomic write, so the full stream must be consumed before
        the Artifact is registered and visible to other Tasks. Also, if the Task
        fails mid-stream, the temp file will be cleaned up and no Artifact will be 
        visible.
        """
        output_key = self._resolve_output_key(output_key)
        data_type = self._resolve_data_type(output_key, data_type)
        ext = f".{extension.lstrip('.')}"

        tmp_path = self.get_temp_path(f"streamed_content_{output_key}{ext}")

        with open(tmp_path, "wb") as f:
            for chunk in stream:
                byte_chunk = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
                f.write(byte_chunk)

        try:
            return self.materialize_file(
                source=tmp_path,  # single path -> 'main' file
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
                    self.logger.warning(
                        "Failed to clean up temporary file: %s, (%s)", tmp_path, str(e)
                    )
