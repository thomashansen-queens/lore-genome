"""
Session management for LoRe Genome.
"""
from contextlib import AbstractContextManager
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lore.core.artifacts import Artifact, ArtifactManager, TransferMode
from lore.core.filelock import acquire_lock, release_lock
from lore.core.manifest import Manifest

if TYPE_CHECKING:
    from lore.core.runtime import Runtime
    from lore.core.tasks import Task, TaskDefinition


def _make_session_logger(session_id: str, log_path: Path) -> logging.Logger:
    """Create a logger for the session."""
    logger = logging.getLogger(f"lore.sessions.{session_id}")
    logger.setLevel(logging.INFO)
    if not any(
        isinstance(h, logging.FileHandler) \
        and getattr(h, "baseFilename", "") == str(log_path) \
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class Session(AbstractContextManager):
    """
    This class is the handle to/instance of a Session. A Session is a container
    for all data (Tasks and Artifacts), isolated from other sessions.
    Manages the 'Lock' (open/close) on the Manifest file.
    """
    def __init__(self, *, path: Path, session_id: str, runtime: "Runtime", read_only: bool = False):
        self.id = session_id
        self._root = path  # absolute path, set once at construction — never mutated
        self.runtime = runtime
        self.read_only = read_only
        self._lock_file_obj = None
        self._dirty = False  # True once any mutation is made; gates save() on __exit__

        # Initialized in __enter__
        self._logger: logging.Logger | None = None
        self._manifest: Manifest | None = None
        self._artifacts: ArtifactManager | None = None

    # ---- Context manager methods ---

    @property
    def dir(self) -> Path:
        """Read-only path to the session root directory. Set at construction by the Runtime."""
        return self._root

    def __enter__(self) -> "Session":
        """
        Open the Session, acquiring the lock on the Manifest.
        Loads Manifest into memory.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        self._logger = _make_session_logger(self.id, self._root / "session.log")

        if not self.read_only:
            lock_path = self._root / ".manifest.lock"
            self._lock_file_obj = open(lock_path, "w", encoding="utf-8")
            acquire_lock(self._lock_file_obj, timeout=10)

        manifest_path = self._root / "manifest.json"
        if manifest_path.exists():
            self._manifest = Manifest.load(manifest_path)
        else:
            self._manifest = Manifest.new(session_id=self.id)
            self._manifest.save(manifest_path)  # write immediately so disk state is consistent from first open

        self._artifacts = ArtifactManager(artifacts_dir=self._root / "artifacts")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Close the session, releasing the lock on the Manifest.
        Saves Manifest to disk.
        """
        if self._logger:
            if exc_type:
                self._logger.error("Session ID: '%s' exiting with error: %s", self.id, exc_value)
            else:
                self._logger.debug("Session ID: '%s' exiting normally.", self.id)

        if self._manifest and not self.read_only and exc_type is None and self._dirty:
            self._manifest.save(self._root / "manifest.json")

        # Release lock and close the lock file
        if self._lock_file_obj:
            release_lock(self._lock_file_obj)
            self._lock_file_obj.close()
            self._lock_file_obj = None

        # Null out in-memory state so the session is clearly "closed"
        self._manifest = None
        self._artifacts = None

        # Sync directory name to manifest name (handles deferred renames from open sessions)
        if not self.read_only and exc_type is None:
            self.runtime.sync_session_dir(self.id)

        # Clean up logger handlers to prevent duplicate logs on re-open
        if self._logger:
            handlers = self._logger.handlers[:]
            for handler in handlers:
                handler.close()
                self._logger.removeHandler(handler)

    # --- Session properties ---

    @property
    def name(self) -> str:
        """Get the session name from the Manifest, whether Session is open or closed"""
        if self._manifest:  # Session is open
            return self._manifest.session_name or self.id

        manifest_path = self._root / "manifest.json"  # Session is closed, read from disk
        if manifest_path.exists():
            try:
                return json.loads(manifest_path.read_text(encoding="utf-8")).get("session_name") or self.id
            except (json.JSONDecodeError, KeyError):
                return self.id  # Fallback to session ID if manifest is corrupted

        return self.id  # Fallback to session ID

    @name.setter
    def name(self, value: str) -> None:
        """Set the session name in the manifest metadata."""
        if self._manifest:  # Session is open
            if self._logger:
                self._logger.debug(
                    "Session name updated internally to '%s', directory updated deferred",
                    value
                )
            self._manifest.session_name = value
            self._dirty = True

        else:  # Session is closed — open it to update the manifest via normal machinery
            with self.runtime.open_session(self.id) as s:
                s.name = value  # Recurses into the open-session path above
            # __exit__ saved the manifest and sync_session_dir moved the directory if safe.
            # If a task was running, the dir rename is deferred until that task's session closes.
            # The session object (_root) is now stale — callers should re-fetch via rt.open_session()

    @property
    def display_size(self) -> str:
        """Returns human-readable size (kB, MB, GB)."""
        if not self._root.exists():
            return "0 B"

        try:
            b = sum(f.stat().st_size for f in self._root.rglob("*") if f.is_file())
        except (FileNotFoundError, PermissionError, OSError):
            return "0 B"

        if b < 1024:
            return f"{b} B"
        elif b < 1024 * 1024:
            return f"{b / 1024:.1f} kB"
        elif b < 1024 * 1024 * 1024:
            return f"{b / (1024 * 1024):.1f} MB"
        else:
            return f"{b / (1024 * 1024 * 1024):.2f} GB"

    @property
    def artifacts(self) -> "ArtifactManager":
        """Gatekeeper guarantees the ArtifactManager properly loaded"""
        if self._artifacts is None:
            raise RuntimeError(
                f"Attempted to access ArtifactManager for Session '{self.id}', "
                "but the Session is not open. Did you use a `with` block?"
            )
        return self._artifacts

    @property
    def manifest(self) -> "Manifest":
        """Gatekeeper guarantees the Manifest is loaded. Raises if Session is not open."""
        if self._manifest is None:
            raise RuntimeError(
                f"Attempted to access Manifest for Session '{self.id}', "
                "but the Session is not open. Did you use a `with` block?"
            )
        return self._manifest

    @property
    def logger(self) -> logging.Logger:
        """Gatekeeper guarantees the Logger active"""
        if self._logger is None:
            raise RuntimeError(
                f"Attempted to access Logger for Session '{self.id}', "
                "but the Session is not open. Did you use a `with` block?"
            )
        return self._logger

    def mark_dirty(self) -> None:
        """Signal that the manifest has been modified and should be saved on close."""
        self._dirty = True

    # --- Task management ---

    def add_task(
        self,
        registry_key: str,
        inputs: dict | None = None,
        name: str | None = None,
        parent_artifact_ids: list[str] | None = None,
        exec_config: dict | None = None,
    ) -> "Task":
        """
        Register a new Task in the Manifest.

        :param registry_key: The key of the Task in the TaskRegistry.
        :param inputs: The input parameters for the Task. None if no inputs, uses defaults.
        :param name: Optional human-readable name for the Task.
        :param parent_artifact_ids: Optional list of parent artifact IDs for lineage.
        :param exec_config: Optional execution configuration for the Task.
        :return: The created Task.
        """
        from uuid import uuid4
        from lore.core.tasks import Task, task_registry

        # Validate registry key
        if not task_registry.get(registry_key):
            raise ValueError(f"Unknown Task key: '{registry_key}'. Is the library imported?")

        # Create Task
        task_id = str(uuid4())

        task = Task(
            id=task_id,
            registry_key=registry_key,
            name=name or registry_key,
            status="DRAFT",
            inputs=inputs or {},
            outputs={},
            exec_config=exec_config or {},
            parent_artifact_ids=parent_artifact_ids or [],
        )
        self.manifest.add_task(task)
        task.update()  # auto-set DRAFT vs PENDING
        self.mark_dirty()
        self.logger.info("Registered task: '%s' (ID: %s)", task.name, task_id)
        return task

    def update_task(
        self,
        task_id: str,
        inputs: dict | None = None,
        exec_config: dict | None = None,
        name: str | None = None,
        parent_artifact_ids: list[str] | None = None,
    ) -> "Task":
        """
        Update a Task's inputs, config, and/or metadata. Returns the newly
        updated Task.
        """
        task = self.manifest.get_task(task_id)
        if task is None:
            raise ValueError(f"Task with ID '{task_id}' not found in Session '{self.id}'.")
        if parent_artifact_ids is not None:
            task.parent_artifact_ids = parent_artifact_ids
        task.update(inputs=inputs, exec_config=exec_config, name=name)
        self.mark_dirty()
        return task

    def rename_task(self, task_id: str, new_name: str) -> str:
        """Renames a Task without altering any other properties. Returns the new name."""
        result = self.manifest.rename_task(task_id, new_name)
        self.mark_dirty()
        return result

    def get_task(self, task_id: str) -> "Task | None":
        """Retrieve a Task by ID. Returns None if not found."""
        return self.manifest.get_task(task_id)

    def list_tasks(self, sort_by: str | None = "created_at", reverse: bool = True) -> list["Task"]:
        """Returns a list of all Tasks in the session."""
        return self.manifest.list_tasks(sort_by=sort_by, reverse=reverse)

    def delete_task(self, task_id: str) -> bool:
        """
        Pass-through to Manifest task deleter. Does not delete Artifacts.
        Returns True if successful.
        """
        task = self.manifest.get_task(task_id)
        if not task:
            raise ValueError(f"Cannot delete non-existent task ID: {task_id}")
        if task.status == "RUNNING":
            self.logger.warning("Deleting a running task: '%s' (ID: %s)", task.name, task_id)
            # raise RuntimeError("Cannot delete a running task (for now). Please wait for it to finish.")

        self.manifest.remove_task(task_id)
        self.mark_dirty()
        self.logger.info("Deleted task: '%s' (ID: %s)", task.name, task_id)
        return True

    def clone_task(self, task_id: str, *, new_name: str | None = None) -> "Task":
        """Clones a Task, preserving lineage"""
        original_task = self.manifest.get_task(task_id)
        if not original_task:
            raise ValueError(f"Cannot clone non-existent task ID: {task_id}")
        if new_name is None:
            new_name = original_task.name

        new_task = self.add_task(
            registry_key=original_task.registry_key,
            inputs=original_task.inputs,
            name=new_name,
            parent_artifact_ids=original_task.parent_artifact_ids,
            exec_config=original_task.exec_config,
        )
        self.mark_dirty()
        self.logger.info("Cloned task: '%s' to '%s' (ID: %s)", original_task.name, new_task.name, new_task.id)
        return new_task

    def resolve_task_outputs(self, task_id: str) -> dict[str, dict[str, Any]]:
        """
        Resolves a Task's raw outputs to Artifact objects and merges them with 
        their TaskOutput metadata (label, description, etc.)
        """
        # May not be needed as a local import
        from lore.core.tasks import task_registry 

        # 1. Fetch and guard
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"Cannot resolve outputs for non-existent task ID: {task_id}")

        if not task.outputs:
            return {}

        task_def = task_registry[task.registry_key]
        schema_fields = task_def.output_model.model_fields if (task_def and task_def.output_model) else {}

        resolved = {}
        for key, val in task.outputs.items():
            field_info = schema_fields.get(key)

            # 2. Sensible defaults in case of sparse schema
            is_artifact = False
            label = key.replace("_", " ").capitalize()
            description = ""
            is_primary = False
            data_type = "unknown"

            # 3. Get rich metadata
            if field_info:
                label = field_info.title or label
                description = field_info.description or description

                # LoRe specific metadata
                extra = getattr(field_info, "json_schema_extra", None)
                if extra is not None and isinstance(extra, dict):
                    is_artifact = extra.get("is_artifact", True)
                    is_primary = extra.get("is_primary", False)
                    data_type = extra.get("data_type", "unknown")
                    label = extra.get("label", label)

            # 4. Resolve Artifacts if applicable — val is always list[str] of artifact IDs
            if not is_artifact:
                resolved_value = val
            else:
                resolved_value = []
                for artifact_id in (val or []):
                    artifact = self.get_artifact(artifact_id)
                    if artifact is None:
                        self.logger.warning(
                            "Output '%s' of task '%s' (ID: %s) references artifact ID '%s' that cannot be found.",
                            key, task.name, task.id, artifact_id
                        )
                        resolved_value.append("MISSING")
                    else:
                        resolved_value.append(artifact)

            # 5. Package the resolved output with its metadata for convenient presentation
            resolved[key] = {
                "key": key,
                "label": label,
                "description": description,
                "data_type": data_type,
                "is_primary": is_primary,
                "is_artifact": is_artifact,
                "value": resolved_value,
            }

        return resolved

    def get_task_log_path(self, task_id: str) -> Path:
        """Returns the expected path to a Task's log file"""
        log_dir = self.dir / "logs"
        log_dir.mkdir(exist_ok=True)
        return log_dir / f"{task_id}.log"

    def get_task_log(self, task_id: str) -> str | None:
        """Safely reads the task log if it exists"""
        log_path = self.get_task_log_path(task_id)
        if log_path.exists():
            return log_path.read_text(encoding="utf-8")
        return None

    # --- Artifact helpers ---

    def get_artifact_path(self, artifact_id: str) -> Path:
        """
        Get fully resolved path to an Artifact by ID (self-healing for missing files)
        """
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact ID: {artifact_id} not found in manifest.")
        path = self.artifacts.resolve_path(artifact_id, recorded_path=artifact.path)
        return path

    def register_artifact(
        self,
        source: Path,
        transfer_mode: TransferMode = TransferMode.COPY,
        name: str | None = None,
        data_type: str = "unknown",
        created_by_task_id: str | None = None,
        parent_artifact_ids: list[str] | None = None,
        metadata: dict | None = None,
    ) -> "Artifact":
        """Factory method to create and register a new Artifact"""
        # 1. Perform file operations
        disk_stats = self.artifacts.ingest(source, name=name, transfer_mode=transfer_mode)

        # 2. Build state model
        artifact = Artifact(
            id=disk_stats["id"],
            name=name or Path(source).stem,
            path=disk_stats["relative_path"],
            size_bytes=disk_stats["size_bytes"],
            hash=disk_stats["hash"],
            data_type=data_type if data_type != "unknown" else disk_stats["extension"],
            created_by_task_id=created_by_task_id,
            parent_artifact_ids=parent_artifact_ids or [],
            created_at=disk_stats["created_at"],
            metadata=metadata or {},
        )

        self.manifest.add_artifact(artifact)
        self.mark_dirty()
        self.logger.info("Registered artifact: '%s' (ID: %s)", artifact.name, artifact.id)

        return artifact

    def get_artifact(self, artifact_id: str) -> "Artifact | None":
        """Retrieve an ArtifactRecord by ID"""
        return self.manifest.get_artifact(artifact_id)

    def list_artifacts(self, sort_by: str = "created_at", reverse: bool = True) -> list["Artifact"]:
        """Returns a list of all Artifacts in the session."""
        return self.manifest.list_artifacts(sort_by=sort_by, reverse=reverse)

    def get_artifacts_by_type(
        self,
        data_types: list[str],
    ) -> list["Artifact"]:
        """Returns a list of Artifacts using prefix matching."""
        if "*" in data_types:
            return self.list_artifacts()

        matches = []
        for a in self.list_artifacts():
            for type_str in data_types:
                if a.data_type == type_str or a.data_type.startswith(f"{type_str}."):
                    matches.append(a)
                    break

        return matches

    def get_artifacts_for_field(self, field_extra: dict) -> list["Artifact"]:
        """
        Returns a list of Artifacts able to satisfy an input field's data type requirements.
        """
        from lore.core.adapters import TableAdapter
        if not field_extra.get("is_artifact"):
            return []

        accepted_data = set(field_extra.get("accepted_data", ["*"]))

        valid_artifacts = []

        for artifact in self.list_artifacts():
            # 1. True wildcard
            if "*" in accepted_data:
                valid_artifacts.append(artifact)
                continue

            # 2. Adapter resolvable types
            resolvable_types = artifact.resolvable_types()
            if accepted_data & resolvable_types:
                valid_artifacts.append(artifact)

            # 3. Check metadata for schema-less matching
            if any (isinstance(adapter, TableAdapter) for adapter in artifact.get_adapters()):
                cols = set(artifact.metadata.get("columns", []))
                cols.update(artifact.metadata.get("keys", []))

                if accepted_data & cols:
                    valid_artifacts.append(artifact)

        return valid_artifacts

    def get_artifact_candidates(self, task_def: "TaskDefinition") -> dict[str, list["Artifact"]]:
        """
        Find valid Artifacts for every field in a Task.

        :returns: dict[field_name: [valid_artifacts]]
        """
        candidates = {}

        for field_name in task_def.input_model.model_fields.keys():
            _, extra = task_def.field_meta(field_name)
            if extra.get("is_artifact"):
                candidates[field_name] = self.get_artifacts_for_field(extra)

        return candidates

    def map_artifacts_to_task_inputs(self, task_def: "TaskDefinition", source_artifact_ids: list[str]) -> dict[str, Any]:
        """
        Given a TaskDefinition and a list of candidate Artifact IDs, determine 
        the best mapping of Artifacts to Task inputs.

        :returns: dict[field_name: artifact_id | list[artifact_id]]
        """
        if not source_artifact_ids:
            return {}

        artifacts = [a for aid in source_artifact_ids if (a := self.get_artifact(aid)) is not None]
        mapping = {}

        for key in task_def.input_model.model_fields.keys():
            _, extra = task_def.field_meta(key)

            accepted_data = set(extra.get("accepted_data", ["*"]))
            is_multiple = extra.get("cardinality", "single") in ("multiple", "pair", "two_or_more")

            for artifact in artifacts:
                resolvable = artifact.resolvable_types()

                if "*" in accepted_data or accepted_data.intersection(resolvable):
                    if is_multiple:
                        mapping.setdefault(key, []).append(artifact.id)
                    else:
                        if key not in mapping:
                            mapping[key] = artifact.id
                        break
        return mapping

    def rename_artifact(self, artifact_id: str, new_name: str) -> str:
        """Renames an Artifact in the Manifest AND moves the file on disk to
        match the new name (ID_Slug name strategy). Returns the new name.
        """
        artifact = self.manifest.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Cannot rename non-existent artifact ID: {artifact_id}")

        if artifact.name == new_name:
            return new_name

        new_path = self.artifacts.rename_file(
            artifact_id,
            old_relative_path=artifact.path,
            new_name=new_name,
            extension=artifact.extension,
        )
        new_name = self.manifest.rename_artifact(artifact_id, new_name, new_path=new_path)
        self.mark_dirty()
        return new_name

    def delete_artifact(self, artifact_id: str):
        """
        Pass-through to Artifacts manager deleter.
        """
        artifact = self.manifest.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Cannot delete non-existent artifact ID: {artifact_id}")

        self.artifacts.delete_file(artifact_id)
        self.manifest.remove_artifact(artifact_id)
        self.mark_dirty()
        self.logger.info("Deleted artifact: '%s' (ID: %s)", artifact_id, artifact_id)
