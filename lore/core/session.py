"""
Session management for LoRe Genome.
"""
from contextlib import AbstractContextManager
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from pydantic import ValidationError

from lore.core.adapters import adapter_registry
from lore.core.artifacts import ArtifactManager, TransferMode
from lore.core.filelock import acquire_lock, release_lock
from lore.core.manifest import Manifest
from lore.core.tasks import Task
from lore.core.utils import is_artifact_id

if TYPE_CHECKING:
    from lore.core.artifacts import Artifact
    from lore.core.runtime import Runtime
    from lore.core.tasks import TaskDefinition


def _make_session_logger(session_id: str, log_path: Path) -> logging.Logger:
    """Create a logger for the session."""
    logger = logging.getLogger(f"lore.sessions.{session_id}")
    logger.setLevel(logging.INFO)
    if not any(
        isinstance(h, logging.FileHandler) \
        and getattr(h, 'baseFilename', '') == str(log_path) \
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class Session(AbstractContextManager):
    """
    This class is the handle to/instance of a Session. A Session is a container
    for all data (Tasks and Artifacts), isolated from other sessions.
    Manages the 'Lock' (open/close) on the Manifest file.
    """
    def __init__(self, *, session_root: Path, session_id: str, runtime: 'Runtime', read_only: bool = False):
        self.session_root = session_root
        self.id = session_id
        self.dir = session_root / session_id
        self.runtime = runtime
        self.read_only = read_only
        self._lock_file_obj = None
        self.logger: logging.Logger

        # Data components
        self.manifest: Manifest
        self.artifacts: ArtifactManager

    # ---- Context manager methods ---

    def __enter__(self) -> 'Session':
        """
        Open the session, acquiring the lock on the Manifest.
        Loads Manifest into memory.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logger = _make_session_logger(self.id, self.dir / "session.log")

        if not self.read_only:
            lock_path = self.dir / "manifest.json.lock"
            self._lock_file_obj = open(lock_path, "w", encoding="utf-8")
            acquire_lock(self._lock_file_obj, timeout=10)

        manifest_path = self.dir / "manifest.json"
        if manifest_path.exists():
            self.manifest = Manifest.load(path=manifest_path)
        else:
            self.manifest = Manifest.create(path=manifest_path, session_id=self.id)

        self.artifacts = ArtifactManager(
            artifacts_dir=self.dir / "artifacts",
            manifest=self.manifest,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Close the session, releasing the lock on the Manifest.
        Saves Manifest to disk.
        """
        if exc_type:
            self.logger.error("Session ID: '%s' exiting with error: %s", self.id, exc_value)
        else:
            self.logger.debug("Session ID: '%s' exiting normally.", self.id)
    
        if not self.read_only and exc_type is None:
            self.manifest.save()

        # Release lock and close the lock file
        if self._lock_file_obj:
            release_lock(self._lock_file_obj)
            self._lock_file_obj.close()
            self._lock_file_obj = None

        # Clean up logger handlers to prevent duplicate logs on re-open
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    # --- Session properties ---

    @property
    def name(self) -> str | None:
        """Get the session name from the manifest metadata."""
        return self.manifest.session_name

    @name.setter
    def name(self, value: str) -> None:
        """Set the session name in the manifest metadata."""
        self.manifest.session_name = value

    @property
    def display_size(self) -> str:
        """Returns human-readable size (kB, MB, GB)."""
        if not self.dir.exists():
            return "0 B"

        try:
            b = sum(f.stat().st_size for f in self.dir.rglob("*") if f.is_file())
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

    # --- Task management ---

    def _prepare_task_input_data(self, registry_key: str, raw_inputs: dict) -> tuple[dict, str]:
        """
        Validate inputs against the Task's input model. If it passes, mark the
        Task as PENDING. Otherwise, just return coerced/defaulted inputs and
        save as a DRAFT.
        """
        from lore.core.tasks import task_registry  # pylint: disable=import-outside-toplevel
        task_def = task_registry.get(registry_key)
        
        status = "PENDING"
        valid_inputs = raw_inputs or {}

        if task_def and task_def.input_model:
            try:
                # Full validation check
                model_inst = task_def.input_model(**valid_inputs)
                return model_inst.model_dump(mode='json'), "PENDING"
            except ValidationError:
                # Fallback to draft
                model_inst = task_def.input_model.model_construct(**valid_inputs)
                return model_inst.model_dump(mode='json'), "DRAFT"

        # If no input model, or no inputs, just pass through as PENDING
        return valid_inputs, status

    def upsert_task(
        self,
        registry_key: str,
        inputs: dict,
        task_name: str | None = None,
        target_task_id: str | None = None,
        parent_artifact_ids: list[str] | None = None,
        exec_config: dict | None = None,
    ) -> Task:
        """
        Creates or updates an existing Task. Handles lineage preservation. Mainly
        used in UI for modifying Tasks before execution. If target_task_id is
        provided and Task is pending, will try to update in-place.
        """
        # 1. Check Task status
        valid_inputs, status = self._prepare_task_input_data(registry_key, inputs)

        # 2. Update existing if it exists
        if target_task_id:
            task = self.get_task(target_task_id)
            if task and task.status in {"PENDING", "DRAFT"}:
                task.inputs = valid_inputs
                task.name = task_name or task.name
                task.status = status
                task.exec_config = exec_config or task.exec_config
                if parent_artifact_ids is not None:
                    current_parents = set(task.parent_artifact_ids)
                    current_parents.update(parent_artifact_ids)
                    task.parent_artifact_ids = sorted(list(current_parents))
                self.manifest.save()
                self.logger.info("Updated task: '%s' (ID: %s)", task.name, task.id)
                return task

        return self.add_task(
            registry_key,
            inputs=valid_inputs,
            task_name=task_name,
            status=status,
            exec_config=exec_config,
            parent_artifact_ids=parent_artifact_ids,
        )

    def add_task(
        self,
        registry_key: str,
        inputs: dict | None = None,
        task_name: str | None = None,
        status: str = "PENDING",
        parent_artifact_ids: list[str] | None = None,
        exec_config: dict | None = None,
    ) -> Task:
        """
        Register a new Task in the Manifest.

        :param registry_key: The key of the Task in the TaskRegistry.
        :param inputs: The input parameters for the Task. None if no inputs, uses defaults.
        :param task_name: Optional human-readable name for the Task.
        :param status: The initial status of the Task.
        :param parent_artifact_ids: Optional list of parent artifact IDs for lineage.
        :param exec_config: Optional execution configuration for the Task.
        :return: The created Task.
        """
        from uuid import uuid4  # pylint: disable=import-outside-toplevel
        from lore.core.tasks import task_registry  # pylint: disable=import-outside-toplevel

        # Validate registry key
        task_def = task_registry.get(registry_key)
        if not task_def:
            raise ValueError(f"Unknown Task key: '{registry_key}'. Is the library imported?")

        # Create Task
        task_id = str(uuid4())

        task = Task(
            id=task_id,
            registry_key=registry_key,
            name=task_name or f"{registry_key} {task_id[:4]}",
            status=status,
            inputs=inputs or {},
            outputs={},
            exec_config=exec_config or {},
            parent_artifact_ids=parent_artifact_ids or [],
        )
        self.manifest.add_task(task)
        self.logger.info("Registered task: '%s' (ID: %s)", task.name, task_id)
        return task

    def rename_task(self, task_id: str, new_name: str) -> bool:
        """Renames a Task without altering any other properties. True if renamed."""
        task = self.manifest.tasks.get(task_id)
        if not task:
            return False
        task.name = new_name
        return True

    def get_task(self, task_id: str) -> Task:
        """Retrieve a Task by ID. Raises ValueError if not found."""
        task = self.manifest.get_task(task_id)
        if not task:
            raise ValueError(f"Task ID: {task_id} not found in session ID: {self.id}")
        return task

    def list_tasks(self, sort_by: str | None = "created_at", reverse: bool = True) -> list[Task]:
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
        self.logger.info("Deleted task: '%s' (ID: %s)", task.name, task_id)
        return True

    def clone_task(self, task_id: str, new_name: str | None = None) -> Task:
        """Clones a Task, preserving the original Task's lineage."""
        original_task = self.manifest.get_task(task_id)
        if not original_task:
            raise ValueError(f"Cannot clone non-existent task ID: {task_id}")
        if not new_name:
            new_name = f"{original_task.name} (clone)"

        valid_inputs, status = self._prepare_task_input_data(
            original_task.registry_key,
            original_task.inputs,
        )
        new_task = self.add_task(
            registry_key=original_task.registry_key,
            inputs=valid_inputs,
            task_name=new_name,
            status=status,
            parent_artifact_ids=original_task.parent_artifact_ids,
            exec_config=original_task.exec_config,
        )
        self.manifest.save()
        self.logger.info("Cloned task: '%s' to '%s' (ID: %s)", original_task.id, new_task.id, new_task.id)
        return new_task

    def resolve_task_outputs(self, task_id: str) -> dict[str, dict[bool, Any]]:
        """
        Resolves a Task's raw outputs to Artifact objects where appropriate
        based on the TaskDefinition schema.
        """
        # May not be needed as a local import
        from lore.core.tasks import task_registry 

        task = self.get_task(task_id)
        if not task.outputs:
            return {}

        task_def = task_registry[task.registry_key]
        schema_fields = task_def.output_model.model_fields if (task_def and task_def.output_model) else {}

        resolved = {}
        for key, val in task.outputs.items():
            if val is None:
                resolved[key] = {"is_artifact": False, "value": None}
                continue

            # Check schema to determine if this slot holds an Artifact ID
            is_artifact = True  # Default to True
            field_info = schema_fields.get(key)
            if field_info and field_info.json_schema_extra is not None:
                is_artifact = field_info.json_schema_extra.get("is_artifact", True)

            if is_artifact:
                try:
                    artifact = self.get_artifact(val)
                    resolved[key] = {"is_artifact": True, "value": artifact}
                except ValueError:
                    resolved[key] = {"is_artifact": True, "value": "MISSING"}
            else:
                resolved[key] = {"is_artifact": False, "value": val}

        return resolved

    # --- Artifact helpers ---

    def get_artifact_path(self, artifact_id: str) -> Path:
        """
        Get fully resolved path to an Artifact by ID (self-healing for missing files)
        """
        path = self.artifacts.get_path(artifact_id)
        return path

    def register_artifact(
        self,
        path: Path,
        name: str | None = None,
        created_by_task_id: str | None = None,
        data_type: str = "unknown",
        metadata: dict | None = None,
        parent_artifact_ids: list[str] | None = None,
        transfer_mode: TransferMode = TransferMode.COPY,
    ) -> "Artifact":
        """Pass-through to Artifacts manager registrator."""
        # automatic lineage
        if parent_artifact_ids is None and created_by_task_id:
            task_record = self.manifest.get_task(created_by_task_id)
            if task_record:
                # Inherit the inputs of the Task that created this Artifact
                parent_artifact_ids = [
                    val for val in task_record.inputs.values()
                    if is_artifact_id(val)  # simple heuristic, could be improved
                ]

        return self.artifacts.register(
            source=path,
            name=name,
            created_by_task_id=created_by_task_id,
            data_type=data_type,
            metadata=metadata,
            parent_artifact_ids=parent_artifact_ids,
            transfer_mode=transfer_mode,
        )

    def ingest_artifact(
        self,
        path: Path,
        name: str | None = None,
        created_by_task_id: str | None = "ingested",
        data_type: str = "unknown",
        metadata: dict | None = None,
        parent_artifact_ids: list[str] | None = None,
        move: bool = False,
    ) -> "Artifact":
        """
        Coordinates ingestion of external or temp file into the Session vault.
        Delegates all file-system operations to ArtifactManager.
        """
        mode = TransferMode.MOVE if move else TransferMode.COPY
        return self.register_artifact(
            path=path,
            name=name,
            created_by_task_id=created_by_task_id,
            data_type=data_type,
            metadata=metadata,
            parent_artifact_ids=parent_artifact_ids,
            transfer_mode=mode,
        )

    def get_artifact(self, artifact_id: str) -> "Artifact":
        """Retrieve an ArtifactRecord by ID. Raises ValueError if not found."""
        artifact = self.manifest.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact ID: {artifact_id} not found in session ID: {self.id}")
        return artifact

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
        if not field_extra.get("is_artifact"):
            return []

        accepted_data = field_extra.get("accepted_data", ["*"])

        matches = []
        for art in self.list_artifacts():
            can_satisfy = False

            for accepted in accepted_data:
                # 1. Native match: The artifact is exactly what the task wants
                if accepted == "*" or art.data_type == accepted:
                    can_satisfy = True
                    break

                # 2. Adapted match: An adapter exists that can bridge the gap
                if adapter_registry.get_adapters(art, must_provide=accepted):
                    can_satisfy = True
                    break

            if can_satisfy:
                matches.append(art)

        return sorted(matches, key=lambda a: a.created_at, reverse=True)

    def get_artifact_candidates(self, task_def: "TaskDefinition") -> dict[str, list["Artifact"]]:
        """
        Find valid Artifacts for every field in a Task.
        Returns: { field_name: [list_of_valid_artifacts] }
        """
        candidates = {}

        for field_name, field_info in task_def.input_model.model_fields.items():
            extra = field_info.json_schema_extra or {}
            if isinstance(extra, dict) and extra.get("is_artifact"):
                candidates[field_name] = self.get_artifacts_for_field(extra)

        return candidates

    def rename_artifact(self, artifact_id: str, new_name: str) -> bool:
        """Renames an Artifact in the Manifest AND moves the file on disk to
        match the new name (ID_Slug name strategy)
        """
        changed = self.artifacts.rename(artifact_id, new_name)
        if changed:
            self.logger.info("Renamed artifact ID: %s to %s", artifact_id, new_name)

        return changed

    def delete_artifact(self, artifact_id: str) -> bool:
        """
        Pass-through to Artifacts manager deleter.
        Returns True if successful.
        """
        success = self.artifacts.delete(artifact_id)
        if success:
            self.logger.info("Deleted artifact: '%s' (ID: %s)", artifact_id, artifact_id)
        return success

    # --- Data access ---

    def preview_artifact(self, artifact_id: str, mode: str = "lines", **kwargs) -> tuple[str, bool]:
        """
        Pass-through to ArtifactManager preview methods.
        Available modes: "lines", "bytes", "adapter"
        """
        if mode == "lines":
            return self.artifacts.read_preview_lines(artifact_id, **kwargs)
        elif mode == "bytes":
            return self.artifacts.read_preview_bytes(artifact_id, **kwargs)
        elif mode == "adapter":
            return self.artifacts.read_adapter_preview(artifact_id, **kwargs)
        else:
            raise ValueError(f"Unknown preview mode: {mode}")

    def load_artifact_data(self, artifact_id: str) -> Any:
        """
        Pass-through: Parses an artifact into a Python object. Parses JSON,
        otherwise raw text.
        NOTE: This loads the file into memory.
        """
        return self.artifacts.load_data(artifact_id)
