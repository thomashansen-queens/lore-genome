"""
Run manifest management for LoRe Genome pipeline.
"""

import inspect
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from lore.core.artifacts import Artifact
from lore.core.tasks import Task
from lore.core.utils import auto_increment, normalize_display_name


def _valid_sort_key(model_cls, key: str) -> bool:
    """Returns True if key is a field or property on the model."""
    if key in model_cls.model_fields:
        return True
    attr = inspect.getattr_static(model_cls, key, None)
    return isinstance(attr, property)


class Manifest(BaseModel):
    """
    Keeps a Manifest (a ledger), tracking Tasks, Artifacts, and metadata. Per Session logging.
    """
    # Session identity
    session_id: str
    session_name: str | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Ledger
    tasks: dict[str, Task] = Field(default_factory=dict)
    artifacts: dict[str, Artifact] = Field(default_factory=dict)

    # --- Construction ---

    @classmethod
    def new(cls, *, session_id: str, session_name: str | None = None) -> "Manifest":
        """Construct a fresh in-memory Manifest. Call save(path) to persist."""
        return cls(session_id=session_id, session_name=session_name)

    @classmethod
    def load(cls, path) -> "Manifest":
        """Load an existing Session Manifest from file."""
        if not path.exists():
            raise FileNotFoundError(f"Session Manifest not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Manifest JSON: {e}") from e
        return cls.model_validate(data)

    # --- Persistence ---

    def rebrand(self, new_id: str) -> None:
        """Replace session identity. Only for use during session cloning for now."""
        self.session_id = new_id
        self.created_at = datetime.now(timezone.utc)

    def save(self, path) -> None:
        """Atomic write Manifest to disk."""
        self.updated_at = datetime.now(timezone.utc)
        data = self.model_dump(mode="json")

        # Atomic write: temp file then move to prevent corruption on crash
        temp_path = path.with_suffix(".tmp")
        try:
            temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp_path.replace(path)
        except OSError as e:
            raise IOError(f"Failed to save Manifest to disk: {e}") from e

    # --- Helpers ---

    def _unique_name(
        self,
        proposed: str | None,
        ignore_id: str | None,
        kind: Literal["tasks", "artifacts"],
        *,
        default: str | None = None,
    ) -> str:
        """Generate unique names, handling cleaning and duplicates with auto-incrementing."""
        existing = [x.name for x in getattr(self, kind).values() if x.name and x.id != ignore_id]
        base = normalize_display_name(proposed, default=default or f"Unnamed {kind[:-1].capitalize()}")
        return auto_increment(base, existing)

    # --- Task CRUD ---

    def add_task(self, task: Task) -> None:
        """Add or update a Task in the manifest."""
        task.name = self._unique_name(task.name, task.id, kind="tasks")
        self.tasks[task.id] = task

    def get_task(self, task_id: str) -> Task | None:
        """Get a Task by ID."""
        return self.tasks.get(task_id)

    def list_tasks(
        self, sort_by: str | None = "created_at", reverse: bool = True,
    ) -> list[Task]:
        """Get a sorted list of all Tasks in the manifest."""
        if sort_by is None:
            return list(self.tasks.values())
        if not _valid_sort_key(Task, sort_by):
            raise ValueError(f"Invalid sort key '{sort_by}' for Task.")
        return sorted(self.tasks.values(), key=lambda t: getattr(t, sort_by), reverse=reverse)

    def rename_task(self, task_id: str, new_name: str | None) -> str:
        """Rename a Task, ensuring uniqueness. Returns the new name."""
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task ID '{task_id}' not found in manifest.")
        task.name = self._unique_name(new_name, task_id, kind="tasks")
        return task.name

    def remove_task(self, task_id: str) -> None:
        """Remove a Task from the manifest by its ID."""
        if task_id in self.tasks:
            del self.tasks[task_id]

    # --- Artifact CRUD ---

    def add_artifact(self, artifact: Artifact) -> None:
        """Add or update an Artifact in the manifest."""
        artifact.name = self._unique_name(artifact.name, artifact.id, kind="artifacts")
        self.artifacts[artifact.id] = artifact

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        """Get an Artifact by ID. Returns None if not found."""
        return self.artifacts.get(artifact_id)

    def list_artifacts(
        self, sort_by: str | None = "created_at", reverse: bool = True,
    ) -> list[Artifact]:
        """Get a sorted list of all Artifacts in the manifest."""
        if sort_by is None:
            return list(self.artifacts.values())
        if not _valid_sort_key(Artifact, sort_by):
            raise ValueError(f"Invalid sort key '{sort_by}' for Artifact.")
        return sorted(self.artifacts.values(), key=lambda a: getattr(a, sort_by), reverse=reverse)

    def rename_artifact(self, artifact_id: str, new_name: str | None, new_path: str | None = None) -> str:
        """Rename an Artifact, ensuring uniqueness. Returns the new name."""
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact ID '{artifact_id}' not found in manifest.")
        artifact.name = self._unique_name(new_name, artifact_id, kind="artifacts")
        if new_path is not None:
            artifact.path = new_path
        return artifact.name

    def remove_artifact(self, artifact_id: str) -> None:
        """Remove an Artifact from the manifest by its ID."""
        if artifact_id in self.artifacts:
            del self.artifacts[artifact_id]

    def get_task_output_artifacts(self, task_id: str) -> dict[str, list[Artifact]]:
        """
        Resolves a Task's output IDs into actual Artifact objects.
        
        Returns: 
            A dictionary mapping output_keys to lists of Artifacts.
            e.g., {"protein_fastas": [Artifact1, Artifact2], "report": []}
        """
        task = self.get_task(task_id)
        if not task or not hasattr(task, "outputs") or not task.outputs:
            return {}

        resolved = {}
        for slot_key, artifact_ids in task.outputs.items():
            resolved_list = []
            for a_id in artifact_ids:
                if a_id in self.artifacts:
                    resolved_list.append(self.artifacts[a_id])
                else:
                    # Defensive logging if a dangling ID ever occurs
                    # TODO: Give Manifest its own logger for such warnings
                    import logging
                    logging.getLogger("lore.manifest").warning(
                        f"Dangling artifact ID '{a_id}' found in Task '{task_id}' outputs."
                    )

            resolved[slot_key] = resolved_list

        return resolved
