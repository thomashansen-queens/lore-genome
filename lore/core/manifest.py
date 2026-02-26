"""
Run manifest management for LoRe Genome pipeline.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lore.core.artifacts import Artifact
from lore.core.tasks import Task


class Manifest:
    """
    Keeps a Manifest (a ledger), tracking Tasks, Artifacts, and metadata. Per Session logging.
    """

    def __init__(self, *, path: Path, data: dict[str, Any]):
        self.path = path
        self._header = {
            k: v for k, v in data.items() if k not in {"tasks", "artifacts"}
        }
        # Hydrate Tasks and Artifacts using pydantic models
        self.tasks: dict[str, Task] = {
            k: Task.model_validate(v) for k, v in data.get("tasks", {}).items()
        }
        self.artifacts: dict[str, Artifact] = {
            k: Artifact.model_validate(v) for k, v in data.get("artifacts", {}).items()
        }

    @property
    def session_id(self) -> str:
        """The session ID this Manifest belongs to."""
        return self._header.get("session_id", "unknown")

    @property
    def session_name(self) -> str:
        """Get the user-visible session name."""
        # Default to a generic name if missing
        return self._header.get("session_name", f"session_{self.session_id[:8]}")

    @session_name.setter
    def session_name(self, value: str) -> None:
        """Set the session name and update the timestamp."""
        self._header["session_name"] = value

    @property
    def updated_at(self) -> datetime | None:
        """Get the last-updated timestamp."""
        ts = self._header.get("updated_at")
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        return None

    # --- Persistence on disk ---

    @classmethod
    def create(
        cls, *, path: Path, session_id: str, session_name: str | None = None
    ) -> "Manifest":
        """Create a new Session Manifest instance."""
        now = datetime.now(timezone.utc).isoformat()
        data = {
            "session_id": session_id,
            "session_name": session_name or f"session_{session_id[:8]}",
            "created_at": now,
            "updated_at": now,
            "tasks": {},
            "artifacts": {},
            "metadata": {},
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return cls(path=path, data=data)

    @classmethod
    def load(cls, *, path: Path) -> "Manifest":
        """Load an existing Session Manifest from file."""
        if not path.exists():
            raise FileNotFoundError(f"Session Manifest not found: {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Manifest JSON: {e}") from e

        return cls(path=path, data=data)

    def save(self) -> None:
        """Atomic write Manifest to disk."""
        self._header["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Serialize full manifest
        data = {
            **self._header,
            "tasks": {k: v.model_dump(mode="json") for k, v in self.tasks.items()},
            "artifacts": {k: v.model_dump(mode="json") for k, v in self.artifacts.items()},
        }

        # Atomic write: Temp file then move. Prevent corruption on crashes.
        temp_path = self.path.with_suffix(".tmp")
        try:
            temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp_path.replace(self.path)
        except OSError as e:
            raise IOError(f"Failed to save Manifest to disk: {e}") from e

    def _touch(self) -> None:
        """Update the updated_at timestamp but don't save. Not sure if useful."""
        self._header["updated_at"] = datetime.now(timezone.utc).isoformat()

    # --- Accessors (CRUD) ---

    def add_task(self, task: Task) -> None:
        """Add or update a task in the manifest."""
        self.tasks[task.id] = task

    def get_task(self, task_id: str) -> Task | None:
        """Get a Task by ID. Returns None if not found."""
        return self.tasks.get(task_id)

    def list_tasks(
        self, sort_by: str | None = "created_at", reverse: bool = True,
    ) -> list[Task]:
        """Get a list of all Tasks in the manifest."""
        if sort_by is None:
            return list(self.tasks.values())

        return sorted(
            list(self.tasks.values()),
            key=lambda task: getattr(task, sort_by),
            reverse=reverse,
        )

    def remove_task(self, task_id: str) -> str | None:
        """
        Remove a Task from the manifest by its ID. Returns the removed task name,
        or None if not found.
        """
        if task_id in self.tasks:
            name = self.tasks[task_id].name
            del self.tasks[task_id]
            return name
        return None

    def add_artifact(self, artifact: Artifact) -> None:
        """Add or update an Artifact in the manifest."""
        self.artifacts[artifact.id] = artifact

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        """Get an Artifact by ID. Returns None if not found."""
        return self.artifacts.get(artifact_id)

    def list_artifacts(
            self, sort_by: str | None = "created_at", reverse: bool = True,
        ) -> list[Artifact]:
        """Get a list of all Artifact in the manifest."""
        if sort_by is None:
            return list(self.artifacts.values())

        return sorted(
            list(self.artifacts.values()),
            key=lambda artifact: getattr(artifact, sort_by),
            reverse=reverse,
        )

    def remove_artifact(self, artifact_id: str) -> bool:
        """Remove an artifact from the manifest by its ID."""
        if artifact_id in self.artifacts:
            del self.artifacts[artifact_id]
            return True
        return False
