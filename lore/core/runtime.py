"""
Runtime context and configuration for LoRe Genome.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import logging
from typing import TYPE_CHECKING

# triggers decorators, registering Tasks and Adapters into global registries
# Ne touchez pas!
import lore.tasks # pylint: disable=unused-import
import lore.adapters # pylint: disable=unused-import

from lore.core.settings import load_settings, load_secrets, Settings, Secrets
from lore.core import paths
# from lore.core.validators import load_validators

if TYPE_CHECKING:
    # Puts 'Session' in the global namespace for type checking only.
    from lore.core.session import Session


@dataclass
class SessionSummary:
    """Typed summary information about a Session."""
    id: str
    name: str
    path: Path
    created_at: datetime
    updated_at: datetime


@dataclass
class Runtime:
    """Configuration for the LoRe Genome Runtime environment."""
    cache_dir: Path
    data_root: Path
    sessions_dir: Path
    settings_dir: Path
    settings: Settings
    secrets: Secrets
    logger: logging.Logger
    # self._queue: queue.Queue()
    # self._worker: threading.Threat(target=self._drain_queue, daemon=True)
    # self._worker.start()

    def ensure_dirs(self) -> None:
        """Ensure all directories needed by the program exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    # --- Session management ---

    def create_session(self, *, session_id: str | None = None) -> 'Session':
        """Create or load a Session within this Runtime."""
        # pylint: disable=import-outside-toplevel
        from lore.core.session import Session  # avoid circular import
        from uuid import uuid4
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid4())[:8]
        ses_path = self.sessions_dir / session_id
        ses_path.mkdir(parents=True, exist_ok=True)
        (ses_path / "artifacts").mkdir(parents=True, exist_ok=True)
        return Session(session_root=self.sessions_dir, session_id=session_id, runtime=self)

    def clone_session(self, *, session_id: str) -> 'Session':
        """
        Clone an existing Session for reuse. Performs a deep copy of the session
        directory, then sanitizes the Manifest for reuse.
        """
        # pylint: disable=import-outside-toplevel
        from lore.core.session import Session  # avoid circular import
        from uuid import uuid4
        from shutil import copytree
        import json
        source_path = self.sessions_dir / session_id
        if not source_path.exists() or not source_path.is_dir():
            raise ValueError(f"Source session '{session_id}' does not exist for cloning.")
        new_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid4())[:8]
        dest_path = self.sessions_dir / new_id
        copytree(source_path, dest_path)
        # Sanitize the manifest: It's a new session, so it should unique IDs
        manifest_path = dest_path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["session_id"] = new_id
            data["created_at"] = datetime.now(timezone.utc).isoformat()
            old_name = data.get('session_name', session_id)
            if not old_name.startswith("Cloned"):
                data['session_name'] = f"Cloned_{old_name}"
            # Reset Task statuses (in case previous tasks were RUNNING or FAILED)
            tasks = data.get("tasks", {})
            for task in tasks.values():
                if task.get("status") in ["RUNNING"]:
                    task["status"] = "PENDING"
                    task["error"] = "Status reset during session clone."
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        return Session(session_root=self.sessions_dir, session_id=new_id, runtime=self)

    def get_session(self, session_id: str, read_only: bool = False) -> 'Session':
        """Load an existing Session by its ID."""
        # pylint: disable=import-outside-toplevel
        from lore.core.session import Session  # avoid circular import
        return Session(session_root=self.sessions_dir, session_id=session_id, runtime=self, read_only=read_only)

    def list_sessions(self, sort_by: str | None = None) -> list[SessionSummary] | None:
        """
        Scans the Sessions directory and returns a summary of each. Newest first.
        """
        results = []
        if not self.sessions_dir.exists():
            raise FileNotFoundError(f"Sessions directory does not exist: {self.sessions_dir}")

        for item in self.sessions_dir.iterdir():
            if not item.is_dir():
                continue
            manifest_path = item / "manifest.json"
            created_at = datetime.fromtimestamp(item.stat().st_ctime, tz=timezone.utc)
            updated_at = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc)
            name = item.name
            if manifest_path.exists():
                try:
                    import json  # pylint: disable=import-outside-toplevel
                    data = json.loads(manifest_path.read_text())
                    name = data.get("session_name", name)
                    if data.get("created_at"):
                        created_at = datetime.fromisoformat(data["created_at"])
                    if data.get("updated_at"):
                        updated_at = datetime.fromisoformat(data["updated_at"])
                except (json.JSONDecodeError, OSError) as e:
                    self.logger.warning("Corrupt manifest found in session %s: %s", item.name, e)
                    name = f"CORRUPT: {name}"
            else:
                name = f"No manifest: {name}"

            results.append(SessionSummary(
                id=item.name,
                name=name,
                path=item,
                created_at=created_at,
                updated_at=updated_at,
            ))
        results.sort(key=lambda x: getattr(x, sort_by or "updated_at"), reverse=True)
        return results

    def delete_session(self, session_id: str) -> None:
        """
        Permanently delete a session and all data by its ID.
        """
        import shutil  # pylint: disable=import-outside-toplevel
        ses_path = self.sessions_dir / session_id
        try:
            shutil.rmtree(ses_path)
            self.logger.info("Deleted session: %s", session_id)
        except FileNotFoundError as e:
            raise ValueError(
                f"Session ID: {session_id} not found for deletion."
            ) from e

    # --- Executor management ---

    def execute_task(self, session_id: str, task_id):
        """
        Passthrough to Executor to run a Task by ID within a Session
        The executor handles opening/closing the session granularly.
        """
        from lore.core.executor import execute_task  # pylint: disable=import-outside-toplevel
        execute_task(self, session_id, task_id)

# --- Runtime management ---

def build_runtime(
    *,
    data_root: Path | None = None,
    verbose: bool = False,
) -> Runtime:
    """Builds and returns a Runtime object for LoRe Genome."""
    # 1. Logging
    logger = logging.getLogger("lore")
    if not logging.getLogger().hasHandlers():
        # don't override uvicorn logging if it's already set up
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    # 2. Set directories
    settings_dir = paths.default_settings_dir()
    cache_dir = paths.default_cache_dir()
    settings = load_settings(settings_dir)
    secrets = load_secrets(settings_dir)
    if data_root is not None:
        data_root = data_root.expanduser().resolve()
    elif env := os.getenv("LORE_DATA_ROOT"):
        data_root = Path(env).expanduser().resolve()
    elif settings.data_root is not None:
        data_root = settings.data_root.expanduser().resolve()
    else:
        data_root = paths.default_data_root()
    sessions_dir = data_root / "sessions"
    # 3. Create context
    rt = Runtime(
        cache_dir=cache_dir,
        data_root=data_root,
        sessions_dir=sessions_dir,
        settings_dir=settings_dir,
        settings=settings,
        secrets=secrets,
        logger=logger,
    )
    rt.ensure_dirs()
    return rt
