"""
Runtime context and configuration for LoRe Genome.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import logging
import logging.handlers
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING, Any

from lore.core.cache import TieredCache
from lore.core.execution.executors import LocalSubprocessExecutor
from lore.core.filelock import is_file_locked
from lore.core.manifest import Manifest
from lore.core.settings import (
    LOG_FORMAT,
    load_settings,
    Settings,
    save_settings,
)
from lore.core import paths
from lore.core.utils import auto_increment, fmt_bytes, slugify
from lore.core.workflows.manager import WorkflowManager

if TYPE_CHECKING:
    from lore.core.sessions import Session
    from lore.core.tasks.models import TaskResults


@dataclass
class SessionSummary:
    """Typed summary information about a Session."""

    id: str
    name: str
    path: Path
    display_size: str
    created_at: datetime | None
    updated_at: datetime | None


@dataclass
class Runtime:
    """Configuration for the LoRe Genome Runtime environment."""

    cache_dir: Path
    settings_dir: Path
    settings: Settings
    logger: logging.Logger
    cache: TieredCache = field(init=False)
    workflows: "WorkflowManager" = field(init=False)
    executor: LocalSubprocessExecutor = field(init=False)

    def __post_init__(self):
        self.cache = TieredCache(
            logger=self.logger,
            cache_dir=self.cache_dir,
            max_bytes=self.settings.cache_ram_mb * 1024 * 1024,
            max_disk=int(self.settings.cache_disk_gb * 1024 * 1024 * 1024),
        )
        self.executor = LocalSubprocessExecutor()
        builtin_workflows_dir = Path(__file__).parent.parent / "builtins" / "workflows"
        self.workflows = WorkflowManager(
            user_dir=self.settings.active_workflows_dir,
            read_dirs=[builtin_workflows_dir],
        )

        # Debug on launch
        self.logger.info("Initialized LoRe Runtime with data root: %s", self.data_root)
        self.logger.debug("Settings: %s", self.settings)
        self.logger.debug("Cache: %s", self.cache.cas_dir)

    @property
    def data_root(self) -> Path:
        """Settings dictates the data root directory."""
        return self.settings.data_root

    @property
    def sessions_dir(self) -> Path:
        """Derived dynamically from the active data_root."""
        return self.data_root / "sessions"

    def ensure_dirs(self) -> None:
        """Ensure all directories needed by the program exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.settings.active_workflows_dir.mkdir(parents=True, exist_ok=True)
        self.settings.active_plugins_dir.mkdir(parents=True, exist_ok=True)

    def update_settings(
        self,
        settings_dict: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Validates and updates settings/secrets. Returns (requires_restart, errors: list[str])
        """
        errors = []
        requires_restart = False

        # 1. Allow safe partial updates
        try:
            settings_updates = {
                k: v for k, v in settings_dict.items() if k in Settings.model_fields.keys()
            }

            if settings_updates:
                new_settings_data = self.settings.model_dump() | settings_updates
                self.settings = Settings.model_validate(new_settings_data)

        except ValueError as e:
            errors.append(f"Validation error: {str(e)}")
            return requires_restart, errors

        # 2. Save to disk
        save_settings(self.settings_dir, self.settings)

        # 3. Dynamic re-binding
        self.workflows.user_dir = self.settings.active_workflows_dir
        self.ensure_dirs()

        return requires_restart, errors

    # --- Session management ---

    def find_session_dir(self, session_id: str) -> Path | None:
        """
        Locates the physical directory for a given Session ID
        """
        # glob perfectly matches the immutable prefix
        matches = list(self.sessions_dir.glob(f"{session_id}*"))
        for match in matches:
            if match.is_dir():
                return match
        return None

    def create_session(self, *, name: str | None = None) -> "Session":
        """Create or load a Session within this Runtime."""
        from uuid import uuid4
        from lore.core.sessions import Session  # avoid circular import

        # 1. Generate a unique, immutable Session ID
        timestamp = datetime.now().strftime("%Y%m%d")
        short_uuid = str(uuid4())[:8]
        session_id = f"{timestamp}_{short_uuid}"

        # 2. Generate the mutable directory path
        slug = slugify(name) if name else None
        dir_name = f"{session_id}_{slug}" if slug else session_id
        ses_path = self.sessions_dir / dir_name

        # 3. Create the directory and return the new Session object
        ses_path.mkdir(parents=True, exist_ok=True)
        (ses_path / "artifacts").mkdir(parents=True, exist_ok=True)

        # 4. Initialize the Session and name if provided
        session = Session(path=ses_path, session_id=session_id, runtime=self)
        if name:
            session.name = name
        return session

    def import_session(self, zip_path: Path) -> str:
        """
        Packs a Session into a .zip archive
        TODO: Improve functionality (e.g. dehydrated session)
        """
        # 1. Unpack as temp file to be imported
        with tempfile.TemporaryDirectory() as temp_dir:
            staging_path = Path(temp_dir)
            shutil.unpack_archive(str(zip_path), str(staging_path))

            # 2. Load Manifest
            manifest_file = staging_path / "manifest.json"
            if not manifest_file.exists():
                raise ValueError(
                    "Invalid archive: Missing manifest.json. Does not quack like a Session."
                )

            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    imported_id = manifest_data.get("session_id")
                    if not imported_id:
                        raise ValueError("Invalid manifest: Missing session_id.")
            except json.JSONDecodeError:
                raise ValueError("Invalid manifest: manifest.json is not valid JSON.")

            # 3. Collision check (TODO: Handle reconciliation here)
            if self.find_session_dir(imported_id):
                raise ValueError(
                    f"Session ID collision: A session with ID '{imported_id}' already exists."
                )

            shutil.move(str(staging_path), str(self.sessions_dir / imported_id))

            self.logger.info(
                "Imported session '%s' with ID '%s'",
                manifest_data.get("session_name", "Unnamed"),
                imported_id,
            )
            return imported_id

    def clone_session(self, *, session_id: str) -> "Session":
        """
        Clone an existing Session for reuse. Performs a deep copy of the Session
        directory, then sanitizes the Manifest for reuse.
        """
        from lore.core.sessions import Session  # avoid circular import
        from lore.core.tasks import TaskStatus
        from uuid import uuid4
        from shutil import copytree

        source_path = self.find_session_dir(session_id)
        if not source_path:
            raise ValueError(f"Cannot clone '{session_id}' that doesn't exist!")
        if not (source_path / "manifest.json").exists():
            raise ValueError(f"Cannot clone '{session_id}': manifest.json is missing")

        # Snapshot Session names
        existing_names = [s.name for s in self.list_sessions()]

        # Generate new ID and copy, stealing the slug from the source directory name
        new_id = f"{datetime.now().strftime('%Y%m%d')}_{str(uuid4())[:8]}"
        slug_part = source_path.name[len(session_id) :]
        dest_path = self.sessions_dir / f"{new_id}{slug_part}"
        copytree(source_path, dest_path)

        # Sanitize the copied manifest through the Manifest class
        manifest_path = dest_path / "manifest.json"
        manifest = Manifest.load(manifest_path)
        manifest.rebrand(new_id)
        if manifest.session_name:
            manifest.session_name = auto_increment(manifest.session_name, existing_names)

        for task in manifest.tasks.values():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.READY
                task.error = "Status reset during session clone."

        manifest.save(manifest_path)

        return Session(path=dest_path, session_id=new_id, runtime=self)

    def open_session(self, session_id: str, read_only: bool = False) -> "Session":
        """Load an existing Session by its ID, finding its actual path on disk."""
        from lore.core.sessions import Session  # avoid circular import

        ses_path = self.find_session_dir(session_id)
        if not ses_path:
            raise FileNotFoundError(f"Session directory for ID '{session_id}' not found.")

        return Session(path=ses_path, session_id=session_id, runtime=self, read_only=read_only)

    def export_session(self, session_id: str, output_dir: Path) -> Path:
        """
        Exports a session as a .zip archive.
        TODO: Add functionality here (e.g. dehydrated)
        """
        session_dir = self.find_session_dir(session_id)
        if not session_dir:
            raise FileNotFoundError(f"Session '{session_id}' not found for export.")

        target_base = output_dir / f"lore_session_{session_id}"
        archive_path = shutil.make_archive(str(target_base), "zip", str(session_dir))

        return Path(archive_path)

    def rename_session(self, session_id: str, new_name: str) -> Path:
        """
        Renames a session. Delegates to Session.name so that the manifest is updated
        via the normal save machinery and the directory is synced on exit.
        If a task is running the directory rename is deferred until that task's session closes.
        Returns the current physical path (which may be the old path if rename was deferred).
        """
        with self.open_session(session_id) as s:
            s.name = new_name
            s.mark_dirty()

        path = self.find_session_dir(session_id)
        if path is None:
            raise FileNotFoundError(
                f"Session directory for ID '{session_id}' vanished after rename!"
            )
        return path

    def sync_session_dir(self, session_id: str) -> None:
        """
        Deferred naming/self-healing function. Checks if the physical directory
        slug matches the Manifest name. If they drift, align them.
        """
        path = self.find_session_dir(session_id)
        if not path:
            # Can't sync if we can't find it
            return

        if is_file_locked(path / ".manifest.lock"):
            # Can't sync if it's locked
            return

        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                tasks = data.get("tasks", {})
                if any(t.get("status") == "RUNNING" for t in tasks.values()):
                    # Don't rename while a task handler is executing
                    return
                name = data.get("session_name")
                if name:
                    expected_slug = slugify(name)
                    expected_dir_name = f"{session_id}_{expected_slug}"
                    if path.name != expected_dir_name:
                        import shutil

                        new_path = self.sessions_dir / expected_dir_name
                        shutil.move(str(path), str(new_path))
                        self.logger.info(
                            "Synced directory for ID: '%s' to match name '%s'", session_id, name
                        )
            except Exception as e:
                self.logger.warning(
                    "Error occurred while syncing session directory for ID '%s': %s", session_id, e
                )

    def list_sessions(self, sort_by: str | None = None) -> list[SessionSummary]:
        """
        Scans the Sessions directory and returns a summary of each. Newest first.
        """

        results = []
        for item in self.sessions_dir.iterdir():
            if not item.is_dir():
                continue

            if not (item / "manifest.json").exists():
                self.logger.warning("Session directory has no manifest, skipping: %s", item.name)
                continue

            session_id = item.name[:17]
            try:
                # Direct manifest reading is faster than going through the Session object
                manifest_data = json.loads((item / "manifest.json").read_text(encoding="utf-8"))
                created = manifest_data.get("created_at")
                updated = manifest_data.get("updated_at")

                results.append(
                    SessionSummary(
                        id=session_id,
                        name=manifest_data.get("session_name", "Unnamed Session"),
                        path=item,
                        display_size=fmt_bytes(manifest_data.get("size_bytes", 0)),
                        created_at=datetime.fromisoformat(created) if created else None,
                        updated_at=datetime.fromisoformat(updated) if updated else None,
                    )
                )
            except Exception as e:
                self.logger.warning("Could not load session %s: %s", item.name, e)

        sentinel = datetime.min.replace(tzinfo=timezone.utc)
        results.sort(key=lambda x: getattr(x, sort_by or "updated_at") or sentinel, reverse=True)
        return results

    def delete_session(self, session_id: str) -> None:
        """
        Permanently delete a session and all data by its ID.
        """
        import shutil

        ses_path = self.find_session_dir(session_id)
        if not ses_path:
            raise ValueError(f"Cannot delete '{session_id}' that doesn't exist.")
        shutil.rmtree(ses_path)
        self.logger.info("Deleted session: %s", session_id)

    # --- Execution management ---

    def execute_session(self, session_id: str) -> None:
        """
        Uses a background process to trigger the Execution Cascade in a Session.
        Returns immediately (non-blocking).
        """
        import subprocess
        import sys

        command = [
            sys.executable, "-m", "lore", "run-session",
            "--session", session_id,
        ]
        self.logger.info("Spawning background Orchestrator for Session: %s", session_id)
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def execute_task(self, session_id: str, task_id: str) -> None:
        """
        Spawn a background process to execute a single Task by its ID.
        Returns immediately (non-blocking).
        """
        import subprocess
        import sys

        command = [
            sys.executable, "-m", "lore", "run-task",
            "--session", session_id,
            "--task", task_id,
        ]
        self.logger.info("Spawning background Orchestrator for single Task %s", task_id)
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def preview_task(
        self, session_id: str, task_key: str, raw_inputs: dict, exec_config: dict | None = None
    ) -> "TaskResults":
        """
        Executes a Task in a special preview mode that does not mutate the Session or its data.
        Useful for quick iterations during development and debugging.
        """
        from lore.core.execution.worker import run_preview_worker

        return run_preview_worker(self, session_id, task_key, raw_inputs, exec_config)


# --- Runtime management ---


def build_runtime(
    *,
    data_root: Path | None = None,
    verbose: bool = False,
) -> Runtime:
    """Factory builds and returns a Runtime object for LoRe Genome."""

    # 1. Logging
    logger = logging.getLogger("lore")
    if not logging.getLogger().hasHandlers():
        # don't override uvicorn logging if it's already set up
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format=LOG_FORMAT,
        )
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # 2. Set directories and load settings
    settings_dir = paths.default_settings_dir()
    settings = load_settings(settings_dir)

    # 3. Cascade: Kwargs > Env vars > Settings file > Defaults
    if data_root is not None:
        settings.data_root = data_root.expanduser().resolve()
    elif env_root := os.getenv("LORE_DATA_ROOT"):
        settings.data_root = Path(env_root).expanduser().resolve()
    else:
        settings.data_root = settings.data_root.expanduser().resolve()

    if env_cache := os.getenv("LORE_CACHE_ROOT"):
        settings.cache_root = Path(env_cache)

    # 4. Create context
    rt = Runtime(
        cache_dir=settings.active_cache_root,
        settings_dir=settings_dir,
        settings=settings,
        logger=logger,
    )
    rt.ensure_dirs()

    # 5. Rotating file handler for runtime logs
    handler = logging.handlers.RotatingFileHandler(
        rt.settings.data_root / "lore_runtime.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=1,
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    rt.logger.addHandler(handler)

    # 6. Load plugins
    from lore.core.plugins import discover_plugins

    built_in_tasks_dir = Path(__file__).parent.parent / "builtins" / "tasks"
    discover_plugins(built_in_tasks_dir)
    built_in_adapters_dir = Path(__file__).parent.parent / "builtins" / "adapters"
    discover_plugins(built_in_adapters_dir)

    # External plugins
    user_plugins_dir = rt.settings.active_plugins_dir
    user_plugins_dir.mkdir(parents=True, exist_ok=True)
    discover_plugins(user_plugins_dir)

    return rt
