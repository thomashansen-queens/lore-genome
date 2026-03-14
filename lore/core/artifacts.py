"""
Manages the physical storage and retrieval of artifacts for a Session
"""

from datetime import datetime, timezone
from enum import Enum
import hashlib
from itertools import islice
from pathlib import Path
import shutil
from typing import Any, TYPE_CHECKING
from pydantic import BaseModel, Field, computed_field

from lore.core.utils import slugify

if TYPE_CHECKING:
    from lore.core.adapters import BaseAdapter
    from lore.core.tasks import TaskDefinition


class TransferMode(Enum):
    """How the handle files during registration"""
    MOVE = "move"  # fast, deletes source (use for temp files)
    COPY = "copy"  # safer, keeps source (use for user files)
    LINK = "link"  # space-efficient, symlinks in artifacts dir to source


class Artifact(BaseModel):
    """
    A discrete unit of data managed by LoRe.
    """
    id: str
    name: str | None = None
    path: str  # relative to Session dir for portability
    size_bytes: int
    hash: str
    data_type: str = "unknown"  # LoRe data type (e.g. "fasta", "json_report")
    # Lineage
    created_by_task_id: str | None = None
    parent_artifact_ids: list[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def filename(self) -> str:
        """Final path component e.g. `genome_report.json`"""
        return Path(self.path).name

    @property
    def extension(self) -> str:
        """Just the file extension without the dot e.g. `json`"""
        return Path(self.path).suffix.lstrip(".")

    @property
    def is_remote(self) -> bool:
        """
        Returns True if the path is an external URI rather than a local file.
        TODO: This is currently simple. Should also handle local paths outside Session
        """
        return self.path.startswith(("http://", "https://", "s3://", "gs://", "ftp://"))

    # --- Pipeline ---

    def resolvable_types(self) -> set[str]:
        """
        Determine semantic types this Artifact can satisfy
        Includes: its base type, explicit adapter types, dynamic schema types
        """
        from lore.core.adapters import TableAdapter

        resolvable = {self.data_type}

        for adapter in self.get_adapters():
            resolvable.update(adapter.provided_types)
            if isinstance(adapter, TableAdapter):
                if adapter.schema:
                    resolvable.update(adapter.schema.keys())

        return resolvable

    def get_adapters(self) -> list["BaseAdapter"]:
        """
        Convenience accessor for valid Adapters for this Artifact.
        """
        from lore.core.adapters import adapter_registry
        return adapter_registry.get_adapters_by_artifact(self)

    def get_push_tasks(self) -> list["TaskDefinition"]:
        """
        Return Tasks that can accept this artifact as a valid input.
        """
        from lore.core.tasks import task_registry
        capable_types = self.resolvable_types()
        return task_registry.compatible_tasks(capable_types)

    # --- UI ---

    @computed_field
    def display_size(self) -> str:
        """Returns human-readable size."""
        b = self.size_bytes
        if b < 1024:
            return f"{b} B"
        elif b < 1024 * 1024:
            return f"{b / 1024:.1f} kB"
        elif b < 1024 * 1024 * 1024:
            return f"{b / (1024 * 1024):.1f} MB"
        else:
            return f"{b / (1024 * 1024 * 1024):.2f} GB"

    @property
    def ui_name(self) -> str:
        """Returns a UI-friendly short name"""
        if self.name:
            return self.name if len(self.name) <= 20 else str(self.name)[:17] + "..."
        else:
            return "<Artifact ID" + self.id[:4] + "...>"


class ArtifactManager:
    """
    The pure-disk librarian for managing Artifacts. Handles physical storage, 
    hashing, and retrieval within the Session's artifact directory.
    """

    def __init__(self, artifacts_dir: Path):
        self.dir = artifacts_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _generate_path(self, artifact_id: str, name: str, extension: str) -> Path:
        """Generates a filesystem path for an artifact based on its ID and name."""
        safe_name = slugify(name)
        id_prefix = artifact_id[:8]
        filename = f"{id_prefix}_{safe_name}.{extension}"
        return self.dir / filename

    def _calculate_hash(self, path: Path) -> str:
        """Streaming SHA256 hash calculation for a file."""
        sha256 = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # --- Artifact management ---

    def ingest(
        self,
        source: Path | str,
        *,
        name: str | None = None,
        transfer_mode: TransferMode = TransferMode.COPY,
    ) -> dict[str, Any]:
        """
        Public dispatcher for ingesting data from a local file or remote URI.
        """
        source_str = str(source).strip()

        # Route 1: Remote URI
        if source_str.startswith(("http://", "https://", "s3://", "gs://", "ftp://")):
            return self._ingest_remote_uri(source_str, name)

        # Route 2: Local Files
        source_path = Path(source_str).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact source file not found: {source_path}")

        if transfer_mode == TransferMode.LINK:
            if not source_path.is_file():
                raise ValueError("Can only link regular files, not directories or special files.")
            if not source_path.is_absolute():
                raise ValueError("Source path must be absolute for linking to avoid ambiguity.")
            return self._ingest_local_symlink(source_path, name)

        return self._ingest_local_copy(source_path, name, transfer_mode)

    def _ingest_remote_uri(self, uri: str, name: str | None) -> dict[str, Any]:
        """
        Registers a remote URI/external reference without downloading. Hash based on URI string.
        """
        artifact_hash = hashlib.sha256(uri.encode("utf-8")).hexdigest()
        artifact_id = artifact_hash[:12]

        filename_part = uri.split("/")[-1]
        ext = filename_part.split(".")[-1] if "." in filename_part else "data"
        return {
            "id": artifact_id,
            "hash": artifact_hash,
            "size_bytes": 0,
            "relative_path": uri,  # Store URI in path for resolution
            "extension": ext,
            "created_at": datetime.now(timezone.utc),
        }

    def _ingest_local_symlink(self, source_path: Path, name: str | None) -> dict[str, Any]:
        """Creates a symlink in the artifacts directory pointing to a source."""
        # 1. Calculate identity
        artifact_hash = self._calculate_hash(source_path)
        artifact_id = artifact_hash[:12]

        # 2. Path resolution
        ext = source_path.suffix.lstrip(".")
        filename = self._generate_path(artifact_id, name or "unnamed", ext)
        target_path = self.dir / filename

        if source_path != target_path:
            try:
                target_path.symlink_to(source_path)
            except OSError as e:
                # self.logger.warning("Symlink failed for %s. Error: %s", source_path, e)
                raise RuntimeError(f"Failed to create symlink for Artifact: {e}") from e

        return self._build_stats_dict(artifact_id, artifact_hash, target_path, source_path, ext)

    def _ingest_local_copy(
        self,
        source_path: Path,
        name: str | None,
        mode: TransferMode,
    ) -> dict[str, Any]:
        """
        Physically copies/moves a file into the Artifact directory. Returns file stats.
        """
        # 1. Calculate identity (Content Addressable-ish)
        artifact_hash = self._calculate_hash(source_path)
        artifact_id = artifact_hash[:12]

        # 2. Path resolution
        ext = source_path.suffix.lstrip(".")
        filename = self._generate_path(artifact_id, name or "unnamed", ext)
        target_path = self.dir / filename

        # 3. Determine destination path and copy if needed
        if source_path != target_path:
            tmp_target = target_path.with_suffix(target_path.suffix + ".tmp")
            try:
                if mode == TransferMode.MOVE:
                    shutil.move(str(source_path), str(tmp_target))
                else:
                    shutil.copy2(str(source_path), str(tmp_target))

                # Atomic swap to final Artifacts dir
                tmp_target.replace(target_path)
            except Exception as e:
                if tmp_target.exists():
                    tmp_target.unlink()
                raise IOError(f"Failed to ingest Artifact: {e}") from e

        return self._build_stats_dict(artifact_id, artifact_hash, target_path, source_path, ext)

    def _build_stats_dict(self, artifact_id: str, artifact_hash: str, target_path: Path, source_path: Path, ext: str) -> dict[str, Any]:
        """Populate Artifact metadata."""
        return {
            "id": artifact_id,
            "hash": artifact_hash,
            "size_bytes": target_path.stat().st_size,
            "relative_path": str(target_path.relative_to(self.dir) if target_path.is_relative_to(self.dir) else target_path.resolve()),
            "extension": ext,
            "created_at": datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc),
        }

    def rename_file(self, artifact_id: str, old_relative_path: str, new_name: str, extension: str) -> str:
        """
        Renames an Artifact on disk to match (ID_Slug name). 
        Returns new relative path string
        """
        # 1. Calculate new path
        current_path = self.dir / old_relative_path
        if not current_path.exists():
            return old_relative_path  # File not managed here

        new_path = self._generate_path(artifact_id, new_name, extension)

        if current_path != new_path:
            try:
                shutil.move(str(current_path), str(new_path))
            except OSError as e:
                raise RuntimeError(f"Failed to rename artifact file: {e}") from e

        return str(new_path.relative_to(self.dir))

    def resolve_path(self, artifact_id: str, recorded_path: str) -> Path:
        """
        Resolved path with self-healing for "ghost files" (i.e., artifacts that
        are in the manifest but not on disk).
        """
        # 1. External files (not in Artifacts dir)
        path = Path(recorded_path)
        if path.is_absolute():
            return path

        # 2. Happy internal file
        full_path = self.dir / path
        if full_path.exists():
            return full_path.resolve()

        # 3. Self-healing: Try to find the file by hash if it's missing (crash corruption)
        id_prefix = artifact_id[:8]
        candidates = list(self.dir.glob(f"{id_prefix}_*"))
        candidates = [c for c in candidates if not c.name.endswith((".tmp", ".bak"))]

        if len(candidates) == 1:
            return candidates[0].resolve()

        raise FileNotFoundError(f"File for Artifact ID: {artifact_id} is missing.")

    def delete_file(self, artifact_id: str) -> None:
        """Physically delete file from disk if it is internal."""
        path = Path(artifact_id)
        if not path.is_absolute():
            path = self.dir / path

        if path.exists() and path.is_relative_to(self.dir):
            path.unlink()
