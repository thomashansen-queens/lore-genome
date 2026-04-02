"""
Module to manage physical storage and references to Artifacts.
"""

from datetime import datetime, timezone
from enum import Enum
import hashlib
from pathlib import Path
import posixpath
import shutil
from typing import Any
from urllib.parse import urlparse

from lore.core.utils import slugify


class TransferMode(Enum):
    """How the handle files during registration"""

    MOVE = "move"  # fast, deletes source (use for temp files)
    COPY = "copy"  # safer, keeps source (use for user files)
    LINK = "link"  # space-efficient, symlinks in artifacts dir to source


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

        clean_path = urlparse(uri).path
        filename_part = posixpath.basename(clean_path)
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
        target_path = self._generate_path(artifact_id, name or "unnamed", ext)

        if source_path != target_path:
            try:
                target_path.symlink_to(source_path)
            except OSError as e:
                # self.logger.warning("Symlink failed for %s. Error: %s", source_path, e)
                raise RuntimeError(f"Failed to create symlink for Artifact: {e}") from e

        return self._build_stats_dict(artifact_id, artifact_hash, target_path, ext)

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
        target_path = self._generate_path(artifact_id, name or "unnamed", ext)

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

        return self._build_stats_dict(artifact_id, artifact_hash, target_path, ext)

    def _build_stats_dict(
        self, artifact_id: str, artifact_hash: str, target_path: Path, ext: str
    ) -> dict[str, Any]:
        """Populate Artifact metadata."""
        return {
            "id": artifact_id,
            "hash": artifact_hash,
            "size_bytes": target_path.stat().st_size,
            "relative_path": str(
                target_path.relative_to(self.dir)
                if target_path.is_relative_to(self.dir)
                else target_path.resolve()
            ),
            "extension": ext,
            "created_at": datetime.fromtimestamp(target_path.stat().st_mtime, tz=timezone.utc),
        }

    def rename_file(
        self, artifact_id: str, old_relative_path: str, new_name: str, extension: str
    ) -> str:
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

    def delete_file(self, relative_path: str) -> None:
        """Physically delete file from disk if it is internal."""
        path = Path(relative_path)
        if not path.is_absolute():
            path = self.dir / path

        if path.exists() and path.is_relative_to(self.dir):
            path.unlink()
