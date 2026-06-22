"""
Module to manage physical storage and references to Artifacts.
Artifacts can be single files or bundles of related files, so functions
to handle both cases are included.
"""

from collections.abc import Mapping
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


def normalize_sources(
    source: "Mapping[str, Path | str] | Path | str",
) -> dict[str, Path]:
    """
    Coerce an Artifact source into a bundle mapping {key: Path}.

    A bare path/str becomes the bundle's single 'main' file; a mapping is used
    as-is but must designate a 'main'.
    """
    if isinstance(source, (str, Path)):
        return {"main": Path(source)}
    if isinstance(source, Mapping):
        if "main" not in source:
            raise ValueError(
                "A multi-file Artifact bundle must designate one file as 'main'. "
                "Please include a 'main' key in the source mapping."
            )
        return {key: Path(value) for key, value in source.items()}
    raise TypeError(
        f"source must be a Path, str, or Mapping[str, Path | str]; got {type(source)}."
    )


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
        # Allow double-extensions like .csv.gz, but avoid doubling all
        if name.lower().endswith(f".{extension.lower()}"):
            name = name[: -len(extension) - 1]

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

    # --- Bundle management ---

    def ingest_bundle(
        self,
        sources: Mapping[str, Path | str],
        *,
        name: str | None = None,
        transfer_mode: TransferMode = TransferMode.COPY,
    ) -> dict[str, Any]:
        """
        Public dispatcher for ingesting a bundle of files.
        Calculates individual file hashes and a master bundle hash. TODO: Merkle tree?
        """
        if "main" not in sources:
            raise ValueError("Artifact bundles must contain a 'main' key.")

        # 1. Pre-calculate hases and sizes (individual)
        file_hashes = {}
        for key, source_str in sources.items():
            source_str = str(source_str).strip()
            if source_str.startswith(("http://", "https://", "s3://", "gs://", "ftp://")):
                file_hashes[key] = hashlib.sha256(source_str.encode("utf-8")).hexdigest()
            else:
                source_path = Path(source_str).resolve()
                if not source_path.exists():
                    raise FileNotFoundError(f"Artifact source file not found: {source_path}")
                file_hashes[key] = self._calculate_hash(source_path)

        # 2. Calculate deterministic master hash for the entire bundle
        sorted_hashes = [f"{k}:{file_hashes[k]}" for k in sorted(file_hashes.keys())]
        master_hash_input = "|".join(sorted_hashes).encode("utf-8")
        master_hash = hashlib.sha256(master_hash_input).hexdigest()
        artifact_id = master_hash[:12]

        # 3. Process each file using the common Master ID prefix
        processed_files = {}
        for key, source_str in sources.items():
            source_str = str(source_str).strip()

            # Extract natural extension for this specific file
            if source_str.startswith(("http://", "https://", "s3://", "gs://", "ftp://")):
                stats = self._ingest_remote_uri(source_str, artifact_id, file_hashes[key])
            else:
                source_path = Path(source_str).resolve()
                ext = source_path.name.split(".", 1)[1] if "." in source_path.name else ""
                target_path = self._generate_path(artifact_id, name or "unnamed", ext)
                stats = self._process_local_file(
                    source_path, target_path, transfer_mode, file_hashes[key],
                )

            processed_files[key] = stats

        return {
            "id": artifact_id,
            "hash": master_hash,
            "files": processed_files,
            "created_at": processed_files["main"]["created_at"],
            "metadata": {},  # TODO: Could aggregate metadata if needed
        }

    def _ingest_remote_uri(self, uri: str, artifact_id: str, file_hash: str) -> dict[str, Any]:
        """Registers a remote URI without downloading."""
        clean_path = urlparse(uri).path
        filename_part = posixpath.basename(clean_path)
        ext = filename_part.split(".", 1)[1] if "." in filename_part else "data"
        return {
            "hash": file_hash,
            "size_bytes": 0,
            "relative_path": uri,
            "extension": ext,
            "original_source": uri,
            "created_at": datetime.now(timezone.utc),
        }

    def _process_local_file(
        self,
        source_path: Path,
        target_path: Path,
        mode: TransferMode,
        file_hash: str,
    ) -> dict[str, Any]:
        """Physically moves/copies/links a single file into the Artifacts directory."""
        if source_path != target_path:
            if mode == TransferMode.LINK:
                if not source_path.is_file() or not source_path.is_absolute():
                    raise ValueError("Links require aboslute paths to regular files.")
                try:
                    target_path.symlink_to(source_path)
                except OSError as e:
                    raise RuntimeError(f"Failed to create symlink for Artifact: {e}") from e
            else:
                tmp_target = target_path.with_suffix(target_path.suffix + ".tmp")
                try:
                    if mode == TransferMode.MOVE:
                        shutil.move(str(source_path), str(tmp_target))
                    else:
                        shutil.copy2(str(source_path), str(tmp_target))
                    tmp_target.replace(target_path)
                except Exception as e:
                    if tmp_target.exists():
                        tmp_target.unlink()
                    raise IOError(f"Failed to ingest Artifact: {e}") from e

        ext = source_path.name.split(".", 1)[1] if "." in source_path.name else ""

        return {
            "hash": file_hash,
            "size_bytes": target_path.stat().st_size,
            "relative_path": str(
                target_path.relative_to(self.dir)
                if target_path.is_relative_to(self.dir)
                else target_path.resolve()
            ),
            "extension": ext,
            "original_source": str(source_path),
            "created_at": datetime.fromtimestamp(target_path.stat().st_mtime, tz=timezone.utc),
        }

    def rename_bundle(
        self,
        artifact_id: str,
        files_dict: Mapping[str, Any],
        new_name: str,
    ) -> dict[str, str]:
        """
        Renames all physical files in a bundle to match the new slug name.
        Returns a dict of {bundle_key: new_relative_path}
        """
        new_paths = {}
        for key, artifact_file in files_dict.items():
            old_path = self.dir / artifact_file.path
            if not old_path.exists():
                new_paths[key] = artifact_file.path
                continue

            ext = artifact_file.extension
            new_path = self._generate_path(artifact_id, new_name, ext)

            if old_path != new_path:
                try:
                    shutil.move(str(old_path), str(new_path))
                except OSError as e:
                    raise RuntimeError(f"Failed to rename artifact file: {e}") from e

            new_paths[key] = str(new_path.relative_to(self.dir))

        return new_paths

    def delete_bundle(self, files_dict: Mapping[str, Any]) -> None:
        """Physically delete all internal files in a bundle from disk."""
        for artifact_file in files_dict.values():
            path = Path(artifact_file.path)
            if not path.is_absolute():
                path = self.dir / path

            if path.exists() and path.is_relative_to(self.dir):
                path.unlink()

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
        candidates = list(self.dir.glob(f"{id_prefix}_*{path.suffix}"))
        candidates = [c for c in candidates if not c.name.endswith((".tmp", ".bak"))]

        if len(candidates) == 1:
            return candidates[0].resolve()

        raise FileNotFoundError(f"File not found for Artifact ID: {artifact_id}")
