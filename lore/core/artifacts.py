"""
Manages the physical storage and retrieval of artifacts for a Session
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
import hashlib
from itertools import islice
from pathlib import Path
import shutil
from typing import Any, BinaryIO, ContextManager, Literal, overload, TextIO, TYPE_CHECKING
from pydantic import BaseModel, Field, computed_field

from lore.core.utils import slugify

if TYPE_CHECKING:
    from lore.core.manifest import Manifest
    from lore.core.adapters import BaseAdapter
    from lore.core.session import Session
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
    path: str  # relative to Session root for portability
    size_bytes: int
    hash: str
    data_type: str = "unknown"  # LoRe data type (e.g. "fasta", "json_report")
    # Lineage
    created_by_task_id: str | None = None
    parent_artifact_ids: list[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

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
    def filename(self) -> str:
        """Final path component e.g. `genome_report.json`"""
        return Path(self.path).name

    @property
    def extension(self) -> str:
        """Just the file extension without the dot e.g. `json`"""
        return Path(self.path).suffix.lstrip(".")

    @property
    def ui_name(self) -> str:
        """Returns a UI-friendly name"""
        if self.name:
            return self.name if len(self.name) <= 16 else str(self.name)[:13] + "..."
        else:
            return "<Artifact ID" + self.id[:4] + "...>"

    def get_adapters(self) -> list["BaseAdapter"]:
        """
        Convenience accessor for valid Adapters for this Artifact.
        """
        from lore.core.adapters import adapter_registry  # pylint: disable=import-outside-toplevel;
        return adapter_registry.get_adapters(self)

    def get_push_tasks(self) -> list[tuple[str, "TaskDefinition"]]:
        """
        Return Tasks that can accept this artifact as a valid input.
        """
        from lore.core.tasks import task_registry  # pylint: disable=import-outside-toplevel
        # 1. Determine all semantic types this Artifact can satisfy (including via Adapters)
        capable_types = {self.data_type}
        for adapter in self.get_adapters():
            capable_types.update(adapter.provided_types)

        # 2. Find Tasks that can accept any of these types as input
        push_tasks: list[tuple[str, "TaskDefinition"]] = []
        for key, task_def in task_registry.all.items():
            model_fields = task_def.input_model.model_fields

            for _, field_info in model_fields.items():
                if field_info.json_schema_extra is None:
                    continue

                accepted = field_info.json_schema_extra.get("accepted_data", [])
                if not isinstance(accepted, (list, set)):
                    continue

                accepted = set(accepted)
                if "*" in accepted or accepted.intersection(capable_types):
                    push_tasks.append((key, task_def))
                    break  # No need to check other fields if one matches

        return push_tasks


class ArtifactManager:
    """
    The librarian class for physical storage, retrieval, and registration of
    Artifacts within a Session.
    """

    def __init__(self, artifacts_dir: Path, manifest: "Manifest"):
        self.dir = artifacts_dir
        self.manifest = manifest
        self.dir.mkdir(parents=True, exist_ok=True)

    def _generate_path(self, artifact_id: str, name: str, extension: str) -> Path:
        """Generates a filesystem path for an artifact based on its ID and name."""
        safe_name = slugify(name)
        if_prefix = artifact_id[:8]
        filename = f"{if_prefix}_{safe_name}.{extension}"
        return self.dir / filename

    # --- Artifact management ---

    def register(
        self,
        source: Path | str,
        *,
        name: str | None = None,
        data_type: str = "unknown",
        created_by_task_id: str | None = None,
        parent_artifact_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transfer_mode: TransferMode = TransferMode.COPY,
    ) -> Artifact:
        """
        Register a file into the session as an Artifact model. Handles copying/
        moving into the Session's artifact directory.

        : param source: Path to the file to register.
        : param name: Optional name for the artifact. Defaults to the source filename.
        : param data_type: Internal data type of the artifact (e.g., "fasta", "json_report"). Defaults to "unknown".
        : param created_by_task_id: Optional Task ID that produced this artifact.
        : param parent_artifact_ids: Optional list of parent Artifact IDs to associate with this Artifact for lineage
        : param metadata: Optional dictionary of metadata to associate with the artifact.
        : param data_type: Type/format of the artifact (e.g., "fasta", "json_report"). Defaults to file extension.
        : param transfer_mode: How to handle the source file (MOVE, COPY, LINK).
        """
        # Guards
        source_path = Path(source).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact source file not found: {source_path}")

        # 1. Calculate identity (Content Addressable-ish)
        artifact_hash = self._calculate_hash(source_path)
        artifact_id = artifact_hash[:12]

        # 2. Path resolution
        filename = self._generate_path(artifact_id, name or source_path.stem, source_path.suffix.lstrip("."))
        target_path = self.dir / filename

        # 3. Determine destination path and copy if needed
        if source_path != target_path:
            tmp_target = target_path.with_suffix(target_path.suffix + ".tmp")
            try:
                if transfer_mode == TransferMode.MOVE:
                    shutil.move(str(source_path), str(tmp_target))
                else:
                    shutil.copy2(str(source_path), str(tmp_target))

                # Atomic swap to final Artifacts dir
                tmp_target.replace(target_path)
            except Exception as e:
                if tmp_target.exists():
                    tmp_target.unlink()
                raise IOError(f"Failed to register Artifact: {e}") from e

        # 4. Relativize for portability
        session_root = self.manifest.path.parent
        if target_path.is_relative_to(session_root):
            rel_path = str(target_path.relative_to(session_root))
        else:
            rel_path = str(target_path.resolve())

        artifact = Artifact(
            id=artifact_id,
            name=name or source_path.stem,
            path=rel_path,
            size_bytes=target_path.stat().st_size,
            hash=artifact_hash,
            data_type=(
                data_type if data_type != "unknown" else target_path.suffix.lstrip(".")
            ),
            created_by_task_id=created_by_task_id,
            parent_artifact_ids=parent_artifact_ids or [],
            metadata=metadata or {},
        )

        self.manifest.add_artifact(artifact)

        return artifact

    def rename(self, artifact_id: str, new_name: str) -> bool:
        """Renames an Artifact AND moves the file on disk to match the new name
        (ID_Slug name strategy). Returns True if a change occurred.
        """
        artifact = self.manifest.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Cannot rename non-existent artifact ID: {artifact_id}")

        if not new_name and new_name != artifact.name:
            return False

        # 1. Calculate new path
        # {id_prefix}_{slugified_name}.{ext}
        current_path = self.dir / artifact.path
        if not current_path.exists():
            artifact.name = new_name
            return True

        new_path = self._generate_path(artifact.id, new_name, artifact.extension)

        # 2. Atomic move
        try:
            shutil.move(str(current_path), str(new_path))
        except OSError as e:
            raise RuntimeError(f"Failed to rename artifact file: {e}") from e

        # 3. Update the Manifest
        artifact.name = new_name
        artifact.path = str(new_path.relative_to(self.dir))

        return True

    def get_path(self, artifact_id: str) -> Path:
        """
        Resolved path with self-healing for "ghost files" (i.e., artifacts that
        are in the manifest but not on disk).
        """
        artifact = self.manifest.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact ID does not exist: {artifact_id}")

        # 1. External files (not in Artifacts dir)
        path = Path(artifact.path)
        if path.is_absolute():
            return path

        # 2. Happy internal file
        path = self.dir / path
        if path.exists():
            return path.resolve()

        # 3. Self-healing: Try to find the file by hash if it's missing (crash corruption)
        id_prefix = artifact.id[:8]
        candidates = list(self.dir.glob(f"{id_prefix}_*"))
        candidates = [c for c in candidates if not c.name.endswith((".tmp", ".bak"))]

        if len(candidates) == 1:
            found_path = candidates[0]
            artifact.path = str(found_path.relative_to(self.dir))
            return found_path.resolve()

        raise FileNotFoundError(f"File for Artifact ID: {artifact_id} is missing.")

    def delete(self, artifact_id: str) -> bool:
        """Delete Artifact from Manifest, then from disk if it is internal."""
        path = self.get_path(artifact_id)
        if path and path.exists():
            if self.dir in path.parents:
                path.unlink()

        return self.manifest.remove_artifact(artifact_id)

    def _calculate_hash(self, path: Path) -> str:
        """Streaming SHA256 hash calculation for a file."""
        sha256 = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # --- File access ---

    @overload
    def open(self, artifact_id: str, mode: Literal["rb"]) -> ContextManager[BinaryIO]: ...
    @overload
    def open(self, artifact_id: str, mode: Literal["r"], *, encoding: str = "utf-8") -> ContextManager[TextIO]: ...
    @contextmanager
    def open(self, artifact_id: str, mode: Literal["rb", "r"] = "rb", *, encoding: str = "utf-8"):
        """
        Context manager for secure file access.
        Resolves path (with self-healing) and yields the file handle.
        """
        path = self.get_path(artifact_id)

        try:
            if mode == "r":
                with path.open("r", encoding=encoding, errors="replace", newline="") as f:
                    yield f
            else:
                with path.open("rb") as f:
                    yield f
        except OSError as e:
            raise RuntimeError(f"OS Error accessing artifact ID: {artifact_id}\ne: {e}") from e

    def read_adapter_preview(self, artifact_id: str, limit_lines: int = 500, max_bytes: int = 50_000_000) -> tuple[str, bool]:
        """
        Fetches raw string data tailored for use in Adapters. If the file can be
        read as lines, returns a line-limited preview. Otherwise, returns whole
        file up to a byte limit.
        """
        path = self.get_path(artifact_id)
        ext = path.suffix.lower()

        with self.open(artifact_id, "r") as f:
            if ext in {".jsonl", ".ndjson", ".csv", ".tsv", ".fasta", ".faa", ".fa"}:
                iterator = islice(f, limit_lines + 1)  # grab extra to check for truncation
                lines = list(iterator)
                is_truncated = len(lines) > limit_lines
                return "".join(lines[:limit_lines]), is_truncated
            else:
                # Slicing lines would interrupt syntax
                content = f.read(max_bytes)
                is_truncated = bool(f.read(1))
                return content, is_truncated

    def read_preview_bytes(self, artifact_id: str, limit_bytes: int = 2_000) -> tuple[str, bool]:
        """
        Reads the first N bytes of a file as text.

        :returns (str, bool): (preview string, is truncated)
        """
        path = self.get_path(artifact_id)
        try:
            file_size = path.stat().st_size
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File for Artifact ID: {artifact_id} vanished!") from e

        with self.open(artifact_id, "rb") as f:
            raw = f.read(limit_bytes)

        preview_str = raw.decode("utf-8", errors="replace")

        is_truncated = False
        if file_size > limit_bytes:
            is_truncated = True
            if "\n" in preview_str:
                preview_str = preview_str.rsplit("\n", 1)[0]

        return preview_str, is_truncated

    def read_preview_lines(self, artifact_id: str, limit_lines: int = 200) -> tuple[str, bool]:
        """
        Reads the first N lines of a text file.

        :returns (str, bool): (preview string, is truncated)
        """
        lines = []
        is_truncated = False
        with self.open(artifact_id, "r") as f:
            iterator = islice(f, limit_lines + 1)  # grab extra to check for truncation
            lines = list(iterator)

            if len(lines) > limit_lines:
                is_truncated = True
                lines = lines[:limit_lines]

        return "".join(lines), is_truncated

    def load_data(self, artifact_id: str) -> Any:
        """
        Load the raw data of an Artifact into memory.
        FUTURE: Add a binary=false option to return bytes for non-text
        """
        # We need the artifact object to check the extension/type
        artifact = self.manifest.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact {artifact_id} not found.")

        path = self.get_path(artifact_id)

        # Safety Guard
        if path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError(f"Artifact {artifact.name} is too large (>100MB) to load into memory.")

        with self.open(artifact_id, "r") as f:
            return f.read()
