"""
Artifact data model definitions. An Artifact is a discrete unit of data managed 
by LoRē. It has various metadata and is tied to a physical file.
"""

from datetime import datetime, timezone
from pathlib import Path
import posixpath
from pydantic import BaseModel, Field, computed_field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from lore.core.adapters import BaseAdapter
    from lore.core.tasks import TaskDefinition


class ArtifactFile(BaseModel):
    """Metadata container for a single file within an Artifact bundle"""
    path: str  # relative to Session dir
    size_bytes: int = 0
    hash: str

    @property
    def is_remote(self) -> bool:
        """Returns True if the path is an external URI rather than a local file."""
        return self.path.startswith(("http://", "https://", "s3://", "gs://", "ftp://"))

    @property
    def filename(self) -> str:
        """Final path component e.g. `genome_report.json`"""
        if self.is_remote:
            from urllib.parse import urlparse
            clean_path = urlparse(self.path).path
            return posixpath.basename(clean_path)
        return Path(self.path).name

    @property
    def extension(self) -> str:
        """Just the file extension without the dot e.g. `json`"""
        if self.is_remote:
            name = self.filename
            return name.split(".")[-1] if "." in name else ""
        return Path(self.path).suffix.lstrip(".")


class Artifact(BaseModel):
    """
    A concrete, discrete unit of data managed by LoRē.
    May represent a single file (e.g. a FASTA file) or a logical bundle
    of related files (e.g. BAM + BAI).
    """
    id: str
    data_type: str = "unknown"  # LoRe data type tags (e.g. "fasta", "json_report")
    name: str | None = None
    files: dict[str, ArtifactFile] = Field(default_factory=dict)

    hash: str

    # Lineage
    created_by_task_id: str | None = None
    created_by_output_key: str | None = None
    parent_artifact_ids: list[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    # --- Derived properties (passthroughs to main file) ---

    @property
    def main_file(self) -> ArtifactFile | None:
        """Returns the primary path for generic engine routing"""
        return self.files.get("main")

    @property
    def path(self) -> str:
        """Returns the primary path for generic engine routing"""
        main = self.main_file
        return main.path if main else ""

    @property
    def is_remote(self) -> bool:
        """Returns True if any part of this bundle is remote"""
        return any(f.is_remote for f in self.files.values())

    @property
    def filename(self) -> str:
        """Returns the filename of the main file"""
        main = self.main_file
        return main.filename if main else ""

    @property
    def extension(self) -> str:
        """Returns the file extension of the main file"""
        main = self.main_file
        return main.extension if main else ""

    # --- UI & Size ---

    @computed_field
    @property
    def size_bytes(self) -> int:
        """Dynamically sums the size of all files in the bundle."""
        return sum(f.size_bytes for f in self.files.values())

    @computed_field
    @property
    def display_size(self) -> str:
        """Returns human-readable total size."""
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
            return f"<Artifact ID: {self.id[:4]}...>"

    # --- Pipeline Logic ---

    def get_adapters(self) -> list["BaseAdapter"]:
        """
        Convenience accessor for valid Adapters for this Artifact.
        """
        from lore.core.adapters import adapter_registry
        return adapter_registry.get_for_type(self.data_type, self.extension)

    def resolvable_types(self) -> set[str]:
        """
        Determine semantic types this Artifact can satisfy. Union of:
            - the artifact's own data_type
            - each adapter's provided_types
            - each TabularAdapter's static schema keys (declared on the class)
            - the artifact's dynamic columns/keys metadata (declared at
              materialization time by the producing task)

        The dynamic columns path lets generic adapters (JsonAdapter, CsvAdapter)
        expose the actual fields of a specific artifact to the matcher without
        a per-format subclass. FutureArtifacts have no metadata, so the dynamic
        path is silently skipped for them.
        """
        from lore.core.adapters import TabularAdapter

        resolvable = {self.data_type}
        has_table_adapter = False

        for adapter in self.get_adapters():
            resolvable.update(adapter.provided_types)

            if isinstance(adapter, TabularAdapter):
                has_table_adapter = True
                schema = getattr(adapter, "schema", None)
                if schema:
                    resolvable.update(schema.keys())

        if has_table_adapter:
            metadata = getattr(self, "metadata", None) or {}
            for declared in ("columns", "keys", "fieldnames"):
                resolvable.update(metadata.get(declared) or [])

        return resolvable

    def get_push_tasks(self) -> list["TaskDefinition"]:
        """
        Return Tasks that can accept this artifact as a valid input.
        """
        from lore.core.tasks import task_registry
        return task_registry.compatible_tasks(self.resolvable_types())
