"""
The BaseAdapter class defines the core interface and default behaviours for 
all Adapters in LoRē.
"""
from abc import ABC
import hashlib
from typing import Any, ClassVar, Iterator
from pydantic import BaseModel, Field


class AdapterPreview(BaseModel):
    """Strict payload returned by all Adapters for UI rendering."""
    data: Any = Field(description="The actual data (records, SVG string, etc.)")
    view_mode: str = Field(description="How the UI should render this data (e.g. 'table', 'svg', 'text')")
    adapter_name: str = Field(description="The name of the Adapter that generated this preview")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Context about the data (e.g. total_rows, file_eof_hit)",
    )


class BaseAdapter(ABC):
    """
    Translates physical Artifacts (files) into structured in-memory representations.
    Handles all format-specific I/O to keep Execution Handlers agnostic.
    """
    accepted_formats: ClassVar[set[str]] = set()  # e.g. {"json", "fasta", "csv"}
    accepted_types: ClassVar[set[str]] = set()  # e.g. {"ncbi_genome_report", "protein_sequence"}
    view_mode: ClassVar[str] = "raw"
    version: ClassVar[str] = "1.0.0"  # Increment when parsing logic changes for hash purposes

    @property
    def name(self) -> str:
        """Adapters do not need a name, but for UI we can use the class name"""
        return self.__class__.__name__

    @classmethod
    def get_hash(cls) -> str:
        """
        Deterministic hash for Manifest provenance. Tracks which version 
        of the adapter was used to process the data.
        """
        fingerprint = f"{cls.__name__}_v{cls.version}".encode('utf-8')
        return hashlib.md5(fingerprint).hexdigest()[:8]

    @property
    def provided_types(self) -> set[str]:
        """The semantic types this adapter guarantees it can produce."""
        return set()

    def provides(self, requirement: str) -> bool:
        """
        Universal check for what this adapter can output. NOTE: Adapters are 
        transitive. If it accepts a data_type, we will assume it provides that 
        data_type, albeit in an adapted format.
        """
        if requirement == "*":
            return True
        return (
            requirement in self.provided_types or
            requirement in self.accepted_types
        )

    # --- Config helper ---

    def _prepare_config(self, config: dict | None = None, **kwargs) -> dict:
        """
        Standardizes configuration across all adapters. Flattens 'options' dict.
        Merges explicit config dicts with kwargs and normalizes common aliases.
        """
        config = config or {}
        options = config.pop("options", {})
        merged = {**config, **options, **kwargs}

        # Normalize aliases (just extension for now)
        if "extension" in merged or "ext" in merged:
            ext = str(merged.get("extension", "") or merged.get("ext", "")).lower()
            merged["ext"] = ext
            merged["extension"] = ext

        return merged

    # --- Data translation ---

    def parse(self, raw_data: Any, config: dict | None = None, **kwargs) -> Any:
        """
        Convert raw bytes/string into native Python objects without applying any
        lossy schemas or transformations.
        """
        return raw_data

    def parse_stream(
        self,
        raw_stream: Iterator[Any],
        config: dict | None = None,
        **kwargs,
    ) -> Iterator[Any]:
        """
        Convert a stream of raw data into Python objects. Yields records one at 
        a time.
        """
        yield from raw_stream

    def adapt(self, raw_data: Any, config: dict | None = None, **kwargs) -> Any:
        """Translate a full block of raw data into adapted format"""
        if isinstance(raw_data, list):
            return [self.adapt_record(r, config, **kwargs) for r in raw_data]
        return self.adapt_record(raw_data, config, **kwargs)

    def adapt_record(self, record: Any, config: dict | None = None, **kwargs) -> Any:
        """Adapts a single item. Override to apply schemas/transformations"""
        return record

    def adapt_stream(self, raw_stream: Iterator[Any], config: dict | None = None, **kwargs) -> Iterator[Any]:
        """Adapt a stream of records maintaining statefulness"""
        return (self.adapt_record(r, config, **kwargs) for r in raw_stream)

    # --- Render and output methods ---

    def serialize(self, records: Any, config: dict | None = None, **kwargs) -> str:
        """Turns adapted data into raw string format. Override as needed."""
        return str(records)

    def preview(
        self,
        raw_data: Any,
        io_metadata: dict,
        config: dict | None = None,
        **kwargs,
    ) -> AdapterPreview:
        """
        Packages data and IO metadata into UI-friendly format.
        io_metadata comes from the Reader (io layer)
        """
        adapted_data = self.adapt(raw_data, config, **kwargs)

        final_metadata = io_metadata.copy()

        if final_metadata.get("total_rows") is None and isinstance(adapted_data, list):
            if io_metadata.get("file_eof_hit", True):
                final_metadata["total_rows"] = len(adapted_data)

        return AdapterPreview(
            data=adapted_data,
            view_mode=self.view_mode,
            adapter_name=self.name,
            metadata=final_metadata,
        )
