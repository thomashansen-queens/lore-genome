"""
An Adapter is a class that converts and flattens domain-specific raw data into
and analysis-ready format.

The BaseAdapter class contains helpers for all Adapters
"""

from abc import ABC
import csv
import hashlib
import itertools
import json
import logging
from typing import Any, Callable, ClassVar, Iterator, TypeVar
from typing import TYPE_CHECKING
from pydantic import BaseModel, Field

from lore.core.artifacts import Artifact

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class AdapterPreview(BaseModel):
    """Strict payload returned by all Adapters for UI rendering."""
    data: Any = Field(description="The actual data (records, SVG string, etc.)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Context about the data")


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

    # --- Data translation ---

    def adapt(self, raw_data: Any, config: dict | None = None) -> Any:
        """Translate a full block of raw data into adapted format"""
        if isinstance(raw_data, list):
            return [self.adapt_record(r, config) for r in raw_data]
        return self.adapt_record(raw_data, config)

    def adapt_record(self, record: Any, config: dict | None = None) -> Any:
        """Adapts a single item. Override to apply schemas/transformations"""
        return record

    def adapt_stream(self, raw_stream: Iterator[Any], config: dict | None = None) -> Iterator[Any]:
        """Adapt a stream of records maintaining statefulness"""
        return (self.adapt_record(r, config) for r in raw_stream)

    # --- Render and output methods ---

    def serialize(self, records: Any, extension: str = "json") -> str:
        """Turns adapted data into raw string format. Override as needed."""
        return str(records)

    def preview(self, raw_data: Any, io_metadata: dict, config: dict | None = None) -> AdapterPreview:
        """
        Packages data and IO metadata into UI-friendly format.
        """
        adapted_data = self.adapt(raw_data, config)

        final_metadata = {
            **io_metadata,
            "view_mode": self.view_mode,
            "adapter_name": self.name,
        }

        if final_metadata.get("total_rows") is None and isinstance(adapted_data, list):
            final_metadata["total_rows"] = len(adapted_data)

        return AdapterPreview(
            data=adapted_data,
            metadata=final_metadata,
        )


class RawAdapter(BaseAdapter):
    """
    Passthrough adapter for plain text and unstructured formats.
    Returns raw content as-is for display.
    """
    accepted_formats: ClassVar[set[str]] = {"txt", "raw", "info", "nfo", "log", "md", "*"}
    accepted_types: ClassVar[set[str]] = {"*"}
    view_mode: ClassVar[str] = "raw"
    version: ClassVar[str] = "1.0.0"

    @property
    def provided_types(self) -> set[str]:
        return {"raw", "text"}

    def adapt(self, raw_data: Any, config: dict | None = None) -> str:
        """Ensures the raw data is returned as a string for display."""
        # 1. Already a string
        if isinstance(raw_data, str):
            return raw_data

        # 2. A list (i.e. lines of text from a reader)
        if isinstance(raw_data, list):
            if not raw_data:
                return ""

            if isinstance(raw_data[0], str):
                return "\n".join(raw_data)

            # 3. A list of parsed objects (e.g., dicts from a JSON reader)
            try:
                return json.dumps(raw_data, indent=2)
            except (TypeError, ValueError):
                # No JSON serialization possible
                return "\n".join(str(item) for item in raw_data)

        # 3. Single parsed object
        try:
            return json.dumps(raw_data, indent=2)
        except (TypeError, ValueError):
            return str(raw_data)


# Type alias for the Field Map: Key -> (JSON Path or Extraction Function)
Schema = dict[str, str | Callable[[dict], Any]]


class TableAdapter(BaseAdapter):
    """
    The bridge between a raw file on disk and a usable tabular Python object.
    Native support for JSON, JSONL, CSV, and TSV.
    TODO: Refactor TableAdapter to separate parse() from adapt(); eliminate Refused Bequest
    smell seen in e.g. FastaAdapter
    """
    accepted_formats: ClassVar[set[str]] = {"json", "ndjson", "jsonl", "csv", "tsv"}
    accepted_types: ClassVar[set[str]] = {"*"}  # e.g. {"ncbi_genome_report", "protein_sequence"}
    view_mode: ClassVar[str] = "table"
    version: ClassVar[str] = "1.0.0"

    @property
    def provided_types(self) -> set[str]:
        """Guarantees that this adapter can provide data in a tabular format"""
        return {"table", "dataframe"}

    @property
    def schema(self) -> Schema:
        """
        Define schema here.
        Format: {"key": "path.to[0].data"} OR {"key": ("path.to[0].data", type_converter)} OR {"key": lambda r: ...}
        Is able to navigate dot notations and nested lists.
        """
        return {}

    @property
    def fields(self) -> list[dict[str, str]]:
        """
        Helper for to construct column definitions and present as a table. If
        schema is undefined, this will be empty!
        """
        return [{"name": k, "label": k.replace("_", " ").title()} for k in self.get_keys()]

    def get_keys(self) -> list[str]:
        """
        Return keys strictly defined in the schema
        """
        return list(self.schema.keys())

    def provides(self, requirement: str) -> bool:
        """
        Can this Adapter provide a specific 'slice' or 'type' of data?
        """
        if super().provides(requirement):
            return True

        if hasattr(self, "schema") and self.schema:
            return requirement in self.schema

        return False

    # --- Adapting methods ---

    def adapt(self, raw_data: Any, config: dict | None = None) -> list[dict]:
        """
        Intercepts monolithic data. Handles headerless text, string lists, and 
        mangled dicts from readers.
        """
        if not raw_data:
            return []

        # 1. Headerless text lines
        if isinstance(raw_data, list) and isinstance(raw_data[0], str):
            # Infer delimiter from config extension, default to comma
            ext = config.get("ext", "") if config else ""
            delimiter = "\t" if ext in ("tsv", "txt") else ","

            # csv.DictReader consumes the list of strings and yields dicts
            dict_reader = csv.DictReader(raw_data, delimiter=delimiter)

            return [self.adapt_record(row, config) for row in dict_reader]

        # 2. Already a list of dicts, no special handling
        return super().adapt(raw_data, config)

    def adapt_record(self, record: dict, config: dict | None = None) -> dict:
        """Applies the schema mapping to a single record"""
        if not self.schema:
            return record

        row = {}
        for target_col, schema_config in self.schema.items():
            # 1. Schema specified a path with a converter function
            if isinstance(schema_config, tuple) and len(schema_config) == 2:
                path, converter = schema_config
                raw_val = self._extract_value(record, path)
                row[target_col] = converter(raw_val) if raw_val is not None else None
            # 2. Schema specified a lambda/callable function
            elif callable(schema_config):
                row[target_col] = schema_config(record)
            # 3. Schema specified a string path (dot notation)
            elif isinstance(schema_config, str):
                row[target_col] = self._extract_value(record, schema_config)

        return self.post_process(row, record)

    def adapt_stream(self, raw_stream: Iterator[Any], config: dict | None = None) -> Iterator[dict]:
        """Allows for stateful adaptation across a stream of records"""
        # 1. Peek at first item, then put it back so the stream is intact
        try:
            first_item = next(raw_stream)
        except StopIteration:
            return  # Empty stream
        full_stream = itertools.chain([first_item], raw_stream)

        # 2. Text stream handling
        if isinstance(first_item, str):
            ext = config.get("ext", "") if config else ""
            delimiter = "\t" if ext in ("tsv", "txt") else ","
            dict_stream = csv.DictReader(full_stream, delimiter=delimiter)
            for row in dict_stream:
                yield self.adapt_record(row, config)

        # 3. Already a stream of dicts, no special handling
        else:
            for row in full_stream:
                yield self.adapt_record(row, config)

    def post_process(self, row: dict, raw_record: dict) -> dict:
        """Override to by-row perform logic"""
        return row

    def _extract_value(self, record: dict, extractor: str | Callable[[dict], Any]) -> Any:
        """
        Helper to pull data using dot-notation ("organism.tax_id") with list 
        indexing allowed ("assembly_stats.contigs[0].length"), or via a custom 
        function (lambda or Callable)
        """
        # Case A: Function extraction
        if callable(extractor):
            try:
                return extractor(record)
            except (KeyError, AttributeError):
                return None

        # Case B: Dot notation path
        val = record
        for part in extractor.split("."):
            try:
                if "[" in part:
                    # Split key and slice (e.g. [5] or [1][2])
                    key, _, brackets = part.partition("[")
                    if key:
                        val = val[key]
                    # Re-attach the bracket, then check for multiple slices
                    for index_str in ("[" + brackets).split("["):
                        if not index_str:
                            continue
                        # After split e.g. "["0]", "2]"] -> [0, 2]
                        val = val[int(index_str.strip("]"))]
                else:
                    val = val[part]
            except (KeyError, TypeError, AttributeError):
                return None
        return val

    # --- Outputs ---

    def preview(self, raw_data, io_metadata, config=None):
        """Override to inject config transformations (e.g. sorting)"""
        result = super().preview(raw_data, io_metadata, config)

        # 1. Inject tabular metadata
        if isinstance(result.data, list) and result.data and isinstance(result.data[0], dict):
            result.metadata["columns"] = list(result.data[0].keys())

        # 2. Apply view-level config transformations (e.g. sorting)
        view_state = (config or {}).get("view_state", {})
        sort_by = view_state.get("sort_by")

        if sort_by and isinstance(result.data, list):
            sort_asc = view_state.get("sort_asc", True)
            # Sort: Pushes None to end, strinigifies values to prevent type errors
            result.data.sort(
                key=lambda r: (
                    1 if r.get(sort_by) is None else 0,
                    str(r.get(sort_by)).lower() if r.get(sort_by) is not None else ""),
                reverse=not sort_asc,
            )

        return result

    def to_dataframe(self, raw_data: Any, config: dict | None = None) -> "pd.DataFrame":
        """
        Converts adapted data directly into a Pandas DataFrame.
        Accepts raw strings/bytes OR a file Path. 
        """
        import pandas as pd
        records = self.adapt(raw_data, config=config)
        df = pd.DataFrame(records)
        df = df.convert_dtypes()
        return df

    def serialize(self, records: list[dict], extension: str = "json") -> str:
        """
        Translates a list of dictionaries back into raw string format. Override
        this in subclasses for other formats (e.g. FASTA, XML, etc.)
        """
        if extension in ("jsonl", "ndjson"):
            return "\n".join(json.dumps(r) for r in records) + "\n"
        return json.dumps(records, indent=2)

    def get_series(self, raw_data: Any, series_type: str) -> list[str] | None:
        """
        Extracts a specific column if defined in the schema or if schema is
        missing, directly from dictionary keys.
        """
        records = self.adapt(raw_data)
        if not records:
            return None

        if series_type in records[0]:
            return [str(r.get(series_type, None)) for r in records]

        return None

    # --- Helper Methods for Data Cleaning and Type Conversion ---

    def safe_int(self, val: Any) -> int | None:
        """Converts to int, stripping commas, returns None if unable to convert."""
        try:
            if isinstance(val, str):
                val = val.replace(",", "")
            return int(val)
        except (ValueError, TypeError, AttributeError):
            return None

    def safe_float(self, val: Any) -> float | None:
        """Converts to float if possible"""
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def safe_date(self, val: Any) -> str | None:
        """Standardize to ISO YYYY-MM-DD"""
        if not val:
            return None
        try:
            return str(val).split("T", maxsplit=1)[0][:10]  # Handle datetime strings and ensure max length of 10
        except (ValueError, TypeError, AttributeError):
            return None

    def safe_str(self, val: Any) -> str | None:
        """Only converts to string if value is not None or empty, otherwise returns None"""
        if not val:
            return None
        return str(val)


class ImageAdapter(BaseAdapter):
    """
    The bridge between image data and a UI-viewable format
    """
    accepted_formats: ClassVar[set[str]] = {"png", "jpg", "jpeg"}  # e.g. {"svg", "png", "jpg", "jpeg", "gif"} 
    accepted_types: ClassVar[set[str]] = set()  # e.g. {"genome_map", "phylo_tree", "protein_structure"}
    view_mode: ClassVar[str] = "image"

    @property
    def provided_types(self) -> set[str]:
        return {"image"}

    def adapt(self, raw_data: Any, config: dict | None = None) -> Any:
        """
        Defaults to raw payload (e.g. SVG string or PNG bytes). Subclasses can 
        override to perform transformations or optimizations (e.g. resizing)
        """
        return raw_data


C = TypeVar("C", bound=type["BaseAdapter"])


class AdapterRegistry:
    """
    Global registry for Adapters. Tasks can query this to find an Adapter for
    a given Artifact and data type.
    """
    def __init__(self):
        self._adapters: dict[str, BaseAdapter] = {}

    def __getitem__(self, key: str) -> BaseAdapter:
        """Allows dict-like access to Adapter definitions"""
        if key not in self._adapters:
            raise KeyError(f"Adapter with key '{key}' not found.")
        return self._adapters[key]

    def __call__(self, cls: None = None) -> Callable[[C], C]:
        """
        Class decorator for registering Adapters.
        Usage:
            @adapter_registry
            class MyAdapter(BaseAdapter):
                ...
        """
        def wrapper(adapter_cls: C) -> C:
            if not issubclass(adapter_cls, BaseAdapter):
                raise TypeError(
                    f"Cannot register {adapter_cls.__name__}. "
                    f"Adapters must inherit from BaseAdapter."
                )
            # Instantiate the adapter and register the instance
            self.register(adapter_cls())
            return adapter_cls

        # If used as a naked decorator (@adapter_registry) without arguments (@adapter_registry())
        if cls is None:
            return wrapper
        return wrapper(cls)

    def register(self, adapter: BaseAdapter) -> None:
        """Should be called at app startup to register new Adapters"""
        if not (adapter.accepted_types or adapter.accepted_formats):
            raise ValueError(f"Adapter {adapter} accepts no types nor formats")
        if adapter.__class__.__name__ in self._adapters:
            logging.debug(f"Overwriting existing adapter with key '{adapter.__class__.__name__}'")
            # raise ValueError(f"Adapter with key '{adapter.__class__.__name__}' is already registered.")
        self._adapters[adapter.__class__.__name__] = adapter

    def get(self, key: str) -> BaseAdapter | None:
        """Safe get method that returns None if adapter is not found"""
        return self._adapters.get(key, None)

    def get_adapters_by_artifact(self, artifact: Artifact, must_provide: str = "*") -> list[BaseAdapter]:
        """
        Finds all adapters that can bridge the gap between this Artifact and the
        Task's data requirements
        """
        return self.get_adapters_by_type(artifact.data_type, artifact.extension, must_provide)

    def get_adapters_by_type(self, data_type: str, extension: str, must_provide: str = "*") -> list[BaseAdapter]:
        """
        Finds all adapters that can bridge the gap between this 
        Physical Artifact and the Task's Logical Requirement.
        """
        matches = []
        for adapter in self._adapters.values():
            # 1. Artifact compatibility: Can I read this?
            format_match = "*" in adapter.accepted_formats or extension in adapter.accepted_formats
            type_match = "*" in adapter.accepted_types or data_type in adapter.accepted_types
            if not (format_match and type_match):
                continue

            # 2. Logical compatibility: Can I provide this?
            if adapter.provides(must_provide):
                matches.append(adapter)

        # 3. Optimization: Sort by 'Expertise'
        def sort_score(a):
            """
            Prioritize Adapters that explicitly provide rather than '*' with 
            semantic type (1.1) scored higher than file format (1.0)
            """
            score = 0
            if extension in a.accepted_formats:
                score += 1.0
            if data_type in a.accepted_types:
                score += 1.1
            return score

        matches.sort(key=sort_score, reverse=True)
        return matches


# Instantiate the global registry and register base adapters.
adapter_registry = AdapterRegistry()

adapter_registry.register(RawAdapter())
adapter_registry.register(TableAdapter())
adapter_registry.register(ImageAdapter())
