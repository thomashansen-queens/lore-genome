"""
An Adapter is a class that converts and flattens domain-specific raw data into
and analysis-ready format.

The BaseAdapter class contains helpers for all Adapters
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Callable, ClassVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from lore.core.artifacts import Artifact


class BaseAdapter(ABC):
    """The base class for all Adapters."""
    accepted_formats: ClassVar[set[str]] = set()  # e.g. {"json", "fasta", "csv"}
    accepted_types: ClassVar[set[str]] = set()  # e.g. {"ncbi_genome_report", "protein_sequence"}
    view_mode: ClassVar[str] = "raw"

    @property
    def name(self) -> str:
        """Adapters do not need a name, but for UI we can use the class name"""
        return self.__class__.__name__

    @property
    def provided_types(self) -> set[str]:
        """The semantic types this adapter guarantees it can produce."""
        return set()

    def provides(self, requirement: str) -> bool:
        """Universal check for what this adapter can output."""
        if requirement == "*":
            return True
        return (
            requirement in self.provided_types or
            requirement in self.accepted_formats
        )

    @abstractmethod
    def adapt(self, raw_data: Any) -> Any:
        """Convert raw data into a usable Python object (e.g. list of dicts, DataFrame, etc.)"""
        pass


# Type alias for the Field Map: Key -> (JSON Path or Extraction Function)
Schema = dict[str, str | Callable[[dict], Any]]


class TableAdapter(BaseAdapter):
    """
    The bridge between a raw file on disk and a usable Python object.
    Abstract methods:
    - load(path): How to read the raw data (override for JSON, Binary, etc)
    - schema: A mapping of standardized keys to JSON paths or extraction functions
    """
    accepted_formats: ClassVar[set[str]] = {"*"}  # e.g. {"json", "fasta", "csv"}
    accepted_types: ClassVar[set[str]] = {"*"}  # e.g. {"ncbi_genome_report", "protein_sequence"}
    view_mode: ClassVar[str] = "table"

    @property
    def provided_types(self) -> set[str]:
        """Guarantees that this adapter can provide data in a tabular format"""
        return {"dataframe"}  # For now, every adapter at least provides a table

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

        # Table-specific schema check for series extraction
        return requirement in self.schema

    # --- Adapting methods ---

    def parse(self, raw_data: Any) -> list[dict]:
        """
        Translates raw strings/bytes into a list of dicts for processing.
        Handles common cases like single dict, list of dicts, or JSONL strings.
        Override in subclass to parse other formats (e.g. binary, fasta, xml)
        """
        # 1. If given a Path, read the file content (maybe dangerous for large files)
        # if isinstance(raw_data, Path):
        #     with open(raw_data, "r", encoding="utf-8") as f:
        #         raw_data = f.read()

        # 2. Already parsed by a previous method
        if isinstance(raw_data, dict):
            return [raw_data]
        if isinstance(raw_data, list):
            return [d for d in raw_data if isinstance(d, dict)]

        # 3. Raw string or bytes
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode("utf-8")
        if not isinstance(raw_data, str):
            return []

        # 4. Try JSONL / NDJSON first
        lines = raw_data.strip().splitlines()
        is_jsonl = False
        if len(lines) > 1:
            # Heuristic: Line must be a valid JSON object
            first_line = next((l for l in lines if l.strip()), None)
            if first_line:
                try:
                    is_jsonl = isinstance(json.loads(first_line), dict)
                except json.JSONDecodeError:
                    is_jsonl = False

        if is_jsonl:
            parsed_lines = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    parsed_obj = json.loads(line)
                    if isinstance(parsed_obj, dict):
                        parsed_lines.append(parsed_obj)
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
            if parsed_lines:
                return parsed_lines

        # 5. Try parsing the entire string as one monolothic JSON object
        try:
            data = json.loads(raw_data)
            if isinstance(data, dict):
                return [data]
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            return []

        # 6. If all parsing attempts fail, return empty list
        return []

    def get_series(self, raw_data: Any, series_type: str) -> list[str] | None:
        """
        Extracts a specific column if defined in the schema or if schema is
        missing, directly from dictionary keys.
        """
        records = self.parse(raw_data)
        if not records:
            return None

        if series_type in self.schema:
            extractor = self.schema[series_type]
            return [str(self._extract_value(r, extractor)) for r in records]

        # If no schema is defined, extract from raw record keys
        if series_type in records[0]:
            return [str(r.get(series_type, None)) for r in records]

        return None

    def post_process(self, row: dict, raw_record: dict) -> dict:
        """
        Override to by-row perform logic
        """
        return row

    def apply_schema(self, records: list[dict]) -> list[dict]:
        """
        Applies the schema mapping to an already-parsed list of dicts.
        """
        if not self.schema:
            return records

        adapted_rows = []
        for rec in records:
            row = {}
            for target_col, config in self.schema.items():
                # 1. Schema specified a path with a converter function
                if isinstance(config, tuple) and len(config) == 2:
                    path, converter = config
                    raw_val = self._extract_value(rec, path)
                    row[target_col] = converter(raw_val) if raw_val is not None else None
                # 2. Schema specified a lambda/callable function
                elif callable(config):
                    row[target_col] = config(rec)
                # 3. Schema specified a string path (dot notation)
                elif isinstance(config, str):
                    row[target_col] = self._extract_value(rec, config)

            row = self.post_process(row, rec)
            adapted_rows.append(row)

        return adapted_rows

    def adapt(self, raw_data: Any) -> list[dict]:
        """
        Auto-magically convert raw data into a flat list of dicts
        """
        records = self.parse(raw_data)
        return self.apply_schema(records)

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
                if "[" in part and part.endswith("]"):
                    key, index = part[:-1].split("[", maxsplit=1)
                    val = val[key][int(index)]
                else:
                    val = val[part]
            except (KeyError, TypeError, AttributeError):
                return None
        return val

    # --- Outputs ---

    def serialize(self, records: list[dict], extension: str = "json") -> str:
        """
        Translates a list of dictionaries back into raw string format. Override
        this in subclasses for other formats (e.g. FASTA, XML, etc.)
        """
        if extension in ("jsonl", "ndjson"):
            return "\n".join(json.dumps(r) for r in records) + "\n"
        return json.dumps(records, indent=2)

    def to_dataframe(self, raw_data_or_path: Any) -> "pd.DataFrame":
        """
        Converts adapted data directly into a Pandas DataFrame.
        Accepts raw strings/bytes OR a file Path. 

        NOTE: This default implementation loads the entire dataset into memory 
        before conversion. Subclasses dealing with massive files (like NCBI 
        JSONL) should override this method to stream the file line-by-line.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel
        is_path = isinstance(raw_data_or_path, Path)
        is_path_string = isinstance(raw_data_or_path, str) and len(raw_data_or_path) < 1024

        if is_path or is_path_string:
            try:
                path_obj = Path(raw_data_or_path)
                if path_obj.is_file():
                    with open(path_obj, "r", encoding="utf-8") as f:
                        raw_data_or_path = f.read()
            except (OSError, IOError):
                pass  # Not a valid file path, treat as raw data

        records = self.adapt(raw_data_or_path)
        return pd.DataFrame(records)

    # --- Helper Methods for Data Cleaning and Type Conversion ---

    def safe_int(self, val: Any) -> int | None:
        try:
            if isinstance(val, str):
                val = val.replace(",", "")  # Comma separators intefere with casting
            return int(val)
        except (ValueError, TypeError, AttributeError):
            return None

    def safe_float(self, val: Any) -> float | None:
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


class ImageAdapter(BaseAdapter):
    """
    The bridge between image data and a UI-viewable format
    """
    accepted_formats: ClassVar[set[str]] = set() # e.g. {"svg", "png", "jpg", "jpeg", "gif"} 
    accepted_types: ClassVar[set[str]] = set() # e.g. {"genome_map", "phylo_tree", "protein_structure"}
    view_mode: ClassVar[str] = "image"

    @property
    def provided_types(self) -> set[str]:
        return {"image"}

    @abstractmethod
    def adapt(self, raw_data: Any) -> str | bytes:
        """
        Often will just return the raw payload (e.g. SVG string or PNG bytes),
        but can be overridden to perform transformations or optimizations (e.g. 
        resizing)
        """
        pass


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

    def register(self, adapter: BaseAdapter) -> None:
        """Should be called at app startup to register new Adapters"""
        if not any(adapter.accepted_types or adapter.accepted_formats):
            raise ValueError(f"Adapter {adapter} accepts no types nor formats")
        if adapter.__class__.__name__ in self._adapters:
            raise ValueError(f"Adapter with key '{adapter.__class__.__name__}' is already registered.")
        self._adapters[adapter.__class__.__name__] = adapter

    def get_adapters(self, artifact: "Artifact", must_provide: str = "*") -> list[BaseAdapter]:
        """
        Finds all adapters that can bridge the gap between this 
        Physical Artifact and the Task's Logical Requirement.
        """
        matches = []
        for adapter in self._adapters.values():
            # 1. Artifact compatibility: Can I read this?
            format_match = "*" in adapter.accepted_formats or artifact.extension in adapter.accepted_formats
            type_match = "*" in adapter.accepted_types or artifact.data_type in adapter.accepted_types
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
            if artifact.extension in a.accepted_formats:
                score += 1.0
            if artifact.data_type in a.accepted_types:
                score += 1.1
            return score

        matches.sort(key=sort_score, reverse=True)
        return matches

adapter_registry = AdapterRegistry()
