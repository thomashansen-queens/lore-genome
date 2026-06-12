"""
Tabular data adapter.
Used by the Materializer to convert raw file contents into structured data
and optionally extract specific series/columns for use in tasks.
"""
from typing import Any, Callable, ClassVar, Iterator, TYPE_CHECKING

from .base import AdapterPreview, BaseAdapter

if TYPE_CHECKING:
    import pandas as pd


# Type alias for the Field Map: Key -> (JSON Path or Extraction Function)
Schema = dict[str, str | Callable[[dict], Any]]


class TabularAdapter(BaseAdapter):
    """
    Adapts data into a list of dictionaries (records) for tabular display.
    """
    view_mode: ClassVar[str] = "table"

    @property
    def provided_types(self) -> set[str]:
        """Guarantees that this adapter can provide data in a tabular format"""
        return {"table", "dataframe"}

    @property
    def schema(self) -> Schema:
        """
        Override this method to define schema here.
        Maps target to column name to one of:
            {"key": "path.to[0].data"}
            {"key": ("path.to[0].data", type_converter)}
            {"key": lambda r: ...}

        Where type_converter is a function that to coerce data types. This class 
        provides some converters that return None if conversion fails:
            safe_int, safe_float, safe_date, safe_str
        Example:
            {"release_date": ("assembly_info.release_date", self.safe_date)}
        """
        return {}

    def provides(self, requirement: str) -> bool:
        """
        Can this Adapter provide a specific 'slice' or 'type' of data?
        """
        if super().provides(requirement):
            return True

        if hasattr(self, "schema") and self.schema:
            return requirement in self.schema

        return False

    def get_keys(self) -> list[str]:
        """
        Return keys strictly defined in the schema
        """
        return list(self.schema.keys())

    @property
    def fields(self) -> list[dict[str, str]]:
        """
        Helper for to construct column definitions and present as a table. If
        schema is undefined, this will be empty!
        """
        return [{"name": k, "label": k.replace("_", " ").title()} for k in self.get_keys()]

    # --- Adapting methods ---

    def parse(self, raw_data: Any, config: dict | None = None, **kwargs) -> Any:
        """
        By default, TabularAdapter assumes the raw data is already in a list of
        dicts format (e.g. JSON). Override to parse other raw formats.
        """
        if isinstance(raw_data, list) and (not raw_data or isinstance(raw_data[0], dict)):
            return raw_data

        raise TypeError(
            f"TabularAdapter expects parsed data to be a list of dicts, but got {type(raw_data)}. "
            f"If you are passing raw strings/bytes, ensure you override parse() to convert "
            f"to a a list of dicts."
        )

    def adapt(self, raw_data: Any, config: dict | None = None, **kwargs) -> list[dict]:
        """
        1. Parse the raw data into a list of dicts (lossless)
        2. Apply the loss schema/transformation (lossy)
        """
        parsed_records = self.parse(raw_data, config, **kwargs)
        if not parsed_records:
            return []
        return [self.adapt_record(row, config, **kwargs) for row in parsed_records]

    def adapt_record(self, record: dict, config: dict | None = None, **kwargs) -> dict:
        """Applies the schema mapping to a single dict-like record"""
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

    def adapt_stream(
        self,
        raw_stream: Iterator[Any],
        config: dict | None = None,
        **kwargs,
    ) -> Iterator[dict]:
        """
        Allows for stateful adaptation across a stream of records.
        1. Parse the raw data into a list of dicts (lossless)
        2. Apply the loss schema/transformation (lossy)

        Yields: An iterator that yields adapted records from an input stream.
        """
        parsed_stream = self.parse_stream(raw_stream, config, **kwargs)
        for row in parsed_stream:
            yield self.adapt_record(row, config, **kwargs)

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

    def preview(
        self,
        raw_data: Any,
        io_metadata: dict[str, Any],
        config: dict | None = None,
        **kwargs,
    ) -> AdapterPreview:
        """Override to inject config transformations (e.g. sorting)"""
        result = super().preview(raw_data, io_metadata, config, **kwargs)
        kwconfig = self._prepare_config(config, **kwargs)

        # 1. Inject tabular metadata
        if isinstance(result.data, list) and result.data and isinstance(result.data[0], dict):
            result.metadata["columns"] = list(result.data[0].keys())

        # 2. Apply view-level config transformations (e.g. sorting)
        view_state = kwconfig.get("view_state", {})
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

    def get_series(
        self,
        raw_data: Any,
        series_type: str,
        config: dict | None = None,
        **kwargs,
    ) -> list[str] | None:
        """
        Extracts a 1D series of data (e.g. a column) from the tabulated data.
        """
        records = self.adapt(raw_data, config=config, **kwargs)
        if not records or not isinstance(records[0], dict):
            return None

        if series_type in records[0]:
            return [str(r.get(series_type)) for r in records if r.get(series_type) is not None]
        return None

    def to_dataframe(self, raw_data: Any, config: dict | None = None, **kwargs) -> "pd.DataFrame":
        """
        Converts adapted data directly into a Pandas DataFrame.
        Accepts raw strings OR a file Path.
        """
        import pandas as pd
        records = self.adapt(raw_data, config=config, **kwargs)
        df = pd.DataFrame(records)
        df = df.convert_dtypes()
        return df

    def serialize(self, records: list[dict], config: dict | None = None, **kwargs) -> str:
        """
        Translates a list of dictionaries or strings back into raw string format. Override
        this in subclasses for other formats (e.g. FASTA, XML, etc.)
        """
        raise NotImplementedError(
            "Serialization not implemented for TabularAdapter. Override in subclass if needed."
        )

    # --- Helper Methods for Data Cleaning and Type Conversion ---

    @staticmethod
    def safe_int(val: Any) -> int | None:
        """Converts to int, stripping commas, returns None if unable to convert."""
        try:
            if isinstance(val, str):
                val = val.replace(",", "")
            return int(val)
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def safe_float(val: Any) -> float | None:
        """Converts to float if possible"""
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def safe_date(val: Any) -> str | None:
        """Standardize to ISO YYYY-MM-DD"""
        if not val:
            return None
        try:
            return str(val).split("T", maxsplit=1)[0][:10]  # Handle datetime strings and ensure max length of 10
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def safe_str(val: Any) -> str | None:
        """Only converts to string if value is not None or empty, otherwise returns None"""
        if not val:
            return None
        return str(val)
