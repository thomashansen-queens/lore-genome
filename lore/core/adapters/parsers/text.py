"""
Show the raw string content of a data file.
"""
import json
from typing import Any, ClassVar

from .. import BaseAdapter


class TextAdapter(BaseAdapter):
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

    def adapt(self, raw_data: Any, config: dict | None = None, **kwargs) -> str:
        """Ensures the raw data is returned as a string for display."""
        kwconfig = self._prepare_config(config, **kwargs)

        # 1. Bytes or string: return as-is
        if isinstance(raw_data, bytes):
            return raw_data.decode("utf-8", errors="replace")

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

        # 4. Single parsed object
        try:
            return json.dumps(raw_data, indent=2)
        except (TypeError, ValueError):
            return str(raw_data)

    # --- Output methods ---

    def serialize(self, records: Any, config: dict | None = None, **kwargs) -> str:
        """Writes adapted text back to a single stirng."""
        if isinstance(records, str):
            return records

        return self.adapt(records, config, **kwargs)
