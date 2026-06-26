"""
Reader for JSON / JSON Lines files using Python's standard 'json' library.

Monolithic JSON files cannot be streamed here; a json_stream reader is
provided in the included 'builtins' plugins, which uses the third-party
ijson library.
"""
from collections.abc import Iterator
import json

from .base import BaseReader

JSON_EXTS = {"json", "jsonl", "ndjson"}
_LINE_DELIMITED = {"jsonl", "ndjson"}


class JsonReader(BaseReader):
    """
    Stdlib JSON reader. Streams JSONL/NDJSON line-by-line; loads monolithic JSON
    whole (the stdlib parser is all-or-nothing).
    """
    def get_metadata(self) -> dict:
        base_meta = self.get_base_metadata()
        base_meta["can_stream"] = base_meta.get("extension") in _LINE_DELIMITED
        return base_meta

    def stream(self, config: dict | None = None, **kwargs) -> Iterator[dict]:
        """
        Yield parsed records from a JSONL/NDJSON file. Monolithic JSON is not
        streamable by the stdlib parser, so this raises NotImplementedError and
        callers fall back to read_full().
        """
        ext = self.path.suffix.lower().lstrip(".")
        if ext not in _LINE_DELIMITED:
            raise NotImplementedError(
                f"{self.__class__.__name__} cannot stream monolithic JSON; "
                "use read_full() instead."
            )

        # Validate the first non-blank line really is a JSON object
        with open(self.path, "r", encoding="utf-8") as f:
            first_line = next((line for line in f if line.strip()), None)
        if not first_line:
            return
        try:
            if not isinstance(json.loads(first_line), dict):
                raise ValueError("Valid JSON, but not a dictionary object")
        except (json.JSONDecodeError, ValueError):
            raise NotImplementedError(f"File '{self.path}' is not valid {ext}")

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Corrupted {ext} data in '{self.path.name}': {str(e)}\n"
                            f"Line content: {line[:100]}..."
                        )

    def read_full(self, config: dict | None = None, **kwargs) -> list[dict]:
        """
        Load the whole file into memory as a list of records. JSONL/NDJSON drains
        the line stream; monolithic JSON is parsed whole and normalized to a list
        (a single object becomes a one-item list; non-dict array items dropped).
        """
        ext = self.path.suffix.lower().lstrip(".")
        if ext in _LINE_DELIMITED:
            return list(self.stream(config, **kwargs))

        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        return []
