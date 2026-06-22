"""
Implementation of a Reader for tabular data formats.
"""
import json
from typing import Iterator
from .base import BaseReader

TABLE_EXTS = {"csv", "tsv", "parquet", "jsonl", "json"}


class TableReader(BaseReader):
    """
    Delegates the loading of tabular data
    (CSV, Parquet, JSONL, JSON array)
    """

    def get_metadata(self) -> dict:
        """
        Deep metadata. For tables, can't do a full row count (would require full
        file scan)
        """
        base_meta = self.get_base_metadata()
        ext = base_meta.get("extension")

        base_meta["can_stream"] = ext in {"csv", "tsv", "txt", "jsonl", "ndjson"}
        return base_meta

    def stream(self, config: dict | None = None, **kwargs) -> Iterator[str | dict]:
        """
        Memory-safe generator for large tabular files
        """
        io_config = {**(config or {}), **kwargs}
        ext = self.path.suffix.lower().lstrip(".")

        if ext in ("csv", "tsv", "txt"):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line

        elif ext in ("jsonl", "ndjson"):
            with open(self.path, "r", encoding="utf-8") as f:
                first_line = next((line for line in f if line.strip()), None)

            if not first_line:
                return

            try:
                parsed = json.loads(first_line)
                if not isinstance(parsed, dict):
                    raise ValueError("Valid JSON, but not a dictionary object")
            except (json.JSONDecodeError, ValueError):
                raise NotImplementedError(f"File '{self.path}' is not valid JSONL")

            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Corrupted JSONL data in '{self.path.name}': {str(e)}\nLine "
                                f"content: {line[:100]}..."
                            )

        else:
            raise NotImplementedError(f"Streaming not implemented for '{ext}'")

    def read_full(self, config: dict | None = None, **kwargs) -> list[str | dict]:
        """
        Loads the entire file into memory (or raises MemoryError if too big)
        """
        io_config = {**(config or {}), **kwargs}
        try:
            return list(self.stream(io_config))
        except NotImplementedError:
            pass  # not streamable, proceed to monolithic JSON

        text = self.path.read_text(encoding="utf-8").strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return [data]  # wrap single JSON object in a list
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]  # filter to dicts only
        except json.JSONDecodeError:
            pass  # not JSON, give up

        return []
