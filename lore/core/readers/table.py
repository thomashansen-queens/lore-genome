"""
Implementation of a Reader for tabular data formats.
"""
from typing import Iterator
from .base import BaseReader

TABLE_EXTS = {"csv", "tsv"}


class TableReader(BaseReader):
    """
    Delegates the loading of new-line demarcated tabular data (CSV, TSV)
    """
    def get_metadata(self) -> dict:
        """
        Deep metadata. For tables, can't do a full row count (would require full
        file scan)
        """
        base_meta = self.get_base_metadata()
        ext = base_meta.get("extension")

        base_meta["can_stream"] = ext in {"csv", "tsv"}
        return base_meta

    def stream(self, config: dict | None = None, **kwargs) -> Iterator[str | dict]:
        """
        Memory-safe generator for large tabular files
        """
        io_config = {**(config or {}), **kwargs}
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                yield line

    def read_full(self, config: dict | None = None, **kwargs) -> list[str | dict]:
        """
        Loads the entire file into memory (or raises MemoryError if too big)
        """
        io_config = {**(config or {}), **kwargs}
        return list(self.stream(io_config))
