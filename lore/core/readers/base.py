"""
Core abstract Reader class. Turns 1s and 0s into useful data structures.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator


# Standalone fallback for the preview RAM ceiling.
# `max_ram_bytes` in config overrides this (sourced from settings.preview_max_ram_bytes)
DEFAULT_MAX_RAM_BYTES = 50 * 1024 * 1024  # 50 MB


class BaseReader(ABC):
    """
    Abstract foundation.
    """
    def __init__(self, path: Path):
        self.path = path

    def get_base_metadata(self) -> dict:
        """
        Universal metadata available to all files without opening them.
        """
        if not self.path.exists():
            return {
                "file_size_bytes": 0,
                "extension": self.path.suffix.lower().lstrip("."),
                "exists": False,
                "reader": self.__class__.__name__,
            }

        return {
            "file_size_bytes": self.path.stat().st_size,
            "extension": self.path.suffix.lower().lstrip("."),
            "exists": True,
            "reader": self.__class__.__name__,
        }

    @abstractmethod
    def get_metadata(self) -> dict:
        """
        Deep metadata. Forces subclasses to peek inside the file if necessary
        (e.g., counting rows for a CSV, or getting dimensions for an Image).
        """
        pass

    def stream(self, config: dict | None = None, **kwargs) -> Iterator[Any]:
        """
        Yield the file's records (lines, dicts, byte-chunks) lazily.

        By default, raises NotImplementedError so a reader for a monolithic
        format (e.g. JSON parsed with the standard `json` lib) can leave streaming
        un-overridden and provide read_full() instead. preview() detects the
        missing stream method and falls back to the whole-file read automatically.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming; "
            "implement read_full() instead."
        )

    def read_full(self, config: dict | None = None, **kwargs) -> Any:
        """
        Load the entire file into memory. Defaults to draining stream() iterator.
        Override for monolithic formats (e.g. a JSON array, an image).
        """
        return list(self.stream(config, **kwargs))

    def _record_size(self, record: Any) -> int:
        """
        Rough in-memory size of one streamed record, used to enforce the preview
        byte ceiling. Defaults to string length; override for binary records.
        """
        return len(str(record))

    def preview(
        self,
        peek_limit: int = 100,
        config: dict | None = None,
        **kwargs,
    ) -> tuple[Any, dict]:
        """
        Universal early-stop preview algorithm.

        Subclasses need only provide how to stream() one record (or, for
        monolithic formats, read_full()). This (BaseReader) takes care of peek/
        eager/RAM-limit behaviour. Presents the UI with what it needs to report
        preview completeness and whether or not the file is too large to preview.

        - strategy "peek" (default): stop the instant the window is full — fast,
          but the grand total stays unknown (total_rows is None).
        - strategy "full"/"eager": stream to EOF, counting every record, but only
          hold up to the RAM ceiling. Basis for "top N of M".
        - non-streamable readers (stream() raises NotImplementedError): fall back
          to a whole-file read_full(), then slice the display window.
        """
        io_config = {**(config or {}), **kwargs}
        strategy = io_config.get("strategy", "peek")
        max_ram_bytes = io_config.get("max_ram_bytes", DEFAULT_MAX_RAM_BYTES)

        data: list = []
        eof_hit = True
        rows_read = 0
        ram_bytes = 0
        ram_limit_hit = False

        try:
            for record in self.stream(io_config):
                rows_read += 1

                # A. Peek: bail once peek limit is reached.
                if strategy == "peek" and rows_read > peek_limit:
                    eof_hit = False
                    break

                # B. Otherwise keep streaming (to count), but only keep up to RAM ceiling.
                if not ram_limit_hit:
                    size = self._record_size(record)
                    if ram_bytes + size > max_ram_bytes:
                        ram_limit_hit = True
                    else:
                        data.append(record)
                        ram_bytes += size

            io_strategy = f"Streamed ({strategy})"
            total_rows = rows_read if (strategy in ("full", "eager") or eof_hit) else None

        except NotImplementedError:
            # Non-streamable format (e.g. a JSON document parsed whole)
            # Let read_full() loads everything. A streaming parser (e.g.
            # ijson) is the only real fix for one too big for RAM.
            data = self.read_full(io_config)
            total_rows = len(data)
            ram_limit_hit = False
            eof_hit = True
            io_strategy = f"Monolithic ({strategy})"

        metadata = self.get_metadata()
        # Tables/dicts expose columns; line-based records (text) don't.
        columns = list(data[0].keys()) if data and isinstance(data[0], dict) else []
        metadata.update({
            "io_strategy": io_strategy,
            "file_eof_hit": eof_hit,
            "preview_limit": peek_limit if strategy == "peek" else None,
            "total_rows": total_rows,
            "total_lines_previewed": len(data),
            "columns": columns,
            "ram_limit_hit": ram_limit_hit,
            "preview_ram_bytes": ram_bytes,
        })
        return data, metadata

    def read_text_chunk(self, max_chars: int = 5000) -> str:
        """
        Universally safe raw byte-to-text inspection for the UI.
        """
        if not self.path.exists():
            return "File not found."
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                return f.read(max_chars)
        except Exception as e:
            return f"Error reading raw file: {str(e)}"
