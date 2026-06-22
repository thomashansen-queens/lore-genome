"""
Core abstract Reader class. Turns 1s and 0s into useful data structures.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator


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

    @abstractmethod
    def stream(self, config: dict | None = None, **kwargs) -> Iterator[Any]:
        """Yields small, memory-safe chunks (lines, dicts, byte-chinks)"""
        pass

    @abstractmethod
    def read_full(self, config: dict | None = None, **kwargs) -> Any:
        """Loads the entire file into memory (or raises MemoryError if too big)"""
        pass

    @abstractmethod
    def preview(self, peek_limit: int, config: dict | None = None, **kwargs) -> tuple[Any, dict]:
        """
        Gets the first `peek_limit` items (lines, dicts, bytes) and metadata about them.
        Returns: (previewed_data, io_metadata_dict)
        Subclasses decide if this uses stream() or read_full().
        """
        pass

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
