"""
Core Data Readers for LoRē.

This package defines the data ingestion layer of LoRē. The BaseReader is the
abstract class that all readers must inherit from. It enables memory-safe streaming,
metadata extraction, and full file reading. It also implements the 'preview' logic,
which allows for quick peeks of very large files.

Data-wise, one reader should be interchangeable with another! The plugin system
allows dependencies to be kept to a minimum and for efficiency to be optimized per-
format. E.g. a CSV reader using 'pandas' would be faster than one using the standard
'csv' lib, and using 'polars' could be faster still.

To register a new reader plugin, the `register` decorator is used.
"""
from pathlib import Path

from .base import BaseReader
from .registry import ReaderRegistry
from .image import ImageReader
from .table import TableReader
from .text import TextReader
from .raw import RawReader


from .image import IMAGE_EXTS
from .table import TABLE_EXTS
from .text import TEXT_EXTS

# Safety check: Avoid reader collisions
_all_extensions_ = (
    list(IMAGE_EXTS) +
    list(TABLE_EXTS) +
    list(TEXT_EXTS)
)
if len(_all_extensions_) != len(set(_all_extensions_)):
    raise ValueError("Collision detected among accepted Reader file types.")


reader_registry = ReaderRegistry()
reader_registry._register_core(TableReader, extensions=list(TABLE_EXTS))
reader_registry._register_core(TextReader, extensions=list(TEXT_EXTS))
reader_registry._register_core(ImageReader, extensions=list(IMAGE_EXTS))


def get_reader_for(path: Path | str) -> BaseReader:
    """
    Factory: resolve and instantiate the registered Reader for a file path.

    Looks the reader class up by file extension (including any registered via
    plugins), then instantiates it with the path. Raises ValueError if no Reader
    is registered for the extension.
    """
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    reader_cls = reader_registry.get_reader(ext)
    if reader_cls is None:
        raise ValueError(f"Unsupported file type: '{ext}' (no Reader registered)")
    return reader_cls(path)


__all__ = [
    "BaseReader",
    "TableReader",
    "TextReader",
    "ImageReader",
    "RawReader",
    "ReaderRegistry",
    "reader_registry",
    "get_reader_for",
]
