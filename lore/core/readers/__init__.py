"""
Core Data Readers for LoRē.

This package defines the data ingestion layer of LoRē. The BaseReader is the
abstract class that all readers must inherit from. It enables memory-safe streaming,
metadata extraction, and full file reading.

The registry allows for different strategies to be registered for different file
types, and is plugin-enabled allowing various file formats to be supported. To
register a new reader, the `register` decorator is used.
"""
from pathlib import Path

from .base import BaseReader
from .registry import ReaderRegistry
from .image import ImageReader
from .table import TableReader
from .text import TextReader


ACCEPTED_BINARY_FILES = {"bam", "vcf", "hdf5", "h5"}
ACCEPTED_IMAGE_FILES = {"png", "jpg", "jpeg", "svg"}
ACCEPTED_TABLE_FILES = {"csv", "tsv", "parquet", "jsonl", "json"}
ACCEPTED_TEXT_FILES = {
    "fasta", "faa", "fa", "fna", "fastq", "fq",
    "pdb", "aln", "txt", "log", "md", "info", "nfo", "raw",
}

# Safety check: Avoid reader collisions
_all_extensions_ = (
    list(ACCEPTED_BINARY_FILES) +
    list(ACCEPTED_IMAGE_FILES) +
    list(ACCEPTED_TABLE_FILES) +
    list(ACCEPTED_TEXT_FILES)
)
if len(_all_extensions_) != len(set(_all_extensions_)):
    raise ValueError("Collision detected among accepted Reader file types.")


reader_registry = ReaderRegistry()
reader_registry._register_core(TableReader, extensions=list(ACCEPTED_TABLE_FILES))
reader_registry._register_core(TextReader, extensions=list(ACCEPTED_TEXT_FILES))
reader_registry._register_core(ImageReader, extensions=list(ACCEPTED_IMAGE_FILES))


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
    "ReaderRegistry",
    "reader_registry",
    "get_reader_for",
]
