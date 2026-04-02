"""
Data access policy and memory-safe file reading.
"""

from abc import ABC, abstractmethod
import csv
import json
from pathlib import Path
from typing import Any, Iterator

ACCEPTED_BINARY_FILES = {"bam", "vcf", "hdf5", "h5"}
ACCEPTED_IMAGE_FILES = {"png", "jpg", "jpeg", "svg"}
ACCEPTED_TABLE_FILES = {"csv", "tsv", "parquet", "jsonl", "json"}
ACCEPTED_TEXT_FILES = {
    "fasta", "faa", "fa", "fastq", "fq",
    "pdb", "aln", "txt", "log", "md", "info", "nfo", "raw",
}


class DataReader(ABC):
    """
    Abstract foundation. Every Reader implements three concepts.
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
            }

        return {
            "file_size_bytes": self.path.stat().st_size,
            "extension": self.path.suffix.lower().lstrip("."),
            "exists": True,
        }

    @abstractmethod
    def get_metadata(self) -> dict:
        """
        Deep metadata. Forces subclasses to peek inside the file if necessary
        (e.g., counting rows for a CSV, or getting dimensions for an Image).
        """
        pass

    @abstractmethod
    def stream(self) -> Iterator[Any]:
        """Yields small, memory-safe chunks (lines, dicts, byte-chinks)"""
        pass

    @abstractmethod
    def read_full(self) -> Any:
        """Loads the entire file into memory (or raises MemoryError if too big)"""
        pass

    @abstractmethod
    def preview(self, limit: int) -> tuple[Any, dict]:
        """
        Gets the first `limit` items (lines, dicts, bytes) and metadata about them.
        Subclasses decide if this uses stream() or read_full()
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


class TableReader(DataReader):
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

    def stream(self, config: dict | None = None) -> Iterator[str | dict]:
        """
        Memory-safe generator for large tabular files
        """
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

    def read_full(self, config: dict | None = None) -> list[str | dict]:
        """
        Loads the entire file into memory (or raises MemoryError if too big)
        """
        try:
            return list(self.stream(config))
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

    def preview(
        self,
        limit: int = 100,
        config: dict | None = None,
    ) -> tuple[list[str | dict], dict]:
        """
        Smart preview with graceful monolithic fallback
        Returns: tuple[preview_data, preview_metadata]
        """
        data = []
        hit_eof = True
        total_rows = None

        try:
            for i, record in enumerate(self.stream(config)):
                if i >= limit:
                    hit_eof = False
                    break
                data.append(record)
            strategy = "Streamed preview"
        except NotImplementedError:
            # fallback to monolithic load
            # TODO: Use ijson to stream monolithic JSON arrays to protect RAM
            all_records = self.read_full(config)
            total_rows = len(all_records)
            data = all_records[:limit]
            hit_eof = total_rows <= limit
            strategy = "Full-load preview"

        metadata = self.get_metadata()

        # CSV/TSV can be headerless, so only assume columns if we have a dict
        columns = []
        if data and isinstance(data[0], dict):
            columns = list(data[0].keys())

        metadata.update(
            {
                "strategy_used": strategy,
                "is_truncated": not hit_eof,
                "preview_limit": limit,
                "total_rows": total_rows,  # Will be None if streamed, which is correct!
                "columns": columns,
            }
        )

        return data, metadata


class ImageReader(DataReader):
    """
    Loads image bytes
    (PNG, JPEG, SVG)
    """

    def get_metadata(self) -> dict:
        meta = self.get_base_metadata()
        ext = meta.get("extension")
        meta["is_vector"] = ext == "svg"
        meta["can_stream"] = False
        return meta

    def stream(self, config: dict | None = None) -> Iterator[Any]:
        """Images are monolithic blocks. Streaming is not supported."""
        raise NotImplementedError("Streaming not supported for monolithic image files.")

    def read_full(self, config: dict | None = None) -> bytes | str:
        """Loads the entire image into memory."""
        ext = self.path.suffix.lower().lstrip(".")
        if ext == "svg":
            return self.path.read_text(encoding="utf-8", errors="replace")
        return self.path.read_bytes()

    def preview(self, limit: int = 0, config: dict | None = None) -> tuple[bytes | str, dict]:
        """
        No image previews! Here, we just enforce a size limit and return full
        """
        meta = self.get_metadata()

        # 10MB safety limit for inline web previews
        MAX_IMAGE_SIZE = 10 * 1024 * 1024

        if meta["file_size_bytes"] > MAX_IMAGE_SIZE:
            raise MemoryError(
                f"Image file ({meta['file_size_bytes'] / 1024 / 1024:.1f} MB) "
                "is too large for an inline preview."
            )

        data = self.read_full(config)

        meta.update(
            {
                "strategy_used": "read_full",
                "is_truncated": False,
            }
        )

        return data, meta


class TextReader(DataReader):
    """
    Handles massive text files (e.g. FASTQ) with chunking/streaming
    (FASTA, FASTQ, PDB, ALN)
    """

    def get_metadata(self) -> dict:
        """Deep metadata. Line counting is skipped as it requires a full scan."""
        meta = self.get_base_metadata()
        meta["can_stream"] = True
        return meta

    def stream(self, config: dict | None = None) -> Iterator[str]:
        """Memory-safe generator yielding one text line at a time."""
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line

    def read_full(self, config: dict | None = None) -> str:
        """Loads the entire file into memory as a single string."""
        return self.path.read_text(encoding="utf-8", errors="replace")

    def preview(self, limit: int = 100, config: dict | None = None) -> tuple[list[str], dict]:
        """
        Smart preview: pulls exactly `limit` lines from the stream.
        """
        lines = []
        hit_eof = True

        for i, line in enumerate(self.stream(config)):
            if i >= limit:
                hit_eof = False
                break
            lines.append(line)

        metadata = self.get_metadata()
        metadata.update(
            {
                "strategy_used": "streamed lines",
                "is_truncated": not hit_eof,
                "preview_limit": limit,
                "total_lines_previewed": len(lines),
            }
        )

        return lines, metadata


class BinaryBioReader(DataReader):
    """
    Handles binary bioinformatics files with chunking/streaming
    (BAM, VCF, HDF5)
    """

    ...


def get_reader_for(path: Path | str) -> DataReader:
    """
    Factory function to select the appropriate Reader based on file extension or MIME type.
    """
    if isinstance(path, str):
        path = Path(path)

    ext = path.suffix.lower().lstrip(".")

    if ext in ACCEPTED_TABLE_FILES:
        return TableReader(path)
    elif ext in ACCEPTED_IMAGE_FILES:
        return ImageReader(path)
    elif ext in ACCEPTED_TEXT_FILES:
        return TextReader(path)
    elif ext in ACCEPTED_BINARY_FILES:
        # return BinaryBioReader(path)
        raise NotImplementedError("Binary bioinformatics files are not yet supported")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
