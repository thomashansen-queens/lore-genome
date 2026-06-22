"""
Core Reader for text-based formats.
"""
from .base import BaseReader
from typing import Iterator

TEXT_EXTS = {
    "fasta", "faa", "fa", "fna", "fastq", "fq",
    "pdb", "aln", "txt", "log", "md", "info", "nfo", "raw",
}


class TextReader(BaseReader):
    """
    Handles massive text files (e.g. FASTQ) with chunking/streaming
    (FASTA, FASTQ, PDB, ALN)
    """

    def get_metadata(self) -> dict:
        """Deep metadata. Line counting is skipped as it requires a full scan."""
        meta = self.get_base_metadata()
        meta["can_stream"] = True
        return meta

    def stream(self, config: dict | None = None, **kwargs) -> Iterator[str]:
        """Memory-safe generator yielding one text line at a time."""
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line

    def read_full(self, config: dict | None = None, **kwargs) -> str:
        """Loads the entire file into memory as a single string."""
        return self.path.read_text(encoding="utf-8", errors="replace")

