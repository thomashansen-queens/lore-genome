"""
Core Reader for text-based formats.
"""
from .base import BaseReader
from typing import Iterator


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

    def preview(
        self,
        peek_limit: int = 100,
        config: dict | None = None,
        **kwargs,
    ) -> tuple[list[str], dict]:
        """
        Smart preview: pulls exactly `peek_limit` lines from the stream.
        """
        lines = []
        hit_eof = True

        for i, line in enumerate(self.stream(config)):
            if i >= peek_limit:
                hit_eof = False
                break
            lines.append(line)

        metadata = self.get_metadata()
        metadata.update(
            {
                "io_strategy": "streamed lines",
                "file_eof_hit": hit_eof,
                "preview_limit": peek_limit,
                "total_lines_previewed": len(lines),
            }
        )

        return lines, metadata
