"""
Raw file reader for when no other reader can be found. Sniffs for binary
content and returns a hex dump or text preview accordingly.
"""
from .base import BaseReader
import binascii
from typing import Iterator


class RawReader(BaseReader):
    """
    A reader that yields either text or hex dump lines.
    """
    def get_metadata(self) -> dict:
        base_meta = super().get_base_metadata()
        base_meta["can_stream"] = True
        return base_meta

    def stream(self, config: dict | None = None, **kwargs) -> Iterator[str]:
        # sniff the first 1024 bytes to check for null bytes (heuristic for binary content)
        is_binary = False
        with open(self.path, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                is_binary = True

        if is_binary:
            with open(self.path, "rb") as f:
                while chunk := f.read(32):  # 32 byte chunks
                    yield binascii.hexlify(chunk, sep=" ", bytes_per_sep=4).decode("ascii")
        else:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    yield line
