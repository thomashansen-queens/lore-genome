"""
Adapter for PNG image data.
"""
import base64
from pathlib import Path
from typing import ClassVar

from lore.core.adapters import adapter_registry, ImageAdapter


class PNGAdapter(ImageAdapter):
    accepted_formats: ClassVar[set[str]] = {"png"}
    accepted_types: ClassVar[set[str]] = {"*"}
    view_mode: ClassVar[str] = "image"

    def parse(self, source: Path) -> bytes:
        return source.read_bytes()

    def adapt(self, path: Path) -> str:
        """HTML requires base64-encoding for inline images."""
        raw_bytes = self.parse(path)
        encoded = base64.b64encode(raw_bytes).decode("utf-8")
        return f"data:image/png;base64,{encoded}"


adapter_registry.register(PNGAdapter())
