"""
FUTURE: This file is a placeholder for any PNG-specific adapter logic. Currently, PNGs are handled by the generic ImageAdapter, but if we want to add PNG-specific features (like metadata extraction, thumbnail generation, etc.) we can implement them here.
"""
import base64
from lore.core.adapters import ImageAdapter

class PNGAdapter(ImageAdapter):
    def adapt(self, raw_bytes: bytes) -> str:
        """HTML requires base64-encoding for inline images"""
        encoded = base64.b64encode(raw_bytes).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
