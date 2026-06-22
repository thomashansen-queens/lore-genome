"""
Core Reader for image formats (PNG, JPEG, SVG).
"""
from typing import Iterator

from .base import BaseReader

IMAGE_EXTS = {"png", "jpg", "jpeg", "svg"}


class ImageReader(BaseReader):
    """
    Loads image bytes (PNG, JPEG) or vector markup (SVG). Images are monolithic
    blocks — they cannot be streamed — so this reader enforces a size ceiling to
    keep inline web previews safe.
    """
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB safety limit for inline previews

    def get_metadata(self) -> dict:
        meta = self.get_base_metadata()
        meta["is_vector"] = meta.get("extension") == "svg"
        meta["can_stream"] = False
        return meta

    def stream(self, config: dict | None = None, **kwargs) -> Iterator:
        """Images are monolithic blocks; streaming is not supported."""
        raise NotImplementedError("Streaming not supported for monolithic image files.")

    def read_full(self, config: dict | None = None, **kwargs) -> bytes | str:
        """Loads the whole image: SVG as text, raster formats as bytes."""
        if self.path.suffix.lower().lstrip(".") == "svg":
            return self.path.read_text(encoding="utf-8", errors="replace")
        return self.path.read_bytes()

    def preview(
        self,
        peek_limit: int = 0,
        config: dict | None = None,
        **kwargs,
    ) -> tuple[bytes | str, dict]:
        """
        Images can't be partially previewed, so this returns the whole file
        (subject to the size ceiling). peek_limit is ignored.
        """
        meta = self.get_metadata()

        if meta["file_size_bytes"] > self.MAX_IMAGE_SIZE:
            raise MemoryError(
                f"Image file ({meta['file_size_bytes'] / 1024 / 1024:.1f} MB) "
                "is too large for an inline preview."
            )

        data = self.read_full(config)

        # The whole file is loaded, so this is the complete result (file_eof_hit).
        meta.update(
            {
                "io_strategy": "read_full",
                "file_eof_hit": True,
            }
        )

        return data, meta
