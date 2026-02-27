"""
Adapter for SVG data
"""
from typing import Any, ClassVar

from lore.core.adapters import adapter_registry, ImageAdapter


class SvgAdapter(ImageAdapter):
    """
    Pass-through adapter for SVG files. 
    Tells the UI to render this directly as an image.
    """
    accepted_formats: ClassVar[set[str]] = {"svg"}
    accepted_types: ClassVar[set[str]] = {"*"}
    view_mode: ClassVar[str] = "image"

    def adapt(self, raw_data: Any) -> str:
        """For SVGs, adapting just means ensuring it's a string."""
        if isinstance(raw_data, bytes):
            return raw_data.decode("utf-8")
        return str(raw_data)

    def to_png(self, path_or_data: Any, scale: float = 2.0):
        """
        FUTURE: helper using something like cairosvg to rasterize
        vector graphics for tasks that require standard images.
        """
        pass


adapter_registry.register(SvgAdapter())
