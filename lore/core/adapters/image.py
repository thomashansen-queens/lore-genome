"""
Placeholder for an ImageAdapter
"""
from typing import Any, ClassVar

from .base import BaseAdapter


class ImageAdapter(BaseAdapter):
    """
    The bridge between image data and a UI-viewable format
    """
    accepted_formats: ClassVar[set[str]] = {"png", "jpg", "jpeg"}  # e.g. {"svg", "png", "jpg", "jpeg", "gif"} 
    accepted_types: ClassVar[set[str]] = set()  # e.g. {"genome_map", "phylo_tree", "protein_structure"}
    view_mode: ClassVar[str] = "image"

    @property
    def provided_types(self) -> set[str]:
        return {"image"}

    def adapt(self, raw_data: Any, config: dict | None = None, **kwargs) -> Any:
        """
        Defaults to raw payload (e.g. SVG string or PNG bytes). Subclasses can 
        override to perform transformations or optimizations (e.g. resizing)
        """
        return raw_data
