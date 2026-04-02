"""
Adapter for SVG data
"""
from pathlib import Path
import re
from typing import ClassVar

import lore.dsl as lore


@lore.adapter()
class SvgAdapter(lore.ImageAdapter):
    """
    Pass-through adapter for SVG files.
    Tells the UI to render embedded XML directly as an vector image.
    """
    accepted_formats: ClassVar[set[str]] = {"svg"}
    accepted_types: ClassVar[set[str]] = {"*"}
    view_mode: ClassVar[str] = "svg"
    version: ClassVar[str] = "1.0.0"

    def provides(self, requirement: str) -> bool:
        if super().provides(requirement):
            return True
        return requirement == "svg"

    def preview(
        self,
        raw_data: str,
        io_metadata: dict,
        config: dict | None = None,
    ) -> lore.AdapterPreview:
        """
        UI Contract: Packages the SVG string and extracts spatial metadata.
        """
        # 1. Extract basic metadata
        # Looks for <svg ... viewBox="0 0 100 100" ... >
        viewbox_match = re.search(r'viewBox="([^"]+)"', raw_data)
        viewbox = viewbox_match.group(1) if viewbox_match else "Unknown"

        final_metadata = {
            **io_metadata,
            "strategy_used": "embedded_xml",
            "viewbox": viewbox,
            "is_truncated": False, # SVGs can't be truncated or they break!
            "view_mode": self.view_mode,
        }

        # 3. Return the Standardized Payload
        return lore.AdapterPreview(
            data=raw_data,
            metadata=final_metadata
        )

    def to_png(self, path: Path, scale: float = 2.0):
        """
        FUTURE: helper using something like cairosvg to rasterize
        vector graphics for tasks that require standard images.
        """
        pass
