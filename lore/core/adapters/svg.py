"""
Adapter for SVG data
"""
from pathlib import Path
import re
from typing import Any, ClassVar

from .base import AdapterPreview, BaseAdapter


class SvgAdapter(BaseAdapter):
    """
    Pass-through adapter for SVG files.
    Tells the UI to render embedded XML directly as an vector image.
    """
    accepted_formats: ClassVar[set[str]] = {"svg"}
    accepted_types: ClassVar[set[str]] = {"*"}
    view_mode: ClassVar[str] = "svg"
    version: ClassVar[str] = "1.0.0"

    def provides(self, requirement: str) -> bool:
        return requirement in ("image", "vector_graphic", "svg")

    def adapt(self, raw_data: Any, config: dict | None = None, **kwargs) -> str:
        """
        SVGs don't need Python adaptation; they just need to be valid strings.
        """
        if isinstance(raw_data, bytes):
            return raw_data.decode("utf-8")
        return str(raw_data)

    def adapt_stream(self, stream: Any, config: dict | None = None, **kwargs):
        raise NotImplementedError("SVG XML must be fully loaded to render; it cannot be streamed.")

    def preview(
        self,
        raw_data: str,
        io_metadata: dict,
        config: dict | None = None,
        **kwargs,
    ) -> AdapterPreview:
        """
        Override preview to inject metadata for UI rendering while preventing
        DOM truncation.
        """
        # 1. Adapt to guarantee a clean string
        svg_string = self.adapt(raw_data, config=config, **kwargs)

        # 2. Extract basic metadata
        # Looks for <svg ... viewBox="0 0 100 100" ... >
        viewbox_match = re.search(r'viewBox="([^"]+)"', raw_data)
        viewbox = viewbox_match.group(1) if viewbox_match else "Unknown"

        final_metadata = {
            **io_metadata,
            "io_strategy": "embedded_xml",
            "viewbox": viewbox,
            "is_truncated": False,  # SVGs can't be truncated or they break!
            "total_rows": 1,  # Not really rows, but tells the UI to treat this as a single item
            "view_mode": self.view_mode,
            "adapter_name": self.name,
        }

        # 3. Return the Standardized Payload
        return AdapterPreview(
            data=raw_data,
            view_mode="svg",
            adapter_name="SvgAdapter",
            metadata=final_metadata
        )

    def to_png(self, path: Path, scale: float = 2.0):
        """
        FUTURE: helper using something like cairosvg to rasterize
        vector graphics for tasks that require standard images.
        """
        pass
