"""
Rudimentary SVG generation utilities.
"""
from dataclasses import dataclass, field
from enum import Enum


class SvgUnits(Enum):
    """Units for SVG dimensions."""
    PIXELS = "px"
    PERCENT = "%"


@dataclass
class SvgStyle:
    """Styling options for SVG elements."""
    fill: str = "none"
    stroke: str = "none"
    stroke_width: float = 1.0
    stroke_color: str = "black"
    opacity: float = 1.0
    font_size: int = 12
    font_family: str = "sans-serif"
    text_anchor: str = "start"  # Options: start, middle, end


@dataclass
class SvgElement:
    """
    Base class for all SVG elements. Handles common logic of ID, Classes, Data
    """
    classes: list[str] = field(default_factory=list)
    data: dict[str, str | int | float] = field(default_factory=dict)
    style: SvgStyle = field(default_factory=SvgStyle)

    def _common_attrs(self) -> str:
        """Compiles common attributes (class, data-*, style) into a string"""
        parts = []

        # 1. CSS classes
        if self.classes:
            parts.append(f'class="{" ".join(self.classes)}"')

        # 2. Data attributes
        for k, v in self.data.items():
            clean_key = k.replace("_", "-")
            parts.append(f'data-{clean_key}="{v}"')

        # 3. Style attribute
        s = self.style
        if s.fill != "none":
            parts.append(f'fill="{s.fill}"')
        if s.stroke != "none":
            parts.append(f'stroke="{s.stroke}"')
        if s.stroke != "none":
            parts.append(f'stroke-width="{s.stroke_width}"')
        if s.opacity != 1.0:
            parts.append(f'opacity="{s.opacity}"')

        return " ".join(parts)

    def render(self) -> str:
        """Overridden by subclasses to return the SVG string for this element."""
        raise NotImplementedError("Subclasses must implement render()")


@dataclass
class SvgRect(SvgElement):
    """Simple rectangle element."""
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0
    rx: float = 0  # Corner radius

    def render(self) -> str:
        return (f'<rect x="{self.x:.2f}" y="{self.y:.2f}" '
                f'width="{self.width:.2f}" height="{self.height:.2f}" '
                f'rx="{self.rx}" {self._common_attrs()} />')


@dataclass
class SvgPolygon(SvgElement):
    """Polygon element defined by a list of (x, y) points"""
    points: list[tuple[float, float]] = field(default_factory=list)

    def render(self) -> str:
        points_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in self.points)
        return f'<polygon points="{points_str}" {self._common_attrs()} />'


@dataclass
class SvgLine(SvgElement):
    """Line element defined by start (x1, y1) and end (x2, y2) coordinates."""
    x1: float = 0
    y1: float = 0
    x2: float = 0
    y2: float = 0

    def render(self) -> str:
        return (f'<line x1="{self.x1:.2f}" y1="{self.y1:.2f}" '
                f'x2="{self.x2:.2f}" y2="{self.y2:.2f}" '
                f'{self._common_attrs()} />')


@dataclass
class SvgText(SvgElement):
    """
    Text element positioned at (x, y) with content in `text`. Style attributes 
    like font are handled by SvgStyle.
    """
    x: float = 0
    y: float = 0
    text: str = ""

    def render(self) -> str:
        s = self.style
        # Inject text-specific styles (font-family, font-size, text-anchor)
        style_attrs = (f'font-family="{s.font_family}" '
                       f'font-size="{s.font_size}" '
                       f'text-anchor="{s.text_anchor}"')

        return (f'<text x="{self.x:.2f}" y="{self.y:.2f}" '
                f'{style_attrs} {self._common_attrs()}>{self.text}</text>')


@dataclass
class SvgGroup(SvgElement):
    """
    <g> tag
    Useful e.g. for 'Tracks'. Allows quick handling of multiple SvgElements that 
    are intrinsically linked (e.g. a gene and its label, or an entire track)
    """
    elements: list[SvgElement] = field(default_factory=list)
    translate_x: float = 0
    translate_y: float = 0
    rotate: float = 0  # Rotation in degrees

    def add(self, element: SvgElement):
        """Add an SvgElement to this group"""
        self.elements.append(element)

    def render(self) -> str:
        transform = ""
        if self.translate_x != 0 or self.translate_y != 0:
            transform = f'transform="translate({self.translate_x:.2f}, {self.translate_y:.2f})"'
        if self.rotate != 0:
            transform += f' rotate({self.rotate})'

        inner_content = "\n  ".join(e.render() for e in self.elements)
        return f'<g {transform} {self._common_attrs()}>\n  {inner_content}\n</g>'


@dataclass
class SvgCanvas:
    """Container for SVG elements and metadata."""
    width: float
    height: float
    elements: list[SvgElement] = field(default_factory=list)

    def add(self, element: SvgElement):
        """Add an SvgElement to the canvas."""
        self.elements.append(element)

    def render(self) -> str:
        header = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                  f'width="{self.width:.2f}" height="{self.height:.2f}"> '
                  f'viewBox="0 0 {self.width:.2f} {self.height:.2f}">')

        # White background for now
        bg = '<rect width="100%" height="100%" fill="white" />'

        body = "\n".join(e.render() for e in self.elements)
        return f"{header}\n{bg}\n{body}\n</svg>"
