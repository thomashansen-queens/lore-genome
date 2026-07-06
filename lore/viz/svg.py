"""
Rudimentary SVG generation utilities.
"""
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class SvgUnits(Enum):
    """Units for SVG dimensions."""
    PIXELS = "px"
    PERCENT = "%"


class SvgElement(BaseModel):
    """
    Base class for all SVG elements. Handles common logic of ID, Classes, Data
    """
    model_config = ConfigDict(extra="allow")
    classes: list[str] = Field(default_factory=list)
    data: dict[str, str | int | float] = Field(default_factory=dict)

    # Explicit fields for IDE autocomplete
    fill: str | None = "none"
    stroke: str | None = "none"
    stroke_width: float | str | None = None
    opacity: float | str | None = None
    font_family: str | None = None
    font_size: int | str | None = None
    text_anchor: str | None = None
    cursor: str | None = None

    def _common_attrs(self, exclude: set[str]) -> str:
        """Compiles common attributes (class, data-*, style) into a string"""
        parts = []

        # 1. CSS classes
        if self.classes:
            parts.append(f"class='{' '.join(self.classes)}'")

        # 2. Data attributes
        for k, v in self.data.items():
            clean_key = k.replace("_", "-")
            parts.append(f"data-{clean_key}='{v}'")

        # 3. Style attribute
        # A. Exclude non-style attributes
        base_exclude = {"classes", "data"} | exclude

        style_attrs = self.model_dump(exclude=base_exclude, exclude_none=True)

        # B. Dump style attributes to SVG style string
        for k, v in style_attrs.items():
            svg_key = k.replace("_", "-")
            parts.append(f"{svg_key}='{v}'")

        return " ".join(parts)

    def render(self) -> str:
        """Overridden by subclasses to return the SVG string for this element."""
        raise NotImplementedError("Subclasses must implement render()")


class SvgRect(SvgElement):
    """Simple rectangle element."""
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0
    rx: float = 0  # Corner radius

    def render(self) -> str:
        attrs = self._common_attrs(exclude={"x", "y", "width", "height", "rx"})
        return (
            f"<rect x='{self.x:.2f}' y='{self.y:.2f}' "
            f"width='{self.width:.2f}' height='{self.height:.2f}' "
            f"rx='{self.rx:.2f}' {attrs} />"
        )


class SvgCircle(SvgElement):
    """Circle element defined by center (cx, cy) and radius r."""
    cx: float = 0
    cy: float = 0
    r: float = 0

    def render(self) -> str:
        attrs = self._common_attrs(exclude={"cx", "cy", "r"})
        return (
            f"<circle cx='{self.cx:.2f}' cy='{self.cy:.2f}' r='{self.r:.2f}' {attrs} />"
        )


class SvgPolygon(SvgElement):
    """Polygon element defined by a list of (x, y) points"""
    points: list[tuple[float, float]] = Field(default_factory=list)

    def render(self) -> str:
        points_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in self.points)
        attrs = self._common_attrs(exclude={"points"})
        return f"<polygon points='{points_str}' {attrs} />"


class SvgLine(SvgElement):
    """Line element defined by start (x1, y1) and end (x2, y2) coordinates."""
    x1: float = 0
    y1: float = 0
    x2: float = 0
    y2: float = 0

    def render(self) -> str:
        attrs = self._common_attrs(exclude={"x1", "y1", "x2", "y2"})
        return (
            f"<line x1='{self.x1:.2f}' y1='{self.y1:.2f}' "
            f"x2='{self.x2:.2f}' y2='{self.y2:.2f}' {attrs} />"
        )


class SvgText(SvgElement):
    """
    Text element positioned at (x, y) with content in `text`. Style attributes 
    like font are handled by SvgStyle.
    """
    x: float = 0
    y: float = 0
    text: str = ""

    fill: str | None = "#111111"
    font_family: str = "sans-serif"
    font_size: int = 12
    text_anchor: str = "start"

    def render(self) -> str:
        attrs = self._common_attrs(exclude={"x", "y", "text"})
        return (
            f"<text x='{self.x:.2f}' y='{self.y:.2f}' "
            f"{attrs}>{self.text}</text>"
        )


class SvgTitle(SvgElement):
    """Title element for SVG, typically used for tooltips."""
    text: str = ""

    def render(self) -> str:
        attrs = self._common_attrs(exclude={"text"})
        return f"<title {attrs}>{self.text}</title>"


class SvgArrow(SvgElement):
    """
    A directional arrow (polygon) optimized for genomic features.
    Draws a 5-point arrow with rectangular body or 7-point if head_thickness is set.
    """
    x_start: float
    x_end: float
    y_center: float
    thickness: float
    head_thickness: float | None = None
    head_width: float = 10.0
    forward: bool = True  # right = forward

    def render(self) -> str:
        hw = min(self.head_width, abs(self.x_end - self.x_start))
        t = self.thickness / 2
        ht = self.head_thickness / 2 if self.head_thickness else None

        # 1. 5-point arrow
        if ht == t or ht is None:
            if self.forward:
                pts = [
                    (self.x_start, self.y_center - t),
                    (self.x_end - hw, self.y_center - t),
                    (self.x_end, self.y_center),
                    (self.x_end - hw, self.y_center + t),
                    (self.x_start, self.y_center + t),
                ]
            else:
                pts = [
                    (self.x_end, self.y_center - t),
                    (self.x_start + hw, self.y_center - t),
                    (self.x_start, self.y_center),
                    (self.x_start + hw, self.y_center + t),
                    (self.x_end, self.y_center + t),
                ]
        # 2. 7-point arrow
        else:
            if self.forward:
                pts = [
                    (self.x_start, self.y_center - t),
                    (self.x_end - self.head_width, self.y_center - t),
                    (self.x_end - self.head_width, self.y_center - ht),
                    (self.x_end, self.y_center),
                    (self.x_end - self.head_width, self.y_center + t),
                    (self.x_end - self.head_width, self.y_center + ht),
                    (self.x_start, self.y_center + ht),
                ]
            else:
                pts = [
                    (self.x_end, self.y_center - t),
                    (self.x_start + self.head_width, self.y_center - t),
                    (self.x_start + self.head_width, self.y_center - ht),
                    (self.x_start, self.y_center),
                    (self.x_start + self.head_width, self.y_center + t),
                    (self.x_start + self.head_width, self.y_center + ht),
                    (self.x_end, self.y_center + ht),
                ]

        pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        attrs = self._common_attrs(exclude={
            "x_start", "x_end", "y_center", "thickness", "head_thickness", "head_width", "forward"
        })
        return f"<polygon points='{pts_str}' {attrs} />"


class SvgGroup(SvgElement):
    """
    <g> tag
    Useful e.g. for 'Tracks'. Allows quick handling of multiple SvgElements that 
    are intrinsically linked (e.g. a gene and its label, or an entire track)
    """
    elements: list[SvgElement] = Field(default_factory=list)
    translate_x: float = 0
    translate_y: float = 0
    rotate: float = 0  # Rotation in degrees

    fill: str | None = None
    stroke: str | None = None

    def add(self, element: SvgElement):
        """Add an SvgElement to this group"""
        self.elements.append(element)

    def render(self) -> str:
        transform = ""
        if self.translate_x != 0 or self.translate_y != 0:
            transform = f"transform='translate({self.translate_x:.2f}, {self.translate_y:.2f})'"
        if self.rotate != 0:
            transform += f" rotate({self.rotate})"

        attrs = self._common_attrs(exclude={"elements", "translate_x", "translate_y", "rotate"})
        inner_content = "\n  ".join(e.render() for e in self.elements)
        tag_internals = f"{transform} {attrs}".strip()
        return f"<g {tag_internals}>\n  {inner_content}\n</g>"


class SvgCanvas(BaseModel):
    """Container for SVG elements and metadata."""
    width: float
    height: float
    elements: list[SvgElement] = Field(default_factory=list)

    def add(self, element: SvgElement):
        """Add an SvgElement to the canvas."""
        self.elements.append(element)

    def render(self) -> str:
        header = (
            f"<svg xmlns='http://www.w3.org/2000/svg' "
            f"width='{self.width:.2f}' height='{self.height:.2f}' "
            f"viewBox='0 0 {self.width:.2f} {self.height:.2f}'>"
        )

        # White background for now
        bg = "<rect width='100%' height='100%' fill='white' />"
        body = "\n".join(e.render() for e in self.elements)

        return f"{header}\n{bg}\n{body}\n</svg>"
