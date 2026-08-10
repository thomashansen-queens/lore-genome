"""
Reusable composite SVG glyphs shared across the viz library.
"""
from . import svg as v


def axis_break(
    x: float,
    y_center: float,
    half_height: float,
    color: str,
    stroke_width: float = 1.5,
) -> v.SvgGroup:
    """
    The '//' broken-axis indicator. Centered on (x, y_center).
    """
    top = y_center - half_height
    bot = y_center + half_height
    group = v.SvgGroup(classes=["break-mark"])

    # White wedge to knock out whatever is underneath (Painter's algorithm)
    group.add(v.SvgPolygon(
        points=[(x - 5, top), (x + 1, bot), (x + 5, bot), (x - 1, top)],
        fill="#FFFFFF",
    ))
    # Diagonal lines
    if stroke_width > 0:
        group.add(v.SvgLine(x1=x - 5, y1=top, x2=x + 1, y2=bot, stroke=color, stroke_width=stroke_width))
        group.add(v.SvgLine(x1=x - 1, y1=top, x2=x + 5, y2=bot, stroke=color, stroke_width=stroke_width))

    return group
