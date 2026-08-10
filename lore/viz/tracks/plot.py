"""
A simple X/Y plot track for a single series of points. Can be used for scatter,
line, or bar plots.
"""
from enum import StrEnum
import math
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING, Literal, Sequence
from .base import BaseTrack
from ..scale import Scale
from .. import svg as v

if TYPE_CHECKING:
    from .stack import TrackTheme


class PlotKind(StrEnum):
    """Kind of plot to render."""
    LINE = "line"
    AREA = "area"
    SCATTER = "scatter"
    BAR = "bar"
    RIBBON = "ribbon"

PlotKindLiteral = Literal["line", "area", "scatter", "bar", "ribbon"]


class AxisConfig(BaseModel):
    """Configuration for an axis in a plot track."""
    show_y_spine: bool = True
    y_ticks: int | list[float] = 3
    y_tick_format: str = ".0f"
    show_y_gridlines: bool = True
    y_min: float | None = None
    y_max: float | None = None


class PlotTrack(BaseTrack):
    """
    A simple X/Y plot track for a single series of points.
    Accepts data as either (x,y) or (x_start,x_end,y) for variable-width bins.
    """
    type: PlotKind | PlotKindLiteral = PlotKind.SCATTER
    data: Sequence[
        tuple[float, float] |
        tuple[float, float, float] |
        tuple[float, float, float, float]
    ]

    axis: AxisConfig = Field(default_factory=AxisConfig)

    def _parse_geom(self, d: tuple) -> tuple[float, float, float, float]:
        """
        Parse a data point tuple into a standard (x_start, x_end, y_min, y_max) format.
        """
        if len(d) == 2:
            return (d[0], d[0], d[1], d[1])  # x, y
        if len(d) == 3 and self.type == PlotKind.RIBBON:
            return (d[0], d[0], d[1], d[2])  # x, y_min, y_max
        if len(d) == 3 and self.type != PlotKind.RIBBON:
            return (d[0], d[1], d[2], d[2])  # x_start, x_end, y
        if len(d) == 4:
            return (d[0], d[1], d[2], d[3])  # x1, x2, y_min, y_max

        raise ValueError(f"Invalid data point format for plot kind '{self.type}': {d}")

    def get_extents(self) -> tuple[float, float]:
        if not self.data:
            return (0.0, 0.0)
        parsed = [self._parse_geom(d) for d in self.data]
        return (min([d[0] for d in parsed]), max([d[1] for d in parsed]))

    def render_payload(self, scale: Scale, theme: "TrackTheme") -> v.SvgGroup:
        group = v.SvgGroup()
        if not self.data:
            return group

        # 1. Resolve Y-axis domain
        parsed = [self._parse_geom(d) for d in self.data]
        y_vals = [y for p in parsed for y in (p[2], p[3])]
        self._active_y_min = self.axis.y_min if self.axis.y_min is not None else min(y_vals)

        if self.axis.y_max is not None:
            self._active_y_max = self.axis.y_max
        else:
            raw_max = max(y_vals)
            self._active_y_max = _nice_number(max(y_vals)) if raw_max > self._active_y_min else raw_max
            if self._active_y_max == self._active_y_min:
                self._active_y_max = self._active_y_min + 1.0

        y_span = self._active_y_max - self._active_y_min
        if y_span == 0:
            y_span = 1.0  # Avoid divide-by-zero

        def project_y(val: float) -> float:
            """Normalize a Y value to the track height (0 is top, 1 is bottom)"""
            clamped_val = max(self._active_y_min, min(val, self._active_y_max))
            norm = (clamped_val - self._active_y_min) / y_span
            return theme.track_height * (1.0 - norm)

        # Bind to self to helpers can access
        self._project_y = project_y

        # 2. Filter points
        visible_data = [
            p for p in parsed
            if p[1] >= scale.domain.start and p[0] <= scale.domain.end
        ]
        if not visible_data:
            return group

        # 3. Downsample
        resolution_limit = max(100, min(int(scale.px_width / 2.0), 1000))
        if len(visible_data) > resolution_limit:
            visible_data = self._downsample(visible_data, resolution_limit)

        # 4. Hand off to render helpers
        if self.type == PlotKind.LINE:
            self._render_line(group, visible_data, scale, theme, fill=False)
        elif self.type == PlotKind.AREA:
            self._render_line(group, visible_data, scale, theme, fill=True)
        elif self.type == PlotKind.SCATTER:
            self._render_scatter(group, visible_data, scale, theme)
        elif self.type == PlotKind.BAR:
            self._render_bar(group, visible_data, scale, theme)
        elif self.type == PlotKind.RIBBON:
            self._render_ribbon(group, visible_data, scale, theme)

        # 5. Render Y-axis overlay
        self._render_y_axis(group, theme, scale)

        return group

    # === Helper methods ===

    # TODO: Add option for smoothed downsampling, maybe make shaded line plot a first-class option?
    def _downsample(self, data: list, limit: int) -> list:
        """
        Downsample the data to a maximum number of points.
        Uses a simple uniform sampling approach.
        """
        chunk_size = max(1, len(data) // limit)
        downsampled = []

        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]

            if self.type == PlotKind.RIBBON:
                # Ribbon shows min/max of each chunk
                x_start = chunk[0][0]
                x_end = chunk[-1][1]
                y_min = min(d[2] for d in chunk)
                y_max = max(d[3] for d in chunk)
                downsampled.append((x_start, x_end, y_min, y_max))
            else:
                # Preserve spikes by keeping max
                best_point = max(chunk, key=lambda d: d[3])
                downsampled.append(best_point)
        return downsampled

    def _render_line(
        self,
        group: v.SvgGroup,
        data: list,
        scale: Scale,
        theme: "TrackTheme",
        fill: bool = False,
    ):
        pts = []
        for d in data:
            x_coord = (d[0] + d[1]) / 2.0
            pts.append((scale.px(x_coord), self._project_y(d[3])))

        if fill:
            first_x = (data[0][0] + data[0][1]) / 2.0
            last_x = (data[-1][0] + data[-1][1]) / 2.0
            y_zero = self._project_y(0.0)

            pts.insert(0, (scale.px(first_x), y_zero))
            pts.append((scale.px(last_x), y_zero))

            group.add(v.SvgPolygon(
                points=pts,
                stroke=theme.color_default_stroke,
                stroke_width=1.0,
                fill=theme.color_default_fill,
                fill_opacity=0.3,
            ))
        else:
            group.add(v.SvgPolyline(
                points=pts,
                stroke=theme.color_default_stroke,
                stroke_width=1.5,
                fill="none",
            ))

    def _render_scatter(
        self,
        group: v.SvgGroup,
        data: list,
        scale: Scale,
        theme: "TrackTheme",
    ):
        for d in data:
            x_coord = (d[0] + d[1]) / 2.0
            group.add(v.SvgCircle(
                cx=scale.px(x_coord),
                cy=self._project_y(d[3]),
                r=2.5,
                fill=theme.color_default_fill,
            ))

    def _render_bar(
        self,
        group: v.SvgGroup,
        data: list,
        scale: Scale,
        theme: "TrackTheme",
    ):
        y_zero = self._project_y(0.0) if self._active_y_min < 0 else self._project_y(self._active_y_min)

        for pt in data:
            px_start, px_end = scale.span(pt[0], pt[1])
            px_y = self._project_y(pt[3])

            bar_y = min(px_y, y_zero)
            bar_h = abs(px_y - y_zero)

            if bar_h > 0:
                group.add(v.SvgRect(
                    x=px_start,
                    y=bar_y,
                    width=max(1.0, px_end - px_start),
                    height=bar_h,
                    fill=theme.color_default_fill,
                ))

    def _render_ribbon(
        self,
        group: v.SvgGroup,
        data: list,
        scale: Scale,
        theme: "TrackTheme",
    ):
        pts_upper = []
        pts_lower = []

        for d in data:
            x_start, x_end, y_min, y_max = d
            px_start = scale.px(x_start)
            px_end = scale.px(x_end)
            py_max = self._project_y(y_max)
            py_min = self._project_y(y_min)

            if px_start != px_end:
                pts_upper.extend([(px_start, py_max), (px_end, py_max)])
                pts_lower.extend([(px_start, py_min), (px_end, py_min)])
            else:
                pts_upper.append((px_start, py_max))
                pts_lower.append((px_start, py_min))

        # Render top left -> top right -> bottom right -> bottom left -> close
        pts_lower.reverse()
        full_pts = pts_upper + pts_lower
        group.add(v.SvgPolygon(
            points=full_pts,
            fill=theme.color_default_fill,
            fill_opacity=0.3,
            stroke=theme.color_default_stroke,
            stroke_width=1.0,
        ))

    # === Y-axis ===

    def _render_y_axis(self, group: v.SvgGroup, theme: "TrackTheme", scale: Scale):
        if not self.axis.show_y_spine and not self.axis.y_ticks:
            return

        # Draw spine
        if self.axis.show_y_spine:
            group.add(v.SvgLine(
                x1=0.0, y1=0.0, x2=0.0, y2=theme.track_height,
                stroke=theme.color_default_stroke, stroke_width=1.0,
            ))

        # Resolve ticks
        if isinstance(self.axis.y_ticks, int):
            step = (self._active_y_max - self._active_y_min) / max(1, self.axis.y_ticks - 1)
            tick_vals = [self._active_y_min + (i * step) for i in range(self.axis.y_ticks)]
        else:
            tick_vals = self.axis.y_ticks

        # Draw ticks and gridlines
        if self.axis.y_ticks:
            if isinstance(self.axis.y_ticks, int):
                step = (self._active_y_max - self._active_y_min) / max(1, self.axis.y_ticks - 1)
                tick_vals = [self._active_y_min + (i * step) for i in range(self.axis.y_ticks)]
            else:
                tick_vals = self.axis.y_ticks

            for val in tick_vals:
                py = self._project_y(val)

                # Gridline
                if self.axis.show_y_gridlines:
                    group.add(v.SvgLine(
                        x1=0.0, y1=py, x2=scale.px_width, y2=py,
                        stroke=theme.color_default_stroke, stroke_width=0.5,
                        stroke_dasharray="4 4",
                    ))

                # Tick text
                group.add(v.SvgText(
                    x=4.0, y=py-4.0, 
                    text=f"{val:{self.axis.y_tick_format}}",
                    font_size=theme.font_size * 0.8,
                    fill=theme.font_color,
                    text_anchor="start"
                ))


def _nice_number(n: int | float, padding: float = 0.05) -> int | float:
    """Return a nice maximum value for plotting slightly above the given number."""
    padded = n * (1 + padding)
    magnitude = 10 ** math.floor(math.log10(padded))
    normalized = padded / magnitude

    for multiplier in (1, 1.5, 2, 2.5, 5, 10):
        if normalized <= multiplier:
            result = multiplier * magnitude
            return int(result) if isinstance(n, int) else result
    raise RuntimeError("Failed to compute a nice number for plotting.")
