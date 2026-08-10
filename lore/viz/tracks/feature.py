"""
Track for rendering 1D features as SVG: genes, ORFs, domains, motifs, etc.

The track never computes virtual coordinates or clamps anything itself. Anything
that needs to be visually compressed (an overly long intron, an intergenic gap,
a circular origin) is expressed as a `Break` on the scale; a feature that crosses
one simply renders shorter via `scale.span()`, and the corresponding '//' glyph
is drawn from `scale.marks`.
"""
from enum import StrEnum
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING

from .base import BaseTrack
from ..glyphs import axis_break
from ..scale import BreakKind, Scale
from .. import svg as v
from .. import text

if TYPE_CHECKING:
    from .stack import TrackTheme


class FeatureShape(StrEnum):
    BOX = "box"
    ARROW_RIGHT = "arrow_right"
    ARROW_LEFT = "arrow_left"


class Feature(BaseModel):
    """
    Data contract for a standard 1D feature, in data coordinates.
    thickness_ratio: amount of the vertical track space to occupy (0.0-1.0)
    head_thickness_ratio: amount of the vertical track space to occupy for the arrowhead (0.0-1.0)
    """
    # Layout and styling
    start: float
    end: float
    shape: FeatureShape = FeatureShape.BOX
    label: str = ""
    fill: str | None = None
    stroke: str | None = None
    thickness_ratio: float = 0.65
    head_thickness_ratio: float | None = None

    # Data for presentation layer
    highlight: bool = False  # i.e. should this feature be 'highlighted'?
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class Backbone(BaseModel):
    """The central axis line beneath the features."""
    show: bool = True
    stroke: str | None = None
    stroke_width: float = 2.0

    def render(
        self,
        group: v.SvgGroup,
        scale: Scale,
        y_center: float,
        theme: "TrackTheme",
        extent: tuple[float, float] | None = None,
    ) -> None:
        if not self.show:
            return

        color = self.stroke or theme.color_backbone

        # A. Draw line. Defaults to plot width; 'extent' overrides for trackwise clamping
        x1, x2 = (0.0, scale.px_width) if extent is None else scale.span(*extent)
        group.add(v.SvgLine(
            x1=x1, y1=y_center, x2=x2, y2=y_center,
            stroke=color, stroke_width=self.stroke_width,
        ))

        # B. Erase the backbone across circular-origin wraps.
        for mark in scale.marks_of(BreakKind.GAP):
            group.add(v.SvgLine(
                x1=mark.x_start, y1=y_center, x2=mark.x_end, y2=y_center,
                stroke="#FFFFFF", stroke_width=self.stroke_width + 1.0,
            ))

            # C. Terminus dots (contig ends)
            group.add(v.SvgCircle(cx=mark.x_start, cy=y_center, r=4, fill=color))
            group.add(v.SvgCircle(cx=mark.x_end, cy=y_center, r=4, fill=color))


class FeatureTrack(BaseTrack):
    """A track of 1D features drawn as boxes or directional arrows"""
    backbone: Backbone = Field(default_factory=Backbone)
    features: list[Feature]
    extent: tuple[float, float] | None = None  # pin the axis span; None => derive from features

    def get_extents(self) -> tuple[float, float]:
        """The track's data footprint: the pinned `extent`, else the feature span."""
        if self.extent is not None:
            return self.extent
        if not self.features:
            return (0.0, 0.0)
        features_min = min(f.start for f in self.features)
        features_max = max(f.end for f in self.features)
        return features_min, features_max

    def render_payload(self, scale: Scale, theme: "TrackTheme") -> v.SvgGroup:
        group = v.SvgGroup()
        y_center = theme.track_height / 2

        # 1. Backbone + between-feature (CUT) break glyphs
        self.backbone.render(group, scale, y_center, theme, extent=self.get_extents())
        for mark in scale.marks_of(BreakKind.CUT):
            group.add(axis_break(mark.x, y_center, theme.track_height * 0.6, theme.color_backbone))

        # 2. Features.
        splices = scale.marks_of(BreakKind.SPLICE)
        for feat in self.features:
            if feat.end < scale.domain.start or feat.start > scale.domain.end:
                continue  # outside the visible window

            px_start, px_end = scale.span(feat.start, feat.end)
            fill = feat.fill or (theme.color_highlight_fill if feat.highlight else theme.color_default_fill)
            stroke = feat.stroke or (theme.color_highlight_stroke if feat.highlight else theme.color_default_stroke)

            feat_group = v.SvgGroup(
                classes=["has-tooltip"] if feat.metadata else [],
                data=feat.metadata,
            )
            if feat.metadata:
                hover_lines = [f"{k}: {v}" for k, v in feat.metadata.items()]
                feat_group.add(v.SvgTitle(text="\n".join(hover_lines)))

            thickness = theme.track_height * feat.thickness_ratio
            head_thickness = theme.track_height * feat.head_thickness_ratio if feat.head_thickness_ratio else None

            # 2a. Geometric body
            if feat.shape == FeatureShape.BOX:
                feat_group.add(v.SvgRect(
                    x=px_start, y=y_center - thickness / 2,
                    width=max(1.0, px_end - px_start), height=thickness,
                    fill=fill, stroke=stroke, stroke_width=1.0,
                ))
            else:
                feat_group.add(v.SvgArrow(
                    x_start=px_start, x_end=px_end, y_center=y_center,
                    thickness=thickness, head_thickness=head_thickness,
                    forward=(feat.shape == FeatureShape.ARROW_RIGHT),
                    fill=fill, stroke=stroke, stroke_width=1.0,
                ))

            # 2b. Within-feature (SPLICE) break glyphs, colored to the feature
            for mark in splices:
                if px_start <= mark.x <= px_end:
                    feat_group.add(axis_break(mark.x, y_center, thickness * 0.6, stroke))

            # 2c. Feature label, if it fits
            if feat.label:
                avail = abs(px_end - px_start)
                label_txt = text.truncate_to_fit(feat.label, avail, theme.font_size)
                if label_txt:
                    text_color = theme.color_highlight_text if feat.highlight else theme.color_default_text
                    feat_group.add(v.SvgText(
                        x=(px_start + px_end) / 2,
                        y=y_center + theme.font_size * 0.35,
                        text=label_txt,
                        text_anchor="middle",
                        fill=text_color,
                        font_family=theme.font_family,
                        font_size=int(theme.font_size * 0.8),
                    ))

            group.add(feat_group)

        return group
