"""
The vertical container that stacks tracks and shares one horizontal axis.
"""
from pydantic import BaseModel
from .base import BaseTrack, LabelPosition
from ..scale import Break, TrackBounds
from .. import svg as v
from .. import text
from ..units import resolve_px


class TrackTheme(BaseModel):
    """Styling and layout parameters for track visualization."""
    font_family: str = "monospace"
    font_size: int = 12
    font_color: str = "#333333"

    track_height: float = 50.0
    track_spacing: float = 5.0

    label_width: float | str = "auto"
    label_margin: float = 10.0
    right_margin: float = 20.0

    # Default theme colors
    color_backbone: str = "#64748B"
    color_highlight_fill: str = "#318686"
    color_highlight_stroke: str = "#2A6B6B"
    color_highlight_text: str = "#FAFFFF"
    color_default_fill: str = "#ADD8E6"
    color_default_stroke: str = "#8BB4C2"
    color_default_text: str = "#333333"

    def resolve_label_width(self, tracks: list[BaseTrack], total_width: float) -> float:
        """
        Resolve `label_width` to pixels for the label gutter.

        - 'auto': widest left-aligned label, capped at 35% of the total width.
        - 'full': widest left-aligned label, uncapped.
        - any length (150, '150px', '30%'): override to be exactly this length.
        """
        if self.label_width in ("auto", "full"):
            is_monospace = "mono" in self.font_family.lower()
            widest = text.widest(
                (t.name for t in tracks if t.label_pos == LabelPosition.LEFT),
                self.font_size,
                is_monospace,
            )
            if not widest:
                return 0.0

            gutter = widest + self.label_margin * 2

            # Cap the gutter only if 'auto' is used
            if self.label_width == "auto":
                gutter = min(gutter, total_width * 0.5)  # cap at 35% width
            return gutter

        # Otherwise, resolve the label_width to pixels
        return resolve_px(self.label_width, total_width)


class TrackStack:
    """Master container that scales and stacks tracks vertically."""
    def __init__(
        self,
        width: float = 1200,
        theme: TrackTheme | dict | None = None,
        **theme_kwargs,
    ):
        self.width = width
        self.tracks: list[BaseTrack] = []
        if isinstance(theme, dict):
            self.theme = TrackTheme(**theme)
        else:
            self.theme = theme or TrackTheme()

        if theme_kwargs:
            self.theme = self.theme.model_copy(update=theme_kwargs)

    def add_track(self, track: BaseTrack):
        """Add a track to the stack."""
        self.tracks.append(track)

    def data_extents(self) -> tuple[float, float]:
        """Union of every track's (min, max) data footprint, sets default domain."""
        spans = [t.get_extents() for t in self.tracks]
        if not spans:
            return (0.0, 1.0)
        lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
        if lo == hi:
            return (lo, hi + 1.0)  # guard zero-width span
        return (lo, hi)

    def render(
        self,
        domain: tuple[float, float] | dict | TrackBounds | None = None,
        breaks: list[Break] | tuple[Break, ...] = (),
    ) -> str:
        """
        Render all tracks against a shared horizontal `domain` (data-space window).

        Pass `domain=None` (the default) to let the stack infer the window from
        its tracks' `get_extents()`. An explicit `domain` is clipped to that data
        footprint via `TrackBounds.resolve`, so the view never runs off the data.

        `breaks` are discontinuities shared by every track (e.g. a locus-wide
        hidden region); per-track breaks are merged in by each track's own scale.
        """
        data_min, data_max = self.data_extents()
        if domain is None:
            domain = TrackBounds(start=data_min, end=data_max)
        else:
            domain = TrackBounds.model_validate(domain).resolve(data_min, data_max)

        # Resolve all labels to set stack gutter width
        label_px = self.theme.resolve_label_width(self.tracks, self.width)

        elements: list[v.SvgElement] = []
        y_cursor = self.theme.track_spacing

        for track in self.tracks:
            # 1. Active theme (CSS-style: per-track kwargs override the stack).
            active = self.theme.model_copy(update=track.theme_kwargs)
            track_group = v.SvgGroup(translate_y=y_cursor)

            # 2. Reserve label gutter and compute the plotting area.
            has_label = track.label_pos == LabelPosition.LEFT
            plot_x = label_px if has_label else 0.0
            plot_w = max(1.0, self.width - plot_x - active.right_margin)

            # TODO: For now, track-level metadata lives on the label
            if has_label:
                name_fit = text.truncate_to_fit(
                    track.name,
                    max_width=label_px - active.label_margin,
                    font_size=active.font_size,
                )
                label_group = v.SvgGroup(classes=["track-label"])

                if track.metadata:
                    hover_lines = [f"{k}: {v}" for k, v in track.metadata.items()]
                    label_group.add(v.SvgTitle(text="\n".join(hover_lines)))

                label_group.add(v.SvgText(
                    x=label_px - active.label_margin,
                    y=active.track_height / 2 + active.font_size * 0.35,
                    text=name_fit,
                    text_anchor="end",
                    font_family=active.font_family,
                    font_size=active.font_size,
                    fill=active.font_color,
                ))
                track_group.add(label_group)

            # 3. Build the scale and draw the payload into the plotting area.
            scale = track.build_scale(domain, plot_w, shared_breaks=breaks)
            payload = track.render_payload(scale, active)
            payload.translate_x = plot_x
            track_group.add(payload)

            # 4. Advance the vertical cursor.
            elements.append(track_group)
            y_cursor += active.track_height + active.track_spacing

        canvas = v.SvgCanvas(width=self.width, height=y_cursor)
        canvas.elements = elements
        return canvas.render()
