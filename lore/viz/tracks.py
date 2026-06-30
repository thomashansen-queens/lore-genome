"""
Minilibrary for track-based visualizations
"""
import dataclasses
from dataclasses import dataclass
from enum import StrEnum
from . import svg as v


class LabelPosition(StrEnum):
    LEFT = "left"
    TOP = "top"
    NONE = "none"


@dataclass
class TrackTheme:
    """Styling and layout parameters for track visualization"""
    font_family: str = "monospace"
    font_size: int = 12
    font_color: str = "#333333"
    track_spacing: float = 5.0
    label_width: float = 150.0
    label_margin: float = 10.0
    right_margin: float = 20.0

    # Default theme colors
    color_backbone: str = "#64748B"
    color_primary_fill: str = "#318686"
    color_primary_stroke: str = "#2A6B6B"
    color_secondary_fill: str = "#ADD8E6"
    color_secondary_stroke: str = "#8BB4C2"
    color_primary_text: str = "#FAFFFF"
    color_secondary_text: str = "#333333"


@dataclass
class TrackBounds:
    """Generic 1D coordinate window (genomic bp, protein aa, etc.)"""
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


# TODO: Should this just be a module/utility functions?
class TextMetrics:
    """Utility class for text dimensions and truncation"""
    @staticmethod
    def _char_width(font_size: int, is_monospace: bool = True) -> float:
        """Estimate the px width of a single character"""
        return font_size * (0.6 if is_monospace else 0.5)

    @staticmethod
    def estimate_width(text: str, font_size: int, is_monospace: bool = True) -> float:
        """Estimate the px width of a string. Assume monospace is ~0.6 * size"""
        return len(text) * TextMetrics._char_width(font_size, is_monospace)

    @staticmethod
    def truncate_to_fit(text: str, max_width: float, font_size: int, is_monospace: bool = True) -> str:
        """Truncate text to fit within max_width, adding ellipsis if needed"""
        max_chars = int(max_width / TextMetrics._char_width(font_size, is_monospace))
        if max_chars < 3:
            # Not enough space for any text
            return ""
        if len(text) > max_chars:
            return text[:max_chars - 1] + "…"
        return text


class BaseTrack:
    """Base class for all tracks"""
    def __init__(
        self,
        name: str,
        height: float = 100,
        label_pos: LabelPosition = LabelPosition.LEFT,
        description: str = "",
        theme_kwargs: dict | None = None,
    ):
        self.name = name
        self.height = height
        self.label_pos = label_pos
        self.description = description
        self.theme_kwargs = theme_kwargs or {}

    def render_payload(self, bounds: TrackBounds, width: float, theme: TrackTheme) -> v.SvgGroup:
        """
        Subclasses must implement this. Returns the actual data graphics scaled to 'width'.
        (0,0) is the top-left corner of the track area.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def render(
        self,
        bounds: TrackBounds,
        allocated_width: float,
        global_theme: TrackTheme,
    ) -> tuple[v.SvgGroup, float]:
        """
        Handles the layout boilerplate (labels, descriptions, offsets).
        Returns the rendered group and the _total_ height consumed by the track
        """
        active_theme = dataclasses.replace(global_theme, **self.theme_kwargs)
        group = v.SvgGroup()
        total_height = self.height

        # 1. Calculate X offsets based on label layout
        payload_x = 0.0
        payload_width = allocated_width

        if self.label_pos == LabelPosition.LEFT:
            payload_x = active_theme.label_width
            payload_width = allocated_width - active_theme.label_width - active_theme.right_margin

            # Draw left label
            group.add(v.SvgText(
                x=active_theme.label_width - active_theme.label_margin,
                y=self.height / 2 + (active_theme.font_size * 0.35),  # Vertical centering
                text=TextMetrics.truncate_to_fit(self.name, active_theme.label_width - 10, active_theme.font_size),
                style=v.SvgStyle(
                    text_anchor="end",
                    font_family=active_theme.font_family,
                    font_size=active_theme.font_size,
                    fill=active_theme.font_color,
                ),
            ))

        elif self.label_pos == LabelPosition.TOP:
            payload_width = allocated_width - active_theme.right_margin

            # Draw top label
            group.add(v.SvgText(
                x=allocated_width / 2,
                y=active_theme.font_size,
                text=TextMetrics.truncate_to_fit(self.name, allocated_width, active_theme.font_size),
                style=v.SvgStyle(
                    text_anchor="middle",
                    font_family=active_theme.font_family,
                    font_size=active_theme.font_size,
                    fill=active_theme.font_color,
                ),
            ))
            # Shift payload down to make room for label
            payload_y_offset = active_theme.font_size + active_theme.label_margin
            total_height += payload_y_offset

            # Wrap the payload in an offset group
            payload_wrapper = v.SvgGroup(translate_x=payload_x, translate_y=payload_y_offset)
            payload_wrapper.add(self.render_payload(bounds, payload_width, active_theme))
            group.add(payload_wrapper)

            return group, total_height

        # 3. Add left-aligned payload (default), no y-shift needed
        payload_wrapper = v.SvgGroup(translate_x=payload_x)
        payload_wrapper.add(self.render_payload(bounds, payload_width, active_theme))
        group.add(payload_wrapper)

        return group, total_height


class TrackStack:
    """Master container that scales and stacks tracks vertically"""
    def __init__(self, width: float = 1200, theme: TrackTheme | None = None):
        self.width = width
        self.theme = theme or TrackTheme()
        self.tracks: list[BaseTrack] = []

    def add_track(self, track: BaseTrack):
        self.tracks.append(track)

    def render(self, bounds: TrackBounds) -> str:
        elements: list[v.SvgElement] = []
        y_cursor = self.theme.track_spacing

        for track in self.tracks:
            track_group, track_height = track.render(bounds, allocated_width=self.width, global_theme=self.theme)
            track_group.translate_y = y_cursor

            elements.append(track_group)
            y_cursor += track_height + self.theme.track_spacing

        canvas = v.SvgCanvas(width=self.width, height=y_cursor)
        canvas.elements = elements

        return canvas.render()
