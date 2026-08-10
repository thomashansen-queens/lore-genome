"""
A scaled sequence track
"""
from typing import TYPE_CHECKING
from pydantic import Field

from .base import BaseTrack
from ..scale import Scale
from .. import svg as v

if TYPE_CHECKING:
    from .stack import TrackTheme


class SequenceTrack(BaseTrack):
    """
    Draws a sequence to scaled to current view. If too zoomed out, just draws the spine.
    sequence: the nucleotide sequence to draw
    name: name of the track
    start: starting position of the drawn sequence in the original sequence
    chars_per_glyph: multiplier for glyphs, allowing e.g. translated codons (default 1)
    """
    sequence: str
    name: str = "Sequence"
    start: int = 0
    chars_per_glyph: int = 1

    def get_extents(self) -> tuple[float, float]:
        end_pos = self.start + len(self.sequence) * self.chars_per_glyph
        return (float(self.start), float(end_pos))

    def render_payload(self, scale: Scale, theme: "TrackTheme") -> v.SvgGroup:
        group = v.SvgGroup()
        y_center = theme.track_height / 2.0

        # 1. Check zoom level (pixel width vs seq length)
        domain_len = scale.domain.length
        px_per_base = (scale.px_width / domain_len) if domain_len > 0 else 0.0

        min_readable_width = theme.font_size * 0.4

        if px_per_base < min_readable_width:
            # Early exit: Too zoomed out, just draw a spine
            end_pos = self.start + len(self.sequence) * self.chars_per_glyph
            px_start, px_end = scale.span(self.start, end_pos)
            group.add(v.SvgLine(
                x1=px_start, y1=y_center, x2=px_end, y2=y_center,
                stroke=theme.color_default_stroke, stroke_width=1.0,
            ))
            return group

        # 2. Draw letters using the Scale engine
        # Clamp font size so it shrinks with zoom, but never exceeds the theme size
        dynamic_font_size = min(theme.font_size, px_per_base * 1.5)

        # 3. Iterate over the visible range of the sequence    
        seq_end_pos = self.start + len(self.sequence) * self.chars_per_glyph
        track_start = max(self.start, int(scale.domain.start))
        track_end = min(seq_end_pos, int(scale.domain.end))

        # Convert genomic bounds to string indices
        start_idx = max(0, (track_start - self.start) // self.chars_per_glyph)
        end_idx = min(len(self.sequence), (track_end - self.start) // self.chars_per_glyph + 1)

        # 4. Draw each glyph in the visible range
        for seq_idx in range(start_idx, end_idx):
            glyph = self.sequence[seq_idx]

            # Base coordinates for this character
            glyph_start = self.start + (seq_idx * self.chars_per_glyph)
            glyph_end = glyph_start + self.chars_per_glyph

            px_start, px_end = scale.span(glyph_start, glyph_end)
            px_center = (px_start + px_end) / 2.0

            group.add(v.SvgText(
                text=glyph,
                x=px_center, y=y_center,
                font_size=dynamic_font_size,
                fill=theme.color_default_stroke,
                text_anchor="middle",
            ))

        return group
