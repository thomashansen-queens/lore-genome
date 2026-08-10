"""
Coordinate scales: the master layout engine of the viz library.

A `Scale` is a piecewise-linear map from a 1D data domain to a pixel range.
This allows a common coordinate system with breaks allowed to compress long spans.

The scale is deliberately axis-agnostic.
"""
import bisect
from enum import StrEnum
from pydantic import BaseModel, model_validator

from . import units


class TrackBounds(BaseModel):
    """Generic 1-D coordinate window (genomic bp, protein aa, value range, etc.)"""
    start: float
    end: float

    @model_validator(mode="before")
    @classmethod
    def _coerce_pair(cls, data):
        """Accept a plain (start, end) tuple/list in addition to a dict/instance."""
        if isinstance(data, (tuple, list)):
            start, end = data
            return {"start": start, "end": end}
        return data

    @property
    def length(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return abs(self.end - self.start)

    def resolve(self, data_min: float, data_max: float) -> "TrackBounds":
        """Autoset bounds to visible range or override with custom domain"""
        start = max(self.start, data_min)
        end = min(self.end, data_max)
        return TrackBounds(start=start, end=end)


class BreakKind(StrEnum):
    """
    What a discontinuity means, which decides how it is drawn:
    - CUT: a typical axis discontuity ('//' zigzag)
    - GAP: Can represent separate fragments, a circular origin, etc. (no zigzag)
    - SPLICE: a clamped region within a feature (intron / clamped gene)
    """
    CUT = "cut"
    GAP = "gap"
    SPLICE = "splice"


class Break(BaseModel):
    """
    A data-space interval [start, end] collapsed to a fixed pixel width (px, )
    """
    start: float
    end: float
    kind: BreakKind = BreakKind.CUT
    width: str | float = "12 px"

    @property
    def span(self) -> float:
        return self.end - self.start

    def resolve_px(self, total_px: float) -> float:
        """Resolve this break's collapsed width to pixels (see `units.resolve_px`)."""
        return units.resolve_px(self.width, total_px)


class BreakMark(BaseModel):
    """A break resolved into pixel space, ready to be drawn as a glyph."""
    x: float  # center of the collapsed region, in pixels
    x_start: float
    x_end: float
    kind: BreakKind


class Scale:
    """
    Piecewise-linear map from a data domain to a pixel range, honoring breaks.
    Automatically handles axis breaks, origin wraps, and strand inversion.
    """
    def __init__(
        self,
        domain: TrackBounds | tuple[float, float] | dict,
        px_width: float,
        breaks: list[Break] | tuple[Break, ...] = (),
        invert: bool = False,
    ):
        self.domain = TrackBounds.model_validate(domain)
        self.px_width = float(px_width)
        self.invert = invert

        # Clip breaks to the visible domain, then sort by start
        clipped: list[Break] = []
        for b in sorted(breaks, key=lambda b: b.start):
            start = max(b.start, self.domain.start)
            end = min(b.end, self.domain.end)
            if end > start:
                clipped.append(b.model_copy(update={"start": start, "end": end}))
        self._breaks = clipped

        # Precompute the (data, pixel) anchor points on initialization
        self._build()

    def _build(self) -> None:
        """Precompute (data, pixel) anchor points; `px()` interpolates between them."""
        break_data = sum(b.span for b in self._breaks)
        break_px = sum(b.resolve_px(self.px_width) for b in self._breaks)

        live_data = max(self.domain.length - break_data, 0.0)
        live_px = max(self.px_width - break_px, 0.0)
        factor = live_px / live_data if live_data > 0 else 0.0

        # Binary search is much more efficient than linear scan for large data ranges
        d_pts: list[float] = [self.domain.start]
        p_pts: list[float] = [0.0]

        cur_data = self.domain.start
        cur_px = 0.0

        for b in self._breaks:
            cur_px += (b.start - cur_data) * factor
            d_pts.append(b.start)
            p_pts.append(cur_px)

            cur_px += b.resolve_px(self.px_width)  # the collapsed break region
            d_pts.append(b.end)
            p_pts.append(cur_px)

            cur_data = b.end

        cur_px += (self.domain.end - cur_data) * factor
        d_pts.append(self.domain.end)
        p_pts.append(cur_px)

        # Flat arrays for bisect search
        self._d_pts = d_pts
        self._p_pts = p_pts

    def px(self, coord: float) -> float:
        """
        Map a single data coordinate to a pixel position within the range.

        Performance Note: 
        Because this method is called inside tight rendering loops (e.g. millions 
        of times for BAM pileups), to avoid O(N) linear scans for axis breaks, 
        we use stdlib `bisect` to perform an O(log N) binary search. This
        instantly finds the correct piecewise segment and its pixel coordinate.
        """
        coord = min(max(coord, self.domain.start), self.domain.end)

        idx = bisect.bisect_right(self._d_pts, coord) - 1

        if idx >= len(self._d_pts) -1:
            value = self._p_pts[-1]
        else:
            d0, d1 = self._d_pts[idx], self._d_pts[idx + 1]
            p0, p1 = self._p_pts[idx], self._p_pts[idx + 1]
            span = d1 - d0
            value = p0 if span == 0 else p0 + (coord - d0) / span * (p1 - p0)

        return (self.px_width - value) if self.invert else value

    def span(self, start: float, end: float) -> tuple[float, float]:
        """Map a data interval to a (left_px, right_px) pair, ordered ascending."""
        a, b = self.px(start), self.px(end)
        return (a, b) if a <= b else (b, a)

    @property
    def marks(self) -> list[BreakMark]:
        """Every break resolved to pixel space, in domain order."""
        out: list[BreakMark] = []
        for b in self._breaks:
            xs, xe = self.px(b.start), self.px(b.end)
            lo, hi = (xs, xe) if xs <= xe else (xe, xs)
            out.append(BreakMark(x=(lo + hi) / 2, x_start=lo, x_end=hi, kind=b.kind))
        return out

    def marks_of(self, kind: BreakKind) -> list[BreakMark]:
        """Every break of a given kind resolved to pixel space, in domain order."""
        return [m for m in self.marks if m.kind == kind]
