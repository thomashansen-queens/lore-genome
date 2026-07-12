"""
Pileup visualization to visualize density and positioning of features mapped
to a 1D reference.
"""
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING
from .base import BaseTrack
from .feature import Feature, FeatureShape
from ..scale import Scale
from .. import text
from .. import svg as v

if TYPE_CHECKING:
    from .stack import TrackTheme


def calculate_coverage(features: list[Feature]) -> list[tuple[float, float, int]]:
    """
    Sweep-line algorithm: https://en.wikipedia.org/wiki/Sweep_line_algorithm
    To map coordinate to coverage depth.
    Returns a list of contiguous blocks: (start, end, depth)
    """
    if not features:
        return []

    # as the line sweeps, record start/finish 'events' of features
    events = []
    for f in features:
        events.append((f.start, 1))  # 1 for start
        events.append((f.end, -1))   # -1 for end

    # Sort events by coordinate. If tied, process removals (-1) before additions (1)
    # to avoid artifical depth spikes at abutted boundaries
    events.sort(key=lambda x: (x[0], x[1]))

    coverage = []
    current_depth = 0
    last_pos = events[0][0]

    for pos, delta in events:
        if pos != last_pos:
            if current_depth > 0:
                coverage.append((last_pos, pos, current_depth))
            last_pos = pos
        current_depth += delta

    return coverage


class SortStrategy(StrEnum):
    """Sorting strategies for pileup tracks."""
    START = "start"  # standard greedy left-to-right (optimally dense)
    MIDPOINT = "midpoint"  # syemmetric mountain-like view
    LENGTH = "length"  # longest reads first
    STRAND = "strand"  # forwards and reverse reads go together


def pack_features(
    features: list[Feature],
    min_gap: float = 0.0,
    sort_strategy: SortStrategy = SortStrategy.START,
) -> tuple[list[tuple[Feature, int]], int]:
    """
    Assigns lanes to features to avoid visual overlap. Strategy can be the optimal
    "greedy algorithm" or others for visualization.
    min_gap: minimum gap between features in the same lane (avoid visual overlap)

    Returns features with their y-axis lane index, total lane count [(feature, lane_index), total]
    """
    def get_sort_key(f: Feature):
        length = f.end - f.start
        if sort_strategy == SortStrategy.START:
            return (f.start, -length)
        elif sort_strategy == SortStrategy.MIDPOINT:
            return ((f.start + f.end) / 2.0, -length)
        elif sort_strategy == SortStrategy.LENGTH:
            return (-length, f.start)
        elif sort_strategy == SortStrategy.STRAND:
            is_forward = f.shape == FeatureShape.ARROW_RIGHT
            return (not is_forward, f.start, -length)
        else:
            # fallback to default rather than raising an error
            return (f.start, -length)

    # 1. Sort by chosen strategy (default: start position)
    sorted_feats = sorted(features, key=get_sort_key)
    lane_ends = []
    packed = []

    # 2. Pack into lanes
    for feat in sorted_feats:
        placed = False
        for i, lane_end in enumerate(lane_ends):
            # optional padding between features to avoid visual overlap
            if feat.start >= lane_end + min_gap:
                lane_ends[i] = feat.end
                packed.append((feat, i))
                placed = True
                break

        if not placed:
            lane_ends.append(feat.end)
            packed.append((feat, len(lane_ends) - 1))

    return packed, len(lane_ends)


class PileupTrack(BaseTrack):
    """
    A track that visualizes the density and positioning of features mapped to a
    1D reference.
    features: List of features to visualize
    packing_gap: Minimum gap between features in the same lane (px)
    lane_padding_ratio: Fraction of lane height to pad between lanes
    max_lanes: Optionally limit the number of lanes to display
    min_lane_height: Minimum height of each lane (px)
    """
    features: list[Feature]
    packing_gap: float = 1.0
    lane_padding_ratio: float = 0.10  # fraction of lane height to pad between lanes

    # Optional limits for dense pileups
    max_lanes: int | None = None
    max_height: float | None = None
    min_lane_height: float = 4.0
    max_lane_height: float | None = None
    lane_height: float | None = None  # overrides dynamic lane height calcs
    sort_strategy: SortStrategy = SortStrategy.START

    @cached_property
    def packing(self) -> tuple[list[tuple[Feature, int]], int]:
        """Pack features into lanes and return the packed list and total lane count"""
        return pack_features(
            self.features,
            min_gap=self.packing_gap,
            sort_strategy=self.sort_strategy,
        )

    def resolve_height(self, theme: "TrackTheme") -> float:
        """
        Expand track hieght if the pileup is too tall.
        Manual height overrides this behaviour.
        """
        _, num_lanes = self.packing
        if self.max_lanes:
            num_lanes = min(num_lanes, self.max_lanes)
        num_lanes = max(1, num_lanes)  # avoid divide-by-zero for empty pileups

        # 1. Fixed height override
        if self.lane_height is not None:
            return num_lanes * self.lane_height

        # 2. Dynamic height based on number of lanes and theme defaults
        required_height = num_lanes * self.min_lane_height
        resolved_height = max(theme.track_height, required_height)

        if self.max_lane_height is not None:
            max_allowed_track_height = num_lanes * self.max_lane_height
            resolved_height = min(resolved_height, max_allowed_track_height)

        # 3. Global max height override
        if self.max_height is not None:
            resolved_height = min(resolved_height, self.max_height)

        return resolved_height

    def get_extents(self) -> tuple[float, float]:
        """The track's data footprint: the feature span."""
        if not self.features:
            return (0.0, 0.0)
        features_min = min(f.start for f in self.features)
        features_max = max(f.end for f in self.features)
        return features_min, features_max

    def get_depth_profile(self) -> list[tuple[float, float, int]]:
        """Return a list of contiguous blocks: (start, end, depth)"""
        return calculate_coverage(self.features)

    def render_payload(self, scale: Scale, theme: "TrackTheme") -> v.SvgGroup:
        """
        Render the payload for the pileup track.
        """
        group = v.SvgGroup()
        if not self.features:
            return group

        # 1. Calculate layout (optional truncation)
        packed_features, num_lanes = self.packing
        if self.max_lanes:
            num_lanes = min(num_lanes, self.max_lanes)
        num_lanes = max(1, num_lanes)  # avoid divide-by-zero for empty pileups

        # 2. Y-axis scaling: lane height is dynamic based on number of lanes
        actual_track_height = self.resolve_height(theme)

        if self.lane_height is not None:
            lane_height = self.lane_height
        else:
            lane_height = actual_track_height / num_lanes

        visual_padding = lane_height * self.lane_padding_ratio
        box_height = max(1.0, lane_height - visual_padding)

        # 3. Draw features
        for feat, lane_idx in packed_features:
            if self.max_lanes and lane_idx >= self.max_lanes:
                continue

            if feat.end < scale.domain.start or feat.start > scale.domain.end:
                continue  # skip features outside the visible domain

            px_start, px_end = scale.span(feat.start, feat.end)

            fill = feat.fill or (theme.color_highlight_fill if feat.highlight else theme.color_default_fill)
            stroke = feat.stroke or (theme.color_highlight_stroke if feat.highlight else theme.color_default_stroke)

            # 4. Tooltip metadata for the feature
            feat_group = v.SvgGroup(classes=["has-tooltip"] if feat.metadata else [], data=feat.metadata)
            if feat.metadata:
                hover_lines = [f"{k}: {v}" for k, v in feat.metadata.items()]
                feat_group.add(v.SvgTitle(text="\n".join(hover_lines)))

            y_top = lane_idx * lane_height + visual_padding / 2

            if feat.shape == FeatureShape.BOX:
                feat_group.add(v.SvgRect(
                    x=px_start, y=y_top,
                    width=max(1.0, px_end - px_start), height=box_height,
                    fill=fill, stroke=stroke, stroke_width=0.5,
                ))
            else:
                feat_group.add(v.SvgArrow(
                    x_start=px_start, x_end=px_end, y_center=y_top + box_height / 2,
                    thickness=box_height,
                    forward=(feat.shape == FeatureShape.ARROW_RIGHT),
                    fill=fill, stroke=stroke, stroke_width=0.5,
                ))

            # 5. Label if there is enough space (both x and y)
            if feat.label:
                if box_height >= theme.font_size * 1.1:
                    avail = abs(px_end - px_start)

                    label_txt = text.truncate_to_fit(feat.label, avail, theme.font_size)
                    if label_txt:
                        feat_group.add(v.SvgText(
                            x=(px_start + px_end) / 2,
                            y=y_top + box_height / 2 + theme.font_size * 0.35,
                            text=label_txt,
                            text_anchor="middle",
                            font_family=theme.font_family,
                            font_size=theme.font_size,
                            fill=theme.font_color,
                        ))

            # 6. Add the feature group to the main group
            group.add(feat_group)

        return group
