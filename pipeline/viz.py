"""
Make output visualizations for pipeline results.

Classes and functions are organized into three conceptual layers:
Biology: GeneFeature. Genomic features straight from NCBI data
Layout: DisplayFeature. Rather than mutating genomic coordinates, store modifications here
Rendering: RenderFeature. Drawing instructions
"""
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable
import pandas as pd

from pipeline.summary import build_cluster_details, build_cluster_context

@dataclass(frozen=True)
class GeneFeature:
    """A simple structure for gene features in the SVG plot."""
    acc: str | None  # None: allow non-protein/uncharacterized features
    name: str
    start: float
    end: float
    orientation: str  # "plus" | "minus" | ""
    context_pos: int  # position relative to anchor gene (0 = anchor, -1 = 1st 5', +1 = 1st 3', etc.)

@dataclass(frozen=True)
class DisplayFeature:
    """Holds modified display coordinates for a gene feature in the SVG plot."""
    f: GeneFeature
    x0: float
    x1: float

@dataclass(frozen=False)
class RenderFeature:
    """Fully resolved draw instructions for one feature."""
    g: GeneFeature
    x0: float
    x1: float
    shape: str               # "arrow" | "rect"
    direction: str           # "right" | "left" | "none"
    fill: str
    stroke: str = "black"
    dash: str = ""           # e.g. "3,2"
    opacity: float = 1.0

    @property
    def width(self) -> float:
        return abs(self.x1 - self.x0)

    @property
    def center(self) -> float:
        return (self.x0 + self.x1) / 2

def _iter_features_from_row(row: pd.Series, context: int) -> list[GeneFeature]:
    """Convert a DataFrame of genes to a list of GeneFeature objects."""
    feats: list[GeneFeature] = []
    # anchor
    feats.append(
        GeneFeature(
            acc=row.get('protein_accession'),
            name=str(row.get('protein_name', '')),
            start=float(row.get('begin', 0)),
            end=float(row.get('end', 0)),
            orientation=str(row.get('orientation', '')),
            context_pos=0,
        )
    )
    # context (if present)
    for side in ('fiveprime', 'threeprime'):
        for i in range(1, context + 1):
            acc = row.get(f'{side}_{i}_acc')
            name = row.get(f'{side}_{i}_name', '')
            b = row.get(f'{side}_{i}_begin')
            e = row.get(f'{side}_{i}_end')
            o = row.get(f'{side}_{i}_orientation', '')
            pos = i*(-1 if side == 'fiveprime' else 1)
            if pd.isna(b) or pd.isna(e):
                continue
            feats.append(
                GeneFeature(
                    acc=acc, name=str(name), start=float(b), end=float(e),
                    orientation=str(o), context_pos=pos,
                )
            )
    return feats

def _normalize_to_anchor_forward(
    feats: list[GeneFeature],
    flip: bool,
    anchor_start: float,
) -> list[GeneFeature]:
    """
    Make anchor strand effectively "plus" by flipping coordinates if anchor is minus.
    Then shift so anchor_start becomes 0.
    :param feats: anchor and context GeneFeatures for a single assembly
    :param anchor_orient: orientation of the anchor gene ("plus" | "minus" | "")
    :param anchor_start: genomic start position of the anchor gene
    :return: list of GeneFeatures with anchor at position 0 and oriented "plus"
    """
    # shift first (so anchor_start becomes 0)
    shifted = [
        GeneFeature(f.acc, f.name, f.start - anchor_start, f.end - anchor_start, f.orientation, f.context_pos)
        for f in feats
    ]
    if not flip:
        return shifted

    # flip around 0: x -> -x, and swap start/end to keep start <= end
    flipped: list[GeneFeature] = []
    anchor_length = shifted[0].end - shifted[0].start
    for f in shifted:
        s = -f.end + anchor_length
        e = -f.start + anchor_length
        pos = f.context_pos  # position is 5' or 3' relative to anchor, so doesn't change
        if f.orientation == "plus":
            orient = "minus"
        elif f.orientation == "minus":
            orient = "plus"
        else:
            orient = f.orientation
        flipped.append(GeneFeature(f.acc, f.name, s, e, orient, pos))
    return flipped

def layout_track_order(
        feats: list[GeneFeature],
        gene_w: float = 1.0,
        gap: float = 0.2,
) -> list[DisplayFeature]:
    """Features are evenly spaced by order, ignoring genomic distance."""
    out = []
    for f in feats:
        pos = f.context_pos  # anchor 0, 5' negative, 3' positive
        x0 = pos * (gene_w + gap)
        x1 = x0 + gene_w
        out.append(DisplayFeature(f, x0, x1))
    return out

def layout_track_bp(
        feats: list[GeneFeature],
) -> list[DisplayFeature]:
    """Features are scaled by actual genomic coordinates."""
    return [DisplayFeature(f, f.start, f.end) for f in feats]

def layout_track_clamped(
    feats: list[GeneFeature],
    max_gap: float = 100.0,
    clamp_genes: bool = False,
) -> list[DisplayFeature]:
    """Walks the features clamping large genes and large gaps to max_gap"""
    if not feats:  # empty track
        return []
    feats_sorted = sorted(feats, key=lambda f: min(f.start, f.end))
    out: list[DisplayFeature] = []
    x_cursor = 0.0
    last_end = min(feats_sorted[0].start, feats_sorted[0].end)  # initialize at far left
    for f in feats_sorted:
        g0 = min(f.start, f.end)
        g1 = max(f.start, f.end)
        gap = g0 - last_end  # can be negative (overlap/frameshift)
        gap_draw = min(gap, max_gap)  # only clamp positive gaps
        # advance cursor by clamped gap width
        x_cursor += gap_draw
        x0 = x_cursor
        x1 = x0 + min(g1 - g0, max_gap) if clamp_genes else x0 + (g1 - g0)
        out.append(DisplayFeature(f, x0, x1))
        x_cursor = x1
        last_end = g1
    return out

def resolve_display(d: DisplayFeature) -> RenderFeature:
    """Final rendering instructions for a GeneFeature"""
    f = d.f
    # shape/direction
    if f.orientation == 'plus':
        shape, direction = 'arrow', 'right'
    elif f.orientation == 'minus':
        shape, direction = 'arrow', 'left'
    else:
        shape, direction = 'rect', 'none'

    # base style
    fill = 'lightblue'
    stroke = 'black'
    dash = ''
    opacity = 1.0

    # missing accession / uncharacterized -> “maybe”
    uncertain = ['putative', 'hypothetical', 'uncharacterized', 'unknown', 'predicted']
    if any(sub in f.name.lower() for sub in uncertain) or f.acc is None:
        fill = '#c6c8cd'       # light gray
        stroke = '#6b7280'     # gray
    # emphasize anchor gene
    if d.f.context_pos == 0:
        fill = '#FFE19A'

    return RenderFeature(
        g=f, x0=d.x0, x1=d.x1,
        shape=shape, direction=direction,
        fill=fill, stroke=stroke, dash=dash, opacity=opacity,
    )

def _arrow_points(x_tail: float, x_head: float, y: float, h: float) -> str:
    """
    SVG polygon points for an arrow from x_tail -> x_head.
    Works for right- and left-pointing arrows.
    """
    if x_tail == x_head:
        x_head = x_tail + 1.0  # avoid zero-length

    dir_sign = 1.0 if x_head > x_tail else -1.0
    length = abs(x_head - x_tail)
    head_len = min(h, length)
    body_head_x = x_head - dir_sign * head_len  # move back from head toward tail

    y0 = y - h / 2
    y1 = y + h / 2
    # vertices for a pentagonal arrow
    pts = [
        (x_tail, y0),
        (body_head_x, y0),
        (x_head, y),
        (body_head_x, y1),
        (x_tail, y1),
    ]
    return " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)

def _truncate(s: str, n: int = 10) -> str:
    """Truncate string s to at most n characters, including ellipsis"""
    s = s.strip()
    return s if len(s) <= n else s[:n-1] + "…"

def _escape(s: str) -> str:
    """Stops XML characters from breaking SVG output"""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def feature_to_svg(
    f: RenderFeature,
    *,
    y_center: float,
    row_height: float,
    # label_fn: Callable[[GeneFeature], str],
) -> str:
    """Return SVG element string for one RenderFeature."""
    gene_h = row_height * 0.6
    x0, x1 = sorted((f.x0, f.x1))
    dash_attr = f' stroke-dasharray="{f.dash}"' if f.dash else ''
    op_attr = f' opacity="{f.opacity:.2f}"' if f.opacity < 1.0 else ''
    # Tooltip label
    acc_txt = f.g.acc if f.g.acc is not None else "N/A"
    title = _escape(f"{acc_txt} | {f.g.name}")
    if f.shape == 'arrow' and f.direction in ("right", "left"):
        if f.direction == 'right':
            pts = _arrow_points(x0, x1, y_center, gene_h)
        else:  # 'left'
            pts = _arrow_points(x1, x0, y_center, gene_h)
        element = (
            f'<polygon points="{pts}" fill="{f.fill}" stroke="{f.stroke}" stroke-width="1"'
            f'{dash_attr}{op_attr}><title>{title}</title></polygon>'
        )
    else:  # rectangle
        element = (
            f'<rect x="{x0:.2f}" y="{y_center-gene_h/2:.2f}" width="{f.width:.2f}" height="{gene_h:.2f}" '
            f'fill="{f.fill}" stroke="{f.stroke}" stroke-width="1"{dash_attr}{op_attr}>'
            f'<title>{title}</title></rect>'
        )
    # Add label if space allows
    text_space = max(int(f.width * 0.15), 1)
    label = _escape(_truncate(f.g.name, text_space))
    if len(label) >= 4:
        element += (
            f'\n<text x="{f.center:.2f}" y="{y_center + 5:.2f}" '
            f'font-size="10" text-anchor="middle">{label}</text>'
        )
    return element

def render_cluster_svg(
    detail_df: pd.DataFrame,
    layout: str = "bp",
    width: int = 1200,
    row_height: int = 20,
    margin: int = 10,
    context: int = 3,
    max_gap: float = 100.0,
    clamp_genes: bool = False,
) -> str:
    """Return SVG text for a cluster context plot."""
    tracks: list[tuple[str, list[RenderFeature]]] = []
    global_min, global_max = float("inf"), float("-inf")
    # Build normalized tracks and global bounds
    for _, row in detail_df.iterrows():
        anchor_orient = str(row.get("orientation", ""))
        anchor_start = float(row.get('begin', 0))
        # Flip genomic neighborhoods so all anchor genes are oriented the same way
        feats = _iter_features_from_row(row, context)
        # Align all anchor genes to start at position 0
        feats = _normalize_to_anchor_forward(feats, True if anchor_orient == "minus" else False, anchor_start)
        # Display mode
        if layout == "bp":
            display_feats = layout_track_bp(feats)
        elif layout == "clamped":
            display_feats = layout_track_clamped(feats, max_gap=max_gap, clamp_genes=clamp_genes)
        else:  # "order"
            if layout != "order":
                logging.warning("Using 'order' layout; genomic distances will be ignored.")
            display_feats = layout_track_order(feats, gene_w=1.0, gap=0.33)
        render_feats = [resolve_display(d) for d in display_feats]
        # label for the track
        track_label = f"{row.get('assembly_accession', '')} | {row.get('protein_accession', '')}"
        tracks.append((track_label, render_feats))
        for f in render_feats:
            global_min = min(global_min, f.x0, f.x1)
            global_max = max(global_max, f.x0, f.x1)
    if global_min == global_max:
        global_max = global_min + 1.0  # avoid zero-width
    # Plot parameters
    height = row_height * len(tracks) + margin * 2
    label_width = max(len(lbl) for lbl, _ in tracks) * 6.5  # approx. character width
    plot_x0 = margin + label_width
    plot_x1 = width - margin
    plot_width = plot_x1 - plot_x0
    def _xscale(x: float) -> float:
        """Helper to scale x coordinates to SVG space."""
        return plot_x0 + (x - global_min) * (plot_width / (global_max - global_min))
    # SVG parts will be collected here, starting with a header
    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    # Draw each track
    for idx, (label, render_feats) in enumerate(tracks):
        y_top = margin + idx * row_height
        y_center = y_top + row_height / 2
        # Backbone line
        track_min_x = _xscale(min(f.x0 for f in render_feats))
        track_max_x = _xscale(max(f.x1 for f in render_feats))
        parts.append(f'<line x1="{track_min_x}" y1="{y_center:.2f}" x2="{track_max_x}" y2="{y_center:.2f}" stroke="black" stroke-width="1"/>')
        # Draw label
        parts.append(f'<text x="{plot_x0 - 6:.2f}" y="{y_center + 5:.2f}" font-size="12" text-anchor="end">{label}</text>')
        # Genes
        for f in render_feats:
            f.x0 = _xscale(f.x0)
            f.x1 = _xscale(f.x1)
            parts.append(feature_to_svg(f, y_center=y_center, row_height=row_height))
    # Finalize SVG
    parts.append("</svg>")
    return "\n".join(parts)

def save_cluster_svg(
    accession: str,
    cache_dir: Path,
    output_path: Path,
    context: int = 3,
    layout: str = "bp",
    max_gap: float = 100.0,
    clamp_genes: bool = False,
    **svg_kwargs,
) -> str:
    """Generate and save an SVG plot for the cluster details DataFrame."""
    detail_df, ann_dfs = build_cluster_details(accession, cache_dir)
    detail_df = build_cluster_context(detail_df, ann_dfs, context)

    svg_text = render_cluster_svg(detail_df, context=context, layout=layout, max_gap=max_gap, clamp_genes=clamp_genes, **svg_kwargs)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_text)
    logging.info("Wrote cluster SVG plot to %s", output_path)
    return svg_text
