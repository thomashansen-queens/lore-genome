"""
Make output visualizations for pipeline results.
"""
from dataclasses import dataclass
import logging
from pathlib import Path
import pandas as pd

from pipeline.summary import build_cluster_details, build_cluster_context

@dataclass(frozen=True)
class GeneFeature:
    """A simple structure for gene features in the SVG plot."""
    acc: str
    name: str
    start: float
    end: float
    orientation: str  # "plus" | "minus" | ""

def _iter_features_from_row(row: pd.Series, context: int) -> list[GeneFeature]:
    """Convert a DataFrame of genes to a list of GeneFeature objects."""
    feats: list[GeneFeature] = []
    # anchor
    feats.append(
        GeneFeature(
            acc=str(row.get('protein_accession', '')),
            name=str(row.get('protein_name', '')),
            start=float(row.get('begin', 0)),
            end=float(row.get('end', 0)),
            orientation=str(row.get('orientation', '')),
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
            if pd.isna(acc) or pd.isna(b) or pd.isna(e):
                continue
            feats.append(
                GeneFeature(
                    acc=str(acc),
                    name=str(name),
                    start=float(b),
                    end=float(e),
                    orientation=str(o),
                )
            )
    return feats

def _normalize_to_anchor_forward(
    feats: list[GeneFeature],
    anchor_orient: str,
    anchor_start: float,
) -> list[GeneFeature]:
    """
    Make anchor strand effectively "plus" by flipping coordinates if anchor is minus.
    Then shift so anchor_start becomes 0.
    """
    # shift first (so anchor_start becomes 0)
    shifted = [
        GeneFeature(f.acc, f.name, f.start - anchor_start, f.end - anchor_start, f.orientation)
        for f in feats
    ]

    if anchor_orient != "minus":
        return shifted

    # flip around 0: x -> -x, and swap start/end to keep start <= end
    flipped: list[GeneFeature] = []
    anchor_length = shifted[0].end - shifted[0].start
    for f in shifted:
        s = -f.end + anchor_length
        e = -f.start + anchor_length
        if f.orientation == "plus":
            orient = "minus"
        elif f.orientation == "minus":
            orient = "plus"
        else:
            orient = f.orientation
        flipped.append(GeneFeature(f.acc, f.name, s, e, orient))
    return flipped

def _arrow_points(x_tail: float, x_head: float, y: float, h: float, head_px: float) -> str:
    """
    SVG polygon points for an arrow from x_tail -> x_head.
    Works for right- and left-pointing arrows.
    """
    if x_tail == x_head:
        x_head = x_tail + 1.0  # avoid zero-length

    dir_sign = 1.0 if x_head > x_tail else -1.0
    length = abs(x_head - x_tail)

    head_len = min(head_px, max(1.0, length * 0.6))
    body_head_x = x_head - dir_sign * head_len  # move back from head toward tail

    y0 = y - h / 2
    y1 = y + h / 2

    pts = [
        (x_tail, y0),
        (body_head_x, y0),
        (x_head, y),
        (body_head_x, y1),
        (x_tail, y1),
    ]
    return " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)
# def _arrow_points(x0: float, x1: float, y: float, h: float, head_px: float) -> str:
#     """
#     Return SVG polygon points for an arrow from x0 to x1.
#     """
#     length = abs(x1 - x0)
#     head_len = min(head_px, max(0.0, length * 0.6))  # head shorter than 60% of l
#     body_end = x1 - head_len

#     y0 = y - h / 2
#     y1 = y + h / 2
#     ym = y

#     pts = [
#         (x0, y0),
#         (body_end, y0),
#         (body_end, y0),
#         (x1, ym),
#         (body_end, y1),
#         (x0, y1),
#     ]
#     return " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)

def _truncate(s: str, n: int = 10) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n-1] + "…"

def render_cluster_svg(
    detail_df: pd.DataFrame,
    *,
    width: int = 1200,
    row_height: int = 20,
    margin: int = 10,
    context: int = 3,
) -> str:
    """Return SVG text for a cluster context plot."""
    tracks: list[tuple[str, list[GeneFeature]]] = []
    global_min = float("inf")
    global_max = float("-inf")
    # Build normalized tracks and global bounds
    for _, row in detail_df.iterrows():
        anchor_orient = str(row.get("orientation", ""))
        anchor_start = float(row.get('begin', 0))
        # Flip genomic neighborhoods so all anchor genes are oriented the same way
        feats = _iter_features_from_row(row, context)
        # Align all anchor genes to start at position 0
        feats = _normalize_to_anchor_forward(feats, anchor_orient, anchor_start)
        # label for the track
        track_label = f"{row.get('assembly_accession', '')} | {row.get('protein_accession', '')}"
        tracks.append((track_label, feats))
        for f in feats:
            global_min = min(global_min, f.start, f.end)
            global_max = max(global_max, f.start, f.end)
    if global_min == global_max:
        global_max = global_min + 1.0  # avoid zero-width
    # Plot parameters
    height = row_height * len(tracks) + margin * 2
    label_width = len(track_label) * 6.5
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
    for idx, (label, feats) in enumerate(tracks):
        y_top = margin + idx * row_height
        y_center = y_top + row_height / 2
        # Backbone line
        track_min_x = _xscale(min(f.start for f in feats))
        track_max_x = _xscale(max(f.end for f in feats))
        parts.append(f'<line x1="{track_min_x}" y1="{y_center:.2f}" x2="{track_max_x}" y2="{y_center:.2f}" stroke="black" stroke-width="1"/>')
        # Draw label
        parts.append(f'<text x="{plot_x0 - 6:.2f}" y="{y_center + 5:.2f}" font-size="12" text-anchor="end">{label}</text>')
        # Genes
        for f in feats:
            x0, x1 = sorted((_xscale(f.start), _xscale(f.end)))
            gene_width = max(x1 - x0, 1.0)
            text_space = max(int(gene_width * 0.15), 1)
            # parts.append(f'<rect x="{x0:.2f}" y="{y_top + 2:.2f}" width="{gene_width:.2f}" height="{row_height - 4}" fill="lightblue" stroke="black" stroke-width="1"/>')
            gene_h = row_height * 0.6
            head_px = gene_h
            safe_label = f"{f.acc} | {f.name}".replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if f.orientation == 'plus':  # right-facing arrow
                pts = _arrow_points(x_tail=x0, x_head=x1, y=y_center, h=gene_h, head_px=head_px)
                parts.append(
                    f'<polygon points="{pts}" fill="lightblue" stroke="black" stroke-width="1">'
                    f'<title>{safe_label}</title></polygon>'
                )
            elif f.orientation == 'minus':  # left-facing arrow
                pts = _arrow_points(x_tail=x1, x_head=x0, y=y_center, h=gene_h, head_px=head_px)
                parts.append(
                    f'<polygon points="{pts}" fill="lightblue" stroke="black" stroke-width="1">'
                    f'<title>{safe_label}</title></polygon>'
                )
            else:  # no orientation? draw rectangle
                parts.append(f'<rect x="{x0:.2f}" y="{y_center - gene_h / 2:.2f}" width="{gene_width:.2f}" height="{gene_h:.2f}"'
                             f' fill="lightblue" stroke="black" stroke-width="1"><title>{safe_label}</title></rect>')
            if gene_width >= 12:  # don't label tiny stubs of genes
                parts.append(f'<text x="{(x0 + x1) / 2:.2f}" y="{y_center + 5:.2f}"'
                            f' font-size="10" text-anchor="middle">{_truncate(f.name, text_space)}</text>')
    # Finalize SVG
    parts.append("</svg>")
    return "\n".join(parts)

def save_cluster_svg(
    accession: str,
    cache_dir: Path,
    output_path: Path,
    context: int = 3,
    **svg_kwargs,
) -> str:
    """Generate and save an SVG plot for the cluster details DataFrame."""
    detail_df, ann_dfs = build_cluster_details(accession, cache_dir)
    detail_df = build_cluster_context(detail_df, ann_dfs, context)

    svg_text = render_cluster_svg(detail_df, context=context, **svg_kwargs)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_text)
    logging.info("Wrote cluster SVG plot to %s", output_path)
    return svg_text
