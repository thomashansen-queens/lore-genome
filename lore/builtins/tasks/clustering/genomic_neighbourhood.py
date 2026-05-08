"""
Analyzes syntenic neighbourhood of a given gene across a set of genome.
"""
import pandas as pd
from typing import Literal

import lore.dsl as lore
from lore import viz as v


class GenomicNeighbourhoodTaskInputs:
    """Inputs for genomic neighbourhood analysis"""
    protein_accessions = lore.ArtifactInput(
        label="Protein accessions",
        description="The protein accession(s) for the gene(s) of interest",
        accepted_data=["protein_accession"],
        select=lore.MULTIPLE,
        load_as=lore.ADAPTED,
    )
    genome_annotations = lore.ArtifactInput(
        label="Genome annotations",
        accepted_data=["ncbi_annotation_packages", "genome_annotations"],
        select=lore.MULTIPLE,
        load_as=lore.ADAPTED,
    )
    context_window_str = lore.ValueInput(
        str | None,
        description="How far up/downstream of the gene of interest to include in the neighbourhood analysis.",
        default="",
        label="Context window size",
        examples=["3, 5", "5000 (if using base pair context)"],
    )
    context_window_type = lore.ValueInput(
        str | None,
        options=["gene_features", "base_pairs"],
        widget="radio",
        description="Set window size in terms of number of gene features or number of base pairs.",
        default="gene_features",
        label="Context window type",
    )
    clamp_gap = lore.ValueInput(
        int | None,
        description="The maximum distance in base pairs to draw to scale. If the distance between two genes exceeds this value, it will be drawn as a broken axis of this length.",
        default=None,
        label="Clamp Gap Size (bp)",
    )
    collapse_replicons = lore.ValueInput(
        bool,
        description="If True, each replicon/chromosome will have one track, showing the neighbourhood for each unique gene found on it. If false, duplicates/paralogues are shown as separate tracks.",
        default=True,
        label="Collapse replicons",
    )
    save_report = lore.ValueInput(
        bool,
        description="Whether to write the genomic neighbourhood to a new report file.",
        default=False,
        label="Write report",
    )


class GenomicNeighbourhoodTaskOutputs:
    """Outputs for genomic neighbourhood analysis"""
    neighbourhood_svg = lore.TaskOutput(
        data_type="svg",
        label="Genomic neighbourhood visualization",
        description="An SVG visualization of the genomic neighbourhood of the gene of interest across the input genomes",
        is_primary=True,
    )
    neighbourhood_report = lore.TaskOutput(
        data_type="genomic_neighbourhood_report",
        label="Genomic Neighbourhood Report",
        description="A report summarizing the genomic neighbourhood of the gene of interest across the input genomes.",
    )


def _set_window(context_window: str) -> tuple[int, int]:
    """
    Parses the context window input and returns the up and downstream window sizes as integers.
    """
    if "," in context_window:
        if context_window.count(",") > 1:
            raise ValueError("Context window should be a single number or two numbers separated by a single comma. Make sure you're not using commas as thousand separators!")
        window_up, window_down = map(int, context_window.split(","))
    else:
        window_up = window_down = int(context_window)

    return abs(window_up), abs(window_down)


def _neighbourhood_by_feature(
    annot_df: pd.DataFrame,
    anchor_idx: int,
    context_window: tuple[int, int],
) -> pd.DataFrame:
    """
    Extracts a neighbourhood of genes around a given anchor gene in a DataFrame 
    of annotations. Unwraps the replicon in case of wrap-around and adds a
    `context_pos` column to indicate position of gene feature relative to the
    anchor.
    """
    # 1. Orientation-awareness
    anchor_row = annot_df.loc[anchor_idx]
    anchor_orient = annot_df.at[anchor_idx, "orientation"]
    win_up, win_down = context_window

    if anchor_orient == "minus":
        # "upstream" is to the right (+)
        start_offset = -win_down
        end_offset = win_up
        # Context pos reverses: leftmost gene is biologically downstream
        ctx_pos = list(range(win_down, -win_up - 1, -1))
    else:
        start_offset = -win_up
        end_offset = win_down
        ctx_pos = list(range(-win_up, win_down + 1))

    # 2. Isolate replicon containing anchor gene
    replicon_df = annot_df[annot_df["chromosome"] == anchor_row["chromosome"]].reset_index(drop=True)
    replicon_length = replicon_df["end"].max()

    # ...then reset index to replicon (using coordinates in case of duplicates)
    rel_anchor_idx = replicon_df[
        (replicon_df["begin"] == anchor_row["begin"]) &
        (replicon_df["end"] == anchor_row["end"])
    ].index[0]
    total_genes = len(replicon_df)

    # 3. Quick extract by index, wrapping if needed
    target_indices = [(rel_anchor_idx + i) % total_genes for i in range(start_offset, end_offset + 1)]
    neighbourhood_df = replicon_df.iloc[target_indices].copy()
    neighbourhood_df["context_pos"] = ctx_pos

    # 4. Detect and unwrap replicon if there is a wrap-around
    prev = 0
    for s in neighbourhood_df["begin"]:
        if s < prev:
            neighbourhood_df["begin"] = neighbourhood_df["begin"].apply(
                lambda x: x + replicon_length if x < s else x
            )
            neighbourhood_df["end"] = neighbourhood_df["end"].apply(
                lambda x: x + replicon_length if x < s else x
            )
            break
        prev = s

    return neighbourhood_df


def _neighbourhood_by_base_pairs(
    annot_df: pd.DataFrame,
    anchor_idx: int,
    context_window: tuple[int, int],
) -> pd.DataFrame:
    """
    Extracts a neighbourhood of genes around a given anchor gene in a DataFrame 
    of annotations, using base pair distance.
    """
    # 1. Orientation-awareness
    anchor_row = annot_df.iloc[anchor_idx]
    anchor_orient = anchor_row["orientation"]
    anchor_start = anchor_row["begin"]
    anchor_end = anchor_row["end"]
    win_up, win_down = context_window

    if anchor_orient == "minus":
        search_start = anchor_start - win_down
        search_end = anchor_end + win_up
    else:
        search_start = anchor_start - win_up
        search_end = anchor_end + win_down

    # 2. Find all genes within the context window
    neighbourhood_df = annot_df[
        (annot_df["chromosome"] == anchor_row["chromosome"]) &
        (annot_df["begin"] <= search_end) &
        (annot_df["end"] >= search_start)
    ].copy()

    # 3. Strand-aware context position
    if anchor_orient == "minus":
        neighbourhood_df["context_pos"] = anchor_idx - neighbourhood_df.index
    else:
        neighbourhood_df["context_pos"] = neighbourhood_df.index - anchor_idx

    return neighbourhood_df


def _normalize_neighbourhood(neighbourhood: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the coordinates of genes in the neighbourhood relative to the anchor gene.
    Zero-centers the coordinates on the anchor gene and flips the coordinates for genes on the minus strand.
    """
    anchor_row = neighbourhood[neighbourhood["context_pos"] == 0].iloc[0]
    anchor_orient = anchor_row["orientation"]
    anchor_start = anchor_row["begin"]
    anchor_end = anchor_row["end"]

    if anchor_orient == "plus":
        neighbourhood["begin"] = neighbourhood["begin"] - anchor_start
        neighbourhood["end"] = neighbourhood["end"] - anchor_start
    else:
        new_begin = anchor_end - neighbourhood["end"]
        new_end = anchor_end - neighbourhood["begin"]

        neighbourhood["begin"] = new_begin
        neighbourhood["end"] = new_end
        neighbourhood["orientation"] = neighbourhood["orientation"].apply(lambda x: "minus" if x == "plus" else "plus")

    return neighbourhood


@lore.memoize(prefix="master_df", ignore="annotations")
def _build_master_df(ctx: lore.ExecutionContext, annotations: list[dict], cache_key: str):
    """
    Builds a master DataFrame from the input annotations. Memoized based on the 
    IDs of input annotation artifacts to avoid repeated processing.
    """
    df = pd.DataFrame(annotations)
    df[["begin", "end", "protein_length"]] = df[["begin", "end", "protein_length"]].astype("Int64")
    # if chromosome info is missing, treat whole assembly as one replicon
    # df["chromosome"] = df["chromosome"].fillna(df["genome_accession"])
    df["chromosome"] = df["chromosome"].fillna("unknown_chromosome")
    return df.sort_values(by=["genome_accession", "begin"]).reset_index(drop=True)


def _extract_neighbourhoods(
    ctx: lore.ExecutionContext,
    annotation_df: pd.DataFrame,
    accessions: list[str],
    window: tuple[int, int],
    window_type: str,
    collapse_replicons: bool,
) -> pd.DataFrame:
    """
    Extracts the genomic neighbourhood for each input accession across the input 
    annotation DataFrame.
    """
    neighbourhood_list = []
    acc_set = set(accessions)  # fast lookup

    # Extract neighbourhoods for each accession into individual DataFrames
    if not collapse_replicons:
        # Option A: One track per accession instance
        for acc in acc_set:
            anchor_rows = annotation_df[annotation_df["protein_accession"] == acc]
            for anchor_idx in anchor_rows.index:
                if window_type == "gene_features":
                    nb = _neighbourhood_by_feature(annotation_df, anchor_idx, window)
                else:
                    nb = _neighbourhood_by_base_pairs(annotation_df, anchor_idx, window)

                nb = _normalize_neighbourhood(nb)
                nb["track_id"] = f"{annotation_df.loc[anchor_idx, 'genome_accession']}_{anchor_idx}"
                neighbourhood_list.append(nb)

    else:
        # Option B: Group by replicon, zero on first occurence of any anchor
        all_anchor_rows = annotation_df[annotation_df["protein_accession"].isin(acc_set)]

        # 1. Group anchors found on the same replicon together
        for (genome, chrom), chrom_group in all_anchor_rows.groupby(["genome_accession", "chromosome"]):
            chrom_nbs = []

            for anchor_idx in chrom_group.index:
                if window_type == "gene_features":
                    nb = _neighbourhood_by_feature(annotation_df, anchor_idx, window)
                else:
                    nb = _neighbourhood_by_base_pairs(annotation_df, anchor_idx, window)

                # Tag anchor genes
                nb["is_anchor"] = nb["protein_accession"].isin(acc_set)
                chrom_nbs.append(nb)

            # 2. Combine and discard overlapping background genes
            combined_nb = pd.concat(chrom_nbs, ignore_index=True)
            combined_nb = combined_nb.drop_duplicates(subset=["protein_accession", "begin", "end"]).copy()

            # 3. Find the most upstream anchor to act as the primary reference
            anchor_orient = combined_nb[combined_nb["is_anchor"]].iloc[0]["orientation"]

            if anchor_orient == "minus":
                combined_nb = combined_nb.sort_values(by="end", ascending=False).reset_index(drop=True)
            else:
                combined_nb = combined_nb.sort_values(by="begin", ascending=True).reset_index(drop=True)
            primary_anchor_idx = combined_nb[combined_nb["is_anchor"]].index[0]

            # 4. Temporarily hack `context_pos` so primary anchor is the reference point
            combined_nb["context_pos"] = -1
            combined_nb.loc[primary_anchor_idx, "context_pos"] = 0

            # 5. Shift the entire track relative to the primary anchor
            combined_nb = _normalize_neighbourhood(combined_nb)

            # 6. Restore anchor highlight status for all anchors
            combined_nb.loc[combined_nb["is_anchor"], "context_pos"] = 0

            # 7. Assign context pos based on distance to nearest anchor (if tied, use positive distance)
            # TODO: Use begin/end coordinates rather than index-based distance
            anchor_indices = combined_nb[combined_nb["is_anchor"]].index.tolist()
            for i in combined_nb.index:
                if not combined_nb.loc[i, "is_anchor"]:
                    # Find closest anchor by array index
                    closest_anchor = min(anchor_indices, key=lambda x: abs(x - i))
                    combined_nb.loc[i, "context_pos"] = i - closest_anchor

            # 8. Tag with a unique Track ID for rendering (one track per replicon)
            combined_nb["track_id"] = f"{genome}_{chrom}"

            combined_nb = combined_nb.drop(columns=["is_anchor"])
            neighbourhood_list.append(combined_nb)

    if not neighbourhood_list:
        raise ValueError("No valid neighbourhoods found for the given accessions.")

    return pd.concat(neighbourhood_list, ignore_index=True)


@lore.task(
    "analysis.genomic_neighbourhood",
    name="Genomic neighbourhood",
    inputs=GenomicNeighbourhoodTaskInputs,
    outputs=GenomicNeighbourhoodTaskOutputs,
    category="clustering",
    icon="☷",
    live_preview=True,
)
def genomic_neighbourhood_analysis(
    ctx: lore.ExecutionContext,
    protein_accessions: list[str],
    genome_annotations: list[dict],
    context_window_str: str | None = None,
    context_window_type: Literal["gene_features", "base_pairs"] = "gene_features",
    save_report: bool = False,
    clamp_gap: int | None = None,
    collapse_replicons: bool = False,
):
    """
    Analyze the genomic neighbourhood of gene(s) of interest across a set of genomes.
    Allows multiple annotation files mostly to help visualize the same gene in 
    different strains/species
    """
    # Defaults
    if not context_window_str:
        context_window_str = "5" if context_window_type == "gene_features" else "5000"
    context_window = _set_window(context_window_str.strip())

    cache_key = "_".join(a.id for a in ctx.input_artifacts.get("genome_annotations", []))
    annotation_df = _build_master_df(ctx, annotations=genome_annotations, cache_key=cache_key)

    neighbourhoods = _extract_neighbourhoods(
        ctx=ctx,
        annotation_df=annotation_df,
        accessions=protein_accessions,
        window=context_window,
        window_type=context_window_type,
        collapse_replicons=collapse_replicons,
    )

    if save_report:
        out_path = ctx.get_temp_path("genomic_neighbourhood_report.csv")
        neighbourhoods.to_csv(out_path, index=False)
        ctx.materialize_file(
            out_path,
            name="neighbourhood_report",
            output_key="neighbourhood_report",
        )

    svg_str = _render_neighbourhood_svg(neighbourhoods, clamp_gap)
    ctx.materialize_content(
        svg_str,
        name="neighbourhood_view",
        output_key="neighbourhood_svg",
        data_type="svg",
        extension="svg",
    )
    ctx.logger.info("Genomic neighbourhood analysis complete.")


def _render_neighbourhood_svg(
    df: pd.DataFrame,
    clamp_gap: int | None = None,
    collapse_duplicates: bool = False,
    svg_theme: dict | None = None,
) -> str:
    """
    Renders the genomic neighbourhood as an SVG string. If `clamp_gap` is provided,
    distances between genes that exceed this value will be visually clamped to 
    this maximum distance.
    """
    config = SVG_CONFIG.copy()
    if svg_theme:
        config.update(svg_theme)

    if df.empty:
        return v.SvgCanvas(width=config["canvas_width"], height=200).render()    
    tracks_data = df.groupby("track_id")

    # 1. Local scale: Clamp sizes if set
    if clamp_gap is not None:
        # TODO: Implement layout_track_clamped logic here
        # Adjusts `begin` and `end` coordinates in the DataFrame
        pass

    # 2. Global scale: Determine overall span
    global_min = df[["begin", "end"]].min().min()
    global_max = df[["begin", "end"]].max().max()
    if global_min == global_max:
        global_max = global_min + 1.0  # Divide-by-zero guard

    plot_width = config["canvas_width"] - config["label_margin"] - config["right_margin"]
    
    def _xscale(bp: float) -> float:
        """Translates a base-pair coordinate to a pixel X-coordinate"""
        percent_of_span = (bp - global_min) / (global_max - global_min)
        return config["label_margin"] + percent_of_span * plot_width

    # 3. Canvas setup
    row_height = config["row_height"]

    canvas_height = (len(tracks_data) * row_height) + config["vert_margin"] * 2
    canvas = v.SvgCanvas(width=config["canvas_width"], height=canvas_height)

    #4. Draw tracks
    for idx, (genome_acc, track_df) in enumerate(tracks_data):
        genome_acc = track_df["genome_accession"].iloc[0]  # in case of duplicates

        anchor_matches = track_df[track_df["context_pos"] == 0]
        anchor_acc = anchor_matches["protein_accession"].iloc[0] if not anchor_matches.empty else "unknown_anchor"

        y_top = config["vert_margin"] + idx * row_height
        y_center = y_top + (row_height / 2)

        track_group = v.SvgGroup(classes=[f"track-{genome_acc}"])

        # A. Track label
        label_text = f"{genome_acc} | {anchor_acc}"
        track_group.add(v.SvgText(
            x=config["label_margin"] - 15,
            y=y_center + 4,  # eyeball centering
            text=label_text,
            style=v.SvgStyle(
                text_anchor="end",
                font_size=config["font_size"],
                font_family=config["font_family"],
            ),
        ))

        # B. Backbone line
        track_min_x = _xscale(track_df[["begin", "end"]].min().min())
        track_max_x = _xscale(track_df[["begin", "end"]].max().max())
        track_group.add(v.SvgLine(
            x1=track_min_x, y1=y_center, x2=track_max_x, y2=y_center,
            style=v.SvgStyle(stroke=config["color_backbone"], stroke_width=2.0),
        ))

        # C. Gene arrows
        for _, gene in track_df.iterrows():
            px_start = _xscale(gene["begin"])
            px_end = _xscale(gene["end"])

            acc = str(gene.get("protein_accession", ""))
            symbol = str(gene["symbol"]) if pd.notna(gene.get("symbol")) else ""
            name = str(gene["name"]) if pd.notna(gene.get("name")) else ""
            locus = str(gene["locus_tag"]) if pd.notna(gene.get("locus_tag")) else ""

            display_label = symbol or name or locus or acc or "unknown"
            gene_group = v.SvgGroup(classes=["gene-container"])

            # i. Hover tooltip
            hover_txt = (
                f"Accession: {acc}\nSymbol: {symbol}\nName: {name}\nLocus tag: {locus}"
                f"\nLocation: {format(gene['begin'], ',d')} – {format(gene['end'], ',d')}"
            )
            gene_group.add(v.SvgTitle(text=hover_txt.strip()))

            # ii. Polygon arrow
            arrow_h = row_height * 0.5  # thickness
            head_w = min(10.0, abs(px_end - px_start))  # arrowhead can't exceed gene length

            y0 = y_center - (arrow_h / 2)
            y1 = y_center + (arrow_h / 2)

            if gene["orientation"] == "plus":
                pts = [
                    (px_start, y0),
                    (px_end - head_w, y0),
                    (px_end, y_center),
                    (px_end - head_w, y1),
                    (px_start, y1),
                ]
            else:
                pts = [
                    (px_end, y0),
                    (px_start + head_w, y0),
                    (px_start, y_center),
                    (px_start + head_w, y1),
                    (px_end, y1),
                ]

            # Color features (anchor gene is highlighted)
            is_anchor = (gene["context_pos"] == 0)
            fill_color = config["color_anchor_fill"] if is_anchor else config["color_context_fill"]
            stroke_color = config["color_anchor_stroke"] if is_anchor else config["color_context_stroke"]

            arrow = v.SvgPolygon(
                points=pts,
                classes=["gene-arrow", "anchor-gene" if is_anchor else "context-gene"],
                data={
                    "accession": gene.get("protein_accession", ""),
                    "symbol": gene.get("symbol", ""),
                    "locus-tag": gene.get("locus_tag", ""),
                    "start": int(gene["begin"]) if pd.notna(gene["begin"]) else "",
                    "end": int(gene["end"]) if pd.notna(gene["end"]) else "",
                    "title": hover_txt.strip(),
                },
                style=v.SvgStyle(fill=fill_color, stroke=stroke_color, stroke_width=1.0),
            )
            gene_group.add(arrow)

            # iii. Text label (if space allows)
            def _trim_label(text: str, width_px: int) -> str:
                """Trims labels to fit if possible"""
                char_width = config["font_size"] * 0.6
                max_chars = int(width_px / char_width)
                if max_chars < 3:
                    return ""
                elif len(text) > max_chars:
                    return text[:max_chars - 1] + "…"
                else:
                    return text

            trimmed_label = _trim_label(display_label, abs(int(px_end - px_start)))
            if trimmed_label:
                # Use white text on the dark anchor background for readability
                text_color = "#FFFFFF" if is_anchor else "#333333"
                label = v.SvgText(
                    x=(px_start + px_end) / 2,
                    y=y_center + (config["font_size"] * 0.35), # True vertical centering for text
                    text=trimmed_label,
                    style=v.SvgStyle(
                        text_anchor="middle",
                        fill=text_color,
                        font_size=config["font_size"] * 0.8,
                        font_family=config["font_family"],
                    ),
                )
                gene_group.add(label)

            track_group.add(gene_group)

        canvas.add(track_group)

    return canvas.render()

SVG_CONFIG = {
    "canvas_width": 1200,
    "row_height": 40,
    "label_margin": 250,
    "right_margin": 50,
    "vert_margin": 10,
    "arrow_thickness_ratio": 0.5,
    "max_arrowhead_width": 10.0,
    "color_backbone": "#64748B",
    "color_anchor_fill": "#318686",
    "color_anchor_stroke": "#2A6B6B",
    "color_context_fill": "#ADD8E6",
    "color_context_stroke": "#8BB4C2",
    "font_family": "monospace",
    "font_size": 12,
}
