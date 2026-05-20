"""
Analyzes syntenic neighbourhood of a given gene across a set of genome.
"""
import pandas as pd
from typing import Any, Literal

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
    circular_wrap = lore.ValueInput(
        bool,
        description="If a genomic neighbourhood hits the end of a replicon, wrap around to continue from the start (useful for circular bacterial chromosomes/plasmids).",
        default=False,
        label="Circular wrap",
    )
    clamp_gap = lore.ValueInput(
        int | None,
        description="The maximum distance in base pairs to draw gaps. If the distance between two genes exceeds this value, it will be drawn as a broken axis of this length.",
        default=None,
        label="Clamp Gap Size (bp)",
    )
    clamp_gene = lore.ValueInput(
        int | None,
        description="The maximum distance in base pairs to draw genes. Draws shortened gene representations.",
        default=None,
        label="Clamp Gene Size (bp)",
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
    circular_wrap: bool,
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
    replicon_df = annot_df[annot_df["replicon"] == anchor_row["replicon"]].reset_index(drop=True)
    replicon_length = int(replicon_df["end"].max())

    # ...then reset index to replicon (using coordinates in case of duplicates)
    rel_anchor_idx = replicon_df[
        (replicon_df["begin"] == anchor_row["begin"]) &
        (replicon_df["end"] == anchor_row["end"])
    ].index[0]
    total_genes = len(replicon_df)

    # 3. Quick extract by index, wrapping if needed
    target_data = []
    for i, circular_pos in zip(range(start_offset, end_offset + 1), ctx_pos):
        idx = rel_anchor_idx + i
        if circular_wrap:
            target_data.append((idx % total_genes, circular_pos))
        else:
            if 0 <= idx < total_genes:
                target_data.append((idx, circular_pos))

    if not target_data:
        return pd.DataFrame()

    indices, valid_ctx_pos = zip(*target_data)
    neighbourhood_df = replicon_df.iloc[list(indices)].copy()
    neighbourhood_df["context_pos"] = list(valid_ctx_pos)

    # 4. If wrapping, detect and unwrap replicon if neighbourhood overshot the origin
    # Track where wraps occur for visualization
    neighbourhood_df["is_wrapped"] = False

    if circular_wrap:
        half_replicon = replicon_length / 2
        anchor_begin = anchor_row["begin"]

        underflow_mask = (neighbourhood_df["begin"] - anchor_begin) > half_replicon
        neighbourhood_df.loc[underflow_mask, "begin"] -= replicon_length
        neighbourhood_df.loc[underflow_mask, "end"] -= replicon_length
        neighbourhood_df.loc[underflow_mask, "is_wrapped"] = True

        overflow_mask = (anchor_begin - neighbourhood_df["begin"]) > half_replicon
        neighbourhood_df.loc[overflow_mask, "begin"] += replicon_length
        neighbourhood_df.loc[overflow_mask, "end"] += replicon_length
        neighbourhood_df.loc[overflow_mask, "is_wrapped"] = True

    return neighbourhood_df


def _neighbourhood_by_base_pairs(
    annot_df: pd.DataFrame,
    anchor_idx: int,
    context_window: tuple[int, int],
    circular_wrap: bool,
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

    # 2. Isolate replicon containing anchor gene
    replicon_id = anchor_row["replicon"]
    replicon_mask = annot_df["replicon"] == replicon_id

    # 3. Further limit mask to only genes that fall within the search window
    primary_mask = (annot_df["begin"] <= search_end) & (annot_df["end"] > search_start)

    if circular_wrap:
        replicon_length = int(annot_df.loc[replicon_mask]["end"].max())
        wrap_masks = []

        # Underflow: Left side of the window wrapped past 0 to the end of the replicon
        if search_start < 0:
            wrap_start = replicon_length + search_start
            wrap_masks.append(annot_df["end"] >= wrap_start)

        # Overflow: Right side of the window wrapped past the end of the replicon to 0
        if search_end > replicon_length:
            wrap_end = search_end - replicon_length
            wrap_masks.append(annot_df["begin"] <= wrap_end)

        if wrap_masks:
            combined_wrap_mask = wrap_masks[0]
            for m in wrap_masks[1:]:
                combined_wrap_mask = combined_wrap_mask | m
            final_mask = replicon_mask & (primary_mask | combined_wrap_mask)
        else:
            final_mask = replicon_mask & primary_mask
    else:
        final_mask = replicon_mask & primary_mask

    neighbourhood_df = annot_df[final_mask].copy()

    # 4. Unwrap: Use half-replicon length as a heuristic to detect if neighbourhood overshot
    # Track where wraps occur for visualization
    neighbourhood_df["is_wrapped"] = False

    if circular_wrap and not neighbourhood_df.empty:
        half_replicon = replicon_length / 2

        # Underflow: Physically at the end, but logically before the anchor
        underflow_mask = (neighbourhood_df["begin"] - anchor_start) > half_replicon
        neighbourhood_df.loc[underflow_mask, "begin"] -= replicon_length
        neighbourhood_df.loc[underflow_mask, "end"] -= replicon_length
        neighbourhood_df.loc[underflow_mask, "is_wrapped"] = True

        # Overflow: Physically at the start, but logically beyond the end
        overflow_mask = (anchor_start - neighbourhood_df["begin"]) > half_replicon
        neighbourhood_df.loc[overflow_mask, "begin"] += replicon_length
        neighbourhood_df.loc[overflow_mask, "end"] += replicon_length
        neighbourhood_df.loc[overflow_mask, "is_wrapped"] = True

    # 5. After unwrapping, re-sort and find current anchor gene index
    neighbourhood_df = neighbourhood_df.sort_values(by="begin").reset_index(drop=False)
    new_anchor_idx = neighbourhood_df[neighbourhood_df["index"] == anchor_idx].index[0]

    # 6. Strand-aware context position
    if anchor_orient == "minus":
        neighbourhood_df["context_pos"] = new_anchor_idx - neighbourhood_df.index
    else:
        neighbourhood_df["context_pos"] = neighbourhood_df.index - new_anchor_idx

    neighbourhood_df = neighbourhood_df.drop(columns=["index"])

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

    # 1. Establish replicon ID
    if "contig" in df.columns:
        df["replicon"] = df["chromosome"].fillna("") + "_" + df["contig"].fillna("")
        df["replicon"] = df["replicon"].str.strip("_")
    else:
        df["replicon"] = df["chromosome"].fillna("unknown_chromosome")
        ctx.logger.warning("No contig ID found. Fragmented assemblies may render incorrectly.")

    df = df.sort_values(by=["genome_accession", "replicon", "begin"]).reset_index(drop=True)
    df["is_n_terminus"] = False
    df["is_c_terminus"] = False
    n_idx = df.groupby("replicon")["begin"].idxmin()
    c_idx = df.groupby("replicon")["end"].idxmax()
    df.loc[n_idx, "is_n_terminus"] = True
    df.loc[c_idx, "is_c_terminus"] = True

    return df


def _extract_neighbourhoods(
    ctx: lore.ExecutionContext,
    annotation_df: pd.DataFrame,
    accessions: list[str],
    window: tuple[int, int],
    window_type: str,
    collapse_replicons: bool,
    circular_wrap: bool,
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
                    nb = _neighbourhood_by_feature(
                        annotation_df,
                        anchor_idx,
                        window,
                        circular_wrap,
                    )
                else:
                    nb = _neighbourhood_by_base_pairs(
                        annotation_df,
                        anchor_idx,
                        window,
                        circular_wrap,
                    )

                nb = _normalize_neighbourhood(nb)
                nb["track_id"] = f"{annotation_df.loc[anchor_idx, 'genome_accession']}_{anchor_idx}"
                neighbourhood_list.append(nb)

    else:
        # Option B: Group by replicon, zero on first occurence of any anchor
        all_anchor_rows = annotation_df[annotation_df["protein_accession"].isin(acc_set)]

        # 1. Group anchors found on the same replicon together
        for (genome, replicon), repl_group in all_anchor_rows.groupby(["genome_accession", "replicon"]):
            repl_nbs = []

            for anchor_idx in repl_group.index:
                if window_type == "gene_features":
                    nb = _neighbourhood_by_feature(
                        annotation_df,
                        anchor_idx,
                        window,
                        circular_wrap,
                    )
                else:
                    nb = _neighbourhood_by_base_pairs(
                        annotation_df,
                        anchor_idx,
                        window,
                        circular_wrap,
                    )

                # Tag anchor genes
                nb["is_anchor"] = nb["protein_accession"].isin(acc_set)
                repl_nbs.append(nb)

            # 2. Combine and discard overlapping background genes
            combined_nb = pd.concat(repl_nbs, ignore_index=True)
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
            combined_nb["track_id"] = f"{genome}_{replicon}"

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
    clamp_gene: int | None = None,
    collapse_replicons: bool = False,
    circular_wrap: bool = False,
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
        circular_wrap=circular_wrap,
    )

    if save_report:
        out_path = ctx.get_temp_path("genomic_neighbourhood_report.csv")
        neighbourhoods.to_csv(out_path, index=False)
        ctx.materialize_file(
            out_path,
            name="neighbourhood_report",
            output_key="neighbourhood_report",
        )

    svg_str = _render_neighbourhood_svg(neighbourhoods, clamp_gap, clamp_gene)
    ctx.materialize_content(
        svg_str,
        name="neighbourhood_view",
        output_key="neighbourhood_svg",
        data_type="svg",
        extension="svg",
    )
    ctx.logger.info("Genomic neighbourhood analysis complete.")


ORIGIN_PADDING_BP = 600

def _apply_virtual_layout(
    df: pd.DataFrame,
    clamp_gap: int | None,
    clamp_gene: int | None,
) -> tuple[pd.DataFrame, dict[str, list[float]], dict[str, list[float]], dict[str, Any]]:
    """
    Computes a virtual layout for all gene tracks in the DataFrame
    Returns:
    - modified df with 'render_begin' and 'render_end' series for visualization
    - dict of gap break x-coordinates (gaps/replicon wraps) keyed by track_id
    - dict of gene break x-coordinates (long genes) keyed by track_id
    - dict of wrap origin x-coordinates (for circular DNA) keyed by track_id
    """
    df = df.sort_values(by=["track_id", "begin"]).copy()

    # 1. Track breaks for visualization
    track_ids = df["track_id"].unique()
    gap_breaks = {t: [] for t in track_ids}
    gene_breaks = {t: [] for t in track_ids}
    wrap_breaks = {t: [] for t in track_ids}

    has_wraps = "is_wrapped" in df.columns and df["is_wrapped"].any()

    if not clamp_gap and not clamp_gene and not has_wraps:
        df["render_begin"] = df["begin"].astype(float)
        df["render_end"] = df["end"].astype(float)
        df["gene_clamped"] = False
        return df, gap_breaks, gene_breaks, wrap_breaks

    # Visual padding size for wraps at the origin (use clamp_gap if present, else default to 100)
    # TODO: Scale this based on track length?
    wrap_padding = ORIGIN_PADDING_BP
    if clamp_gap and wrap_padding > clamp_gap:
        wrap_padding = clamp_gap

    # 2. Gene clamping
    df["gene_length"] = df["end"] - df["begin"]
    df["gene_shrink"] = 0.0
    df["is_clamped_gene"] = False

    if clamp_gene:
        clamp_mask = df["gene_length"] > clamp_gene
        df.loc[clamp_mask, "gene_shrink"] = df["gene_length"] - clamp_gene
        df.loc[clamp_mask, "is_clamped_gene"] = True

    # 3. Gap clamping
    df["gap"] = df["begin"] - df.groupby("track_id")["end"].shift(1)
    df["gap_shrink"] = 0.0

    if clamp_gap:
        gap_mask = df["gap"] > clamp_gap
        df.loc[gap_mask, "gap_shrink"] = df["gap"] - clamp_gap

    # 4. Wrap padding: If the neighbourhood wraps at the origin, add a small gap
    if has_wraps:
        # Find the boundary where the wrap occurs
        df["prev_wrapped"] = (
            df.groupby("track_id")["is_wrapped"]
            .shift(1)
            .astype("boolean")  # pandas nullable boolean to avoid downcasting warnings
            .fillna(df["is_wrapped"])
        )
        df["wrap_boundary"] = df["is_wrapped"] != df["prev_wrapped"]

        # Use *negative* shrink to insert space (big brain moment)
        wrap_mask = df["wrap_boundary"] == True
        df.loc[wrap_mask, "gap_shrink"] = df.loc[wrap_mask, "gap"] - wrap_padding

    # 4. Track cumulative shift of coordinates due to clamping
    df["prev_gene_shrink"] = df.groupby("track_id")["gene_shrink"].shift(1).fillna(0.0)
    df["prev_total_shrink"] = df["prev_gene_shrink"] + df["gap_shrink"]

    df["cum_shrink_start"] = df.groupby("track_id")["prev_total_shrink"].cumsum().fillna(0.0)
    df["cum_shrink_end"] = df["cum_shrink_start"] + df["gene_shrink"]

    # 4. Apply virtual coordinates and re-zero anchors
    df["render_begin"] = df["begin"] - df["cum_shrink_start"]
    df["render_end"] = df["end"] - df["cum_shrink_end"]

    anchor_starts = df[df["context_pos"] == 0].groupby("track_id")["render_begin"].first()
    track_offsets = df["track_id"].map(anchor_starts).fillna(0)
    df["render_begin"] -= track_offsets
    df["render_end"] -= track_offsets

    # 5. Record break positions for viz
    if clamp_gap:
        wrap_col_exists = "is_wrapped" in df.columns
        if wrap_col_exists:
            gap_breaks_df = df[(df["gap_shrink"] > 0) & (~df["wrap_boundary"])]
        else:
            gap_breaks_df = df[df["gap_shrink"] > 0]

        for _, row in gap_breaks_df.iterrows():
            gap_breaks[row["track_id"]].append(row["render_begin"] - (clamp_gap / 2))

    if clamp_gene:
        gene_breaks_df = df[df["is_clamped_gene"] == True]
        for _, row in gene_breaks_df.iterrows():
            gene_breaks[row["track_id"]].append(row["render_begin"] + (clamp_gene / 2))

    if has_wraps:
        wrap_breaks_df = df[df["wrap_boundary"] == True]
        for _, row in wrap_breaks_df.iterrows():
            left_bp = row["render_begin"] - wrap_padding
            right_bp = row["render_begin"]
            wrap_breaks[row["track_id"]].append((left_bp, right_bp))

    df = df.drop(columns=["gap", "gene_length", "gene_shrink", "gap_shrink",
                          "prev_gene_shrink", "prev_total_shrink", "cum_shrink_start",
                          "cum_shrink_end", "prev_wrapped", "wrap_boundary"], errors="ignore")

    return df, gap_breaks, gene_breaks, wrap_breaks


def _render_neighbourhood_svg(
    df: pd.DataFrame,
    clamp_gap: int | None = None,
    clamp_gene: int | None = None,
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

    # 1. Delegate layout calculations to a helper function, applying clamping as needed
    df, gap_breaks, gene_breaks, wrap_breaks = _apply_virtual_layout(df, clamp_gap, clamp_gene)
    tracks_data = df.groupby("track_id")

    # 2. Global scale: Determine overall span
    global_min = df[["render_begin", "render_end"]].min().min()
    global_max = df[["render_begin", "render_end"]].max().max()
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
    for idx, (track_id, track_df) in enumerate(tracks_data):
        track_id = str(track_id)  # make the static type checker happy :)

        # Track-level metadata and simple positioning
        genome_acc = track_df["genome_accession"].iloc[0]  # in case of duplicates

        anchor_matches = track_df[track_df["context_pos"] == 0]
        anchor_acc = (
            anchor_matches["protein_accession"].iloc[0]
            if not anchor_matches.empty
            else "unknown_anchor"
        )

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
        track_min_x = _xscale(track_df[["render_begin", "render_end"]].min().min())
        track_max_x = _xscale(track_df[["render_begin", "render_end"]].max().max())

        # i. Simple line indicating span of the track
        track_group.add(v.SvgLine(
            x1=track_min_x, y1=y_center, x2=track_max_x, y2=y_center,
            style=v.SvgStyle(stroke=config["color_backbone"], stroke_width=2.0),
        ))

        # ii. Wrap around to origin: Hide backbone line
        for left_bp, right_bp in wrap_breaks.get(track_id, []):
            lx = _xscale(left_bp)
            rx = _xscale(right_bp)
            track_group.add(v.SvgLine(
                x1=lx, y1=y_center, x2=rx, y2=y_center,
                style=v.SvgStyle(stroke="#FFFFFF", stroke_width=3.0),
            ))

        # iii. Gap breaks (a '//' symbol across the track)
        for break_bp in gap_breaks.get(track_id, []):
            bx = _xscale(break_bp)
            # Draw broken axis indicator: a '//' symbol across the clamped gene
            # White space
            track_group.add(v.SvgPolygon(
                points=[(bx-5, y_center-row_height*0.6), (bx+1, y_center+row_height*0.6),
                        (bx+5, y_center+row_height*0.6), (bx-1, y_center-row_height*0.6)],
                style=v.SvgStyle(fill="#FFFFFF", stroke="none"),
            ))

            # With lines
            track_group.add(v.SvgLine(
                x1=bx-1, y1=y_center-row_height*0.6, x2=bx+5, y2=y_center+row_height*0.6,
                style=v.SvgStyle(stroke=config["color_backbone"], stroke_width=1.5),
            ))
            track_group.add(v.SvgLine(
                x1=bx-5, y1=y_center-row_height*0.6, x2=bx+1, y2=y_center+row_height*0.6,
                style=v.SvgStyle(stroke=config["color_backbone"], stroke_width=1.5),
            ))

        # iv. Indicate termini of contigs (a dot indicating the terminus of the DNA fragment)
        for _, gene in track_df.iterrows():
            if gene.get("is_n_terminus"):
                tx = _xscale(gene["render_begin"])
                track_group.add(v.SvgCircle(
                    cx=tx, cy=y_center, r=4,
                    style=v.SvgStyle(fill=config["color_backbone"], stroke="none"),
                ))
            if gene.get("is_c_terminus"):
                tx = _xscale(gene["render_end"])
                track_group.add(v.SvgCircle(
                    cx=tx, cy=y_center, r=4,
                    style=v.SvgStyle(fill=config["color_backbone"], stroke="none"),
                ))

        # C. Gene arrows
        for _, gene in track_df.iterrows():
            px_start = _xscale(gene["render_begin"])
            px_end = _xscale(gene["render_end"])

            acc = str(gene.get("protein_accession", ""))
            symbol = str(gene["symbol"]) if pd.notna(gene.get("symbol")) else ""
            name = str(gene["name"]) if pd.notna(gene.get("name")) else ""
            locus = str(gene["locus_tag"]) if pd.notna(gene.get("locus_tag")) else ""

            display_label = symbol or name or locus or acc or "unknown"
            gene_group = v.SvgGroup(classes=["gene-container"])

            # i. Hover tooltip (using original coordinates)
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

            # iii. Gene break indicator (if gene is clamped)
            if gene.get("is_clamped_gene") and clamp_gene:
                gx = _xscale(gene["render_begin"] + (clamp_gene / 2))

                # Draw broken axis indicator: a '//' symbol across the clamped gene
                # White space
                gene_group.add(v.SvgPolygon(
                    points=[(gx-5, y_center+arrow_h*0.6), (gx+1, y_center-arrow_h*0.6),
                            (gx+5, y_center-arrow_h*0.6), (gx-1, y_center+arrow_h*0.6)],
                    style=v.SvgStyle(fill="#FFFFFF", stroke="none"),
                ))
                # Lines
                gene_group.add(v.SvgLine(
                    x1=gx-1, y1=y_center+arrow_h*0.6, x2=gx+5, y2=y_center-arrow_h*0.6,
                    style=v.SvgStyle(stroke=stroke_color, stroke_width=1.0)
                ))
                gene_group.add(v.SvgLine(
                    x1=gx-5, y1=y_center+arrow_h*0.6, x2=gx+1, y2=y_center-arrow_h*0.6,
                    style=v.SvgStyle(stroke=stroke_color, stroke_width=1.0)
                ))

            # iv. Text label (if space allows)
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

            text_label = _trim_label(display_label, abs(int(px_end - px_start)))
            if text_label:
                # Use white text on the dark anchor background for readability
                text_color = config["color_anchor_text"] if is_anchor else config["color_context_text"]
                label = v.SvgText(
                    x=(px_start + px_end) / 2,
                    y=y_center + (config["font_size"] * 0.35), # True vertical centering for text
                    text=text_label,
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
    "color_anchor_text": "#FAFFFF",
    "color_context_text": "#333333",
    "font_family": "monospace",
    "font_size": 12,
}
