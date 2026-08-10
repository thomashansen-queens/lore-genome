"""
Analyzes syntenic neighbourhood of a given gene across a set of genome.
"""
from collections.abc import Iterator
import pandas as pd
from typing import Literal

import lore
from lore import viz as v


class GenomicNeighbourhoodTaskInputs:
    """Inputs for genomic neighbourhood analysis"""
    protein_accession = lore.ArtifactInput(
        label="Protein accessions",
        description="The protein accession(s) for the gene(s) of interest",
        accepted_data=["protein_accession"],
        select="multiple",
        load_as="adapted",
    )
    genome_annotation = lore.ArtifactInput(
        label="Genome annotations",
        accepted_data=["ncbi_annotation_packages", "genome_annotations"],
        select="multiple",
        load_as="adapted_stream",
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
    include_metadata = lore.ValueInput(
        bool,
        description="Whether to include metadata in the SVG hover tooltip. Disable to reduce SVG file size.",
        default=True,
        label="Tooltip data",
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


@lore.task(
    "analysis.genomic_neighbourhood",
    name="Genomic neighbourhood",
    inputs=GenomicNeighbourhoodTaskInputs,
    outputs=GenomicNeighbourhoodTaskOutputs,
    category="clustering",
    icon="☷",
    preview_mode="live_full",
)
def genomic_neighbourhood_analysis(
    ctx: lore.ExecutionContext,
    protein_accession: list[str],
    genome_annotation: Iterator[dict],
    context_window_str: str | None = None,
    context_window_type: Literal["gene_features", "base_pairs"] = "gene_features",
    save_report: bool = False,
    clamp_gap: int | None = None,
    clamp_gene: int | None = None,
    collapse_replicons: bool = False,
    circular_wrap: bool = False,
    include_metadata: bool = True,
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

    cache_key = "_".join(a.id for a in ctx.input_artifacts.get("genome_annotation", []))
    annotation_df = _build_master_df(ctx, annotations=genome_annotation, cache_key=cache_key)

    neighbourhoods = _extract_neighbourhoods(
        ctx=ctx,
        annotation_df=annotation_df,
        accessions=protein_accession,
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

    svg_str = _render_neighbourhood_svg(
        neighbourhoods, clamp_gap, clamp_gene, include_metadata,
        window=context_window, window_type=context_window_type,
    )
    ctx.materialize_content(
        svg_str,
        name="neighbourhood_view",
        output_key="neighbourhood_svg",
        data_type="svg",
        extension="svg",
    )
    ctx.logger.info("Genomic neighbourhood analysis complete.")

############################################################
# COMPUTATION HELPERS
############################################################

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
def _build_master_df(ctx: lore.ExecutionContext, annotations: Iterator[dict], cache_key: str):
    """
    Builds a master DataFrame from the input annotations. Memoized based on the 
    IDs of input annotation artifacts to avoid repeated processing.
    """
    df = pd.DataFrame(annotations)
    try:
        df[["begin", "end", "protein_length"]] = df[["begin", "end", "protein_length"]].astype("Int64")
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError("Could not find 'begin', 'end', or 'protein_length' column in 'Genome annotations'. Please ensure you've provided the correct Genome annotations file.") from e

    # Preserve untouched genomic coordinates: downstream alignment mutates begin/end
    # in place (zero-centering + strand flips), but the tooltip must show the real ones.
    df["begin_original"] = df["begin"]
    df["end_original"] = df["end"]

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

############################################################
# VISUALIZATION
############################################################
CANVAS_WIDTH = 1200

def _render_neighbourhood_svg(
    df: pd.DataFrame,
    clamp_gap: int | None = None,
    clamp_gene: int | None = None,
    include_metadata: bool = True,
    window: tuple[int, int] | None = None,
    window_type: str = "gene_features",
) -> str:
    """
    Renders the genomic neighbourhood as an SVG string. If `clamp_gap` is provided,
    distances between genes that exceed this value will be visually clamped to 
    this maximum distance.
    NOTE: In case of multiple anchor genes on the same replicon, if collapse_replicon, the extent is the minimum of both to the maximum of both... in the future, could make separate extents
    """
    if df.empty:
        return v.SvgCanvas(width=CANVAS_WIDTH, height=200).render()

    # 1. Slide all coordinates to anchor genes sit at 0
    anchor_starts = df[df["context_pos"] == 0].groupby("track_id")["begin"].first()
    offsets = df["track_id"].map(anchor_starts).fillna(0)

    df["shifted_begin"] = df["begin"] - offsets
    df["shifted_end"] = df["end"] - offsets

    df = df.sort_values(by=["track_id", "shifted_begin"])

    # 2. Build track names; the stack sizes its own label gutter ("auto") and
    #    infers the shared domain from each track's feature extents.
    tracks_by_id = []
    for track_id, track_df in df.groupby("track_id"):
        genome_acc = track_df["genome_accession"].iloc[0]
        anchor_matches = track_df[track_df["context_pos"] == 0]
        anchor_acc = anchor_matches["protein_accession"].iloc[0] if not anchor_matches.empty else "unknown"
        track_name = f"{genome_acc} | {anchor_acc}"
        tracks_by_id.append((track_id, track_df, track_name))

    stack = v.TrackStack(
        width=CANVAS_WIDTH,
        track_height=40.0,
        track_spacing=2.0,
    )

    # 4. Add each track to the stack
    for track_id, track_df, track_name in tracks_by_id:

        # 5. Map DataFrame to Feature objects
        features = []
        breaks = []

        prev_end = None
        prev_wrapped = None

        for _, row in track_df.iterrows():
            curr_start = row["shifted_begin"]
            curr_end = row["shifted_end"]
            is_wrapped = row.get("is_wrapped", False)

            # 6. Evaluate breaks
            if prev_end is not None:
                gap = curr_start - prev_end
                is_wrap_boundary = (is_wrapped != prev_wrapped)

                if is_wrap_boundary:
                    breaks.append(v.Break(start=prev_end, end=curr_start, kind=v.BreakKind.GAP, width="30px"))
                elif clamp_gap and gap > clamp_gap:
                    breaks.append(v.Break(
                        start=prev_end + (clamp_gap / 2),
                        end=curr_start - (clamp_gap / 2),
                        kind=v.BreakKind.CUT,
                        width=0,
                    ))

            gene_len = curr_end - curr_start
            if clamp_gene and gene_len > clamp_gene:
                breaks.append(v.Break(
                    start=curr_start + (clamp_gene / 2),
                    end=curr_end - (clamp_gene / 2),
                    kind=v.BreakKind.SPLICE,
                    width=0,
                ))

            if row.get("is_n_terminus"):
                breaks.append(v.Break(start=curr_start, end=curr_start, kind=v.BreakKind.GAP, width=0))
            if row.get("is_c_terminus"):
                breaks.append(v.Break(start=curr_end, end=curr_end, kind=v.BreakKind.GAP, width=0))

            prev_end = curr_end
            prev_wrapped = is_wrapped

            # 7. Feature geometry
            acc = str(row.get("protein_accession", ""))
            sym = str(row.get("symbol", "")) if pd.notna(row.get("symbol")) else ""
            locus = str(row.get("locus_tag", "")) if pd.notna(row.get("locus_tag")) else ""

            disp = sym or row.get("name", "")

            if row["orientation"] == "plus":
                shape = v.FeatureShape.ARROW_RIGHT
            elif row["orientation"] == "minus":
                shape = v.FeatureShape.ARROW_LEFT
            else:
                shape = v.FeatureShape.BOX

            # 8. Package data for tooltip hover text
            feat_meta = {}
            if include_metadata:
                begin = row.get("begin_original")
                end = row.get("end_original")
                has_coords = pd.notna(begin) and pd.notna(end)
                length = int(end) - int(begin) if has_coords else None
                feat_meta = {
                    "Accession": acc,
                    "Name": row.get("name", "N/A"),
                    "Symbol": sym if sym else "N/A",
                    "Locus": locus if locus else "N/A",
                    "Replicon": row.get("replicon", "N/A"),
                    "Begin": int(begin) if has_coords else "N/A",
                    "End": int(end) if has_coords else "N/A",
                    "Length": f"{length:,}" if length is not None else "N/A",
                    "Orientation": row.get("orientation", "N/A"),
                }

            # 9. Add feature to the Track
            features.append(v.Feature(
                start=curr_start,
                end=curr_end,
                shape=shape,
                label=disp,
                highlight=(row["context_pos"] == 0),
                metadata=feat_meta,
            ))

        # 10. In base-pairs mode, set backbone to window extent
        track_extent = None
        if window_type == "base_pairs" and window is not None:
            anchors = track_df[track_df["context_pos"] == 0]
            if not anchors.empty:
                win_up, win_down = window
                lo = anchors["shifted_begin"].min() - win_up
                hi = anchors["shifted_end"].max() + win_down
                track_extent = (float(lo), float(hi))
        else:
            track_extent = (
                float(track_df["shifted_begin"].min()),
                float(track_df["shifted_end"].max())
            )

        # 11. Build Track-level metadata and add to Stack
        if include_metadata:
            if not track_df[track_df["context_pos"] == 0].empty:
                # In case of collapsed replicons, multiple anchor genes may coexist
                prot_acc = track_df[track_df["context_pos"] == 0]["protein_accession"].to_list()
                prot_acc = prot_acc[0] if len(prot_acc) == 1 else ", ".join(prot_acc)
            else:
                prot_acc = "unknown"

            track_meta = {
                "genome_accession": track_df["genome_accession"].iloc[0],
                "protein_accession": prot_acc,
                "replicon": track_df["replicon"].iloc[0],
                "extent": abs(track_extent[1] - track_extent[0]) if track_extent else None,
            }

        stack.add_track(v.FeatureTrack(
            name=track_name,
            features=features,
            breaks=breaks,
            extent=track_extent,
            metadata=track_meta,
        ))

    return stack.render()
