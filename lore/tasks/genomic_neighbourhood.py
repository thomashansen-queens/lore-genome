"""
Analyzes syntenic neighbourhood of a given gene across a set of genome.
"""
import pandas as pd

from lore.core.adapters import TableAdapter, adapter_registry
from lore.core.executor import ExecutionContext
from lore.core.tasks import ArtifactInput, ValueInput, TaskOutput, task_registry, Cardinality, Materialization
from lore import viz as v


class GenomicNeighbourhoodTaskInputs:
    """Inputs for genomic neighbourhood analysis"""
    protein_accessions = ArtifactInput(
        description="The protein accession(s) for the gene(s) of interest",
        accepted_data=["protein_accession"],
        cardinality=Cardinality.ONE_OR_MORE,
        load_as=Materialization.CONTENT,
    )
    genome_annotations = ArtifactInput(
        label="Genome annotations",
        accepted_data=["ncbi_annotation_packages"],
        cardinality=Cardinality.ONE_OR_MORE,
        load_as=Materialization.PATH,
    )
    context_window_str = ValueInput(
        str | None,
        description="How far up/downstream of the gene of interest to include in the neighbourhood analysis.",
        default="",
        label="Context window size",
        examples=["3, 5", "5000 (if using base pair context)"],
    )
    context_window_type = ValueInput(
        str,
        options=["gene_features", "base_pairs"],
        widget="radio",
        description="Set window size in terms of number of gene features or number of base pairs.",
        default="gene_features",
        label="Context window type",
    )
    clamp_gap = ValueInput(
        int | None,
        description="The maximum distance in base pairs to draw to scale. If the distance between two genes exceeds this value, it will be drawn as a broken axis of this length.",
        default=None,
        label="Clamp Gap Size (bp)",
    )
    save_report = ValueInput(
        bool,
        description="Whether to write the genomic neighbourhood to a new report file.",
        default=True,
        label="Write report",
    )


class GenomicNeighbourhoodTaskOutputs:
    """Outputs for genomic neighbourhood analysis"""
    neighbourhood_svg = TaskOutput(
        data_type="svg",
        label="Genomic Neighbourhood Visualization",
        description="An SVG visualization of the genomic neighbourhood of the gene of interest across the input genomes",
        is_primary=True,
    )
    neighbourhood_report = TaskOutput(
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

    if window_up < 0 or window_down < 0:
        raise ValueError("Context window sizes must be non-negative integers.")
    return window_up, window_down


def _neighbourhood_by_feature(
    annot_df: pd.DataFrame,
    anchor: str,
    context_window: tuple[int, int],
) -> pd.DataFrame:
    """
    Extracts a neighbourhood of genes around a given anchor gene in a DataFrame 
    of annotations. Unwraps the replicon in case of wrap-around and adds a
    `context_pos` column to indicate position of gene feature relative to the
    anchor.
    """
    anchor_idx = annot_df[annot_df["protein_accession"] == anchor].index[0]
    replicon_df = annot_df[annot_df["chromosome"] == annot_df.loc[anchor_idx, "chromosome"]].reset_index(drop=True)
    replicon_length = replicon_df["end"].max()
    # then reset index to replicon
    anchor_idx = replicon_df[replicon_df["protein_accession"] == anchor].index[0]
    total_genes = len(replicon_df)

    # Quick extract by index, wrapping if needed
    target_indices = [(anchor_idx + i) % total_genes for i in range(-context_window[0], context_window[1] + 1)]
    neighbourhood_df = replicon_df.iloc[target_indices].copy()
    neighbourhood_df["context_pos"] = list(range(-context_window[0], context_window[1] + 1))

    # Detect and unwrap replicon if there is a wrap-around
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
    anchor: str,
    context_window: tuple[int, int],
) -> pd.DataFrame:
    """
    Extracts a neighbourhood of genes around a given anchor gene in a DataFrame 
    of annotations, using base pair distance.
    """
    anchor_row = annot_df[annot_df["protein_accession"] == anchor].iloc[0]
    anchor_start = anchor_row["begin"]
    anchor_end = anchor_row["end"]

    # Find all genes within the context window
    neighbourhood_df = annot_df[
        (annot_df["chromosome"] == anchor_row["chromosome"]) &
        (
            (annot_df["begin"] <= anchor_end + context_window[1]) &
            (annot_df["end"] >= anchor_start - context_window[0])
        )
    ].copy()

    # Add context position relative to the anchor gene
    anchor_idx = annot_df[annot_df["protein_accession"] == anchor].index[0]
    neighbourhood_df["context_pos"] = neighbourhood_df.index - anchor_idx

    return neighbourhood_df


def _normalize_neighbourhood(
    neighbourhood: pd.DataFrame,
    anchor: str,
) -> pd.DataFrame:
    """
    Normalizes the coordinates of genes in the neighbourhood relative to the anchor gene.
    Zero-centers the coordinates on the anchor gene and flips the coordinates for genes on the minus strand.
    """
    anchor_row = neighbourhood[neighbourhood["protein_accession"] == anchor].iloc[0]
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


@task_registry.register(
    "analysis.genomic_neighbourhood",
    name="Genomic neighbourhood",
    inputs=GenomicNeighbourhoodTaskInputs,
    outputs=GenomicNeighbourhoodTaskOutputs,
    category="analysis",
    icon="☷",
)
def genomic_neighbourhood_analysis(
    ctx: ExecutionContext,
    protein_accessions: list[str],
    genome_annotations: list[str],
    context_window_str: str,
    context_window_type: str,
    save_report: bool,
    clamp_gap: int | None = None,
):
    """
    Analyze the genomic neighbourhood of gene(s) of interest across a set of genomes.
    Allows multiple annotation files mostly to help visualize the same gene in 
    different strains/species
    """
    # Defaults
    if not context_window_str:
        context_window_str = "5" if context_window_type == "gene_features" else "5000"

    adapter = adapter_registry["NcbiGenomeAnnotationsAdapter"]
    if not isinstance(adapter, TableAdapter):
        raise TypeError("Expected NcbiGenomeAnnotationsAdapter to be a TableAdapter")

    context_window = _set_window(context_window_str.strip())
    df_list = [adapter.to_dataframe(path) for path in genome_annotations]
    annot_df = pd.concat(df_list, ignore_index=True)
    # just in case
    annot_df[["begin", "end", "protein_length"]] = (
        annot_df[["begin", "end", "protein_length"]].astype("Int64")
    )
    annot_df = annot_df.sort_values(by=["genome_accession", "begin"]).reset_index(drop=True)

    neighbourhoods = []
    for acc in protein_accessions:
        neighbourhood = None
        if context_window_type == "gene_features":
            neighbourhood = _neighbourhood_by_feature(annot_df, acc, context_window)
        else:  # "base_pairs"
            neighbourhood = _neighbourhood_by_base_pairs(annot_df, acc, context_window)

        neighbourhood = _normalize_neighbourhood(neighbourhood, acc)
        neighbourhoods.append(neighbourhood)

    neighbourhoods = pd.concat(neighbourhoods, ignore_index=True)
    if save_report:
        out_path = ctx.get_temp_path("genomic_neighbourhood_report.csv")
        neighbourhoods.to_csv(out_path, index=False)
        ctx.materialize_file(
            out_path,
            name="neighbourhood_report",
            output_key="neighbourhood_report",
            data_type="genome_annotations",
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
    canvas_width: int = 1200,
) -> str:
    """
    Renders the genomic neighbourhood as an SVG string. If `clamp_gap` is provided,
    distances between genes that exceed this value will be visually clamped to 
    this maximum distance.
    """
    if df.empty:
        return v.SvgCanvas(width=canvas_width, height=200).render()

    # 1. Local scale: Clamp sizes if set
    if clamp_gap is not None:
        # FUTURE: Implement layout_track_clamped logic here
        # Adjusts `begin` and `end` coordinates in the DataFrame
        pass

    # 2. Global scale: Determine overall span
    global_min = df[["begin", "end"]].min().min()
    global_max = df[["begin", "end"]].max().max()
    if global_min == global_max:
        global_max = global_min + 1.0  # Divide-by-zero guard

    label_margin = 250
    right_margin = 50
    vert_margin = 10
    plot_width = canvas_width - label_margin - right_margin

    def _xscale(bp: float) -> float:
        """Translates a base-pair coordinate to a pixel X-coordinate"""
        percent_of_span = (bp - global_min) / (global_max - global_min)
        return label_margin + percent_of_span * plot_width

    # 3. Canvas setup
    row_height = 40
    # CHECK THIS
    tracks_data = df.groupby("genome_accession")

    canvas_height = (len(tracks_data) * row_height) + vert_margin * 2
    canvas = v.SvgCanvas(width=canvas_width, height=canvas_height)

    #4. Draw tracks
    for idx, (genome_acc, track_df) in enumerate(tracks_data):
        anchor_acc = track_df['protein_accession'][track_df['context_pos'] == 0].iloc[0]
        y_top = vert_margin + idx * row_height
        y_center = y_top + (row_height / 2)

        track_group = v.SvgGroup(classes=[f"track-{genome_acc}"])

        # A. Track label
        label_text = f"{genome_acc} | {anchor_acc}"  # just grab the first one for now
        track_group.add(v.SvgText(
            x=label_margin - 15,
            y=y_center + 4,  # eyeball centering
            text=label_text,
            style=v.SvgStyle(text_anchor="end", font_size=12, font_family="monospace"),
        ))

        # B. Backbone line
        track_min_x = _xscale(track_df[["begin", "end"]].min().min())
        track_max_x = _xscale(track_df[["begin", "end"]].max().max())
        track_group.add(v.SvgLine(
            x1=track_min_x, y1=y_center, x2=track_max_x, y2=y_center,
            style=v.SvgStyle(stroke="#64748B", stroke_width=2.0),
        ))

        # C. Gene arrows
        for _, gene in track_df.iterrows():
            px_start = _xscale(gene["begin"])
            px_end = _xscale(gene["end"])

            # Geometry setup
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

            # Highlight anchor gene
            is_anchor = (gene["context_pos"] == 0)
            fill_color = "#318686" if is_anchor else "#ADD8E6"
            stroke_color = "#2A6B6B" if is_anchor else "#8BB4C2"

            arrow = v.SvgPolygon(
                points=pts,
                classes=["gene-arrow", "anchor-gene" if is_anchor else "context-gene"],
                data={
                    "accession": gene["protein_accession"],
                    "symbol": gene.get("symbol", ""),
                    "locus-tag": gene.get("locus_tag", ""),
                    "start": gene["begin"],
                    "end": gene["end"],
                },
                style=v.SvgStyle(fill=fill_color, stroke=stroke_color, stroke_width=1.0),
            )
            track_group.add(arrow)

        canvas.add(track_group)

    return canvas.render()
