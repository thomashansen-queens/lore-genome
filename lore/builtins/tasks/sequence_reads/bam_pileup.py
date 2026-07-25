"""
Plugin to visualize where BAM reads were mapped to a reference genome.
"""
from collections.abc import Iterator
import lore
from lore import viz as v
import pysam


class BamPileupInputs:
    """Input data for the BamPileup task"""
    bam_file = lore.ArtifactInput(
        accepted_data=["bam"],
        select="single",
        load_as="path",
        description="BAM+BAI file containing aligned reads",
    )
    genome_fasta = lore.ArtifactInput(
        accepted_data=["genome_fasta", "nucleotide_fasta", "fna"],
        select="single",
        load_as="adapted",
        description="Reference genome FASTA file",
    )
    replicon = lore.ValueInput(
        str,
        label="Target Replicon/Chromosome",
        description="The exact accession/ID of the contig (e.g., NC_000913.3).",
    )
    start_bp = lore.ValueInput(
        int,
        label="Start Position (bp)",
        default=0,
    )
    end_bp = lore.ValueInput(
        int,
        label="End Position (bp)",
        default=5000,
    )
    overhang_reads = lore.ValueInput(
        bool,
        label="Show Overhanging Reads",
        default=False,
        description="Whether to truncate the display to the exact window, or to show how far reads extend beyond the window.",
    )
    max_depth = lore.ValueInput(
        int,
        label="Max depth",
        default=1000,
        min=-1,
        description="Maximum depth of coverage to display. Set to -1 or leave blank for no limit.",
    )
    max_reads = lore.ValueInput(
        int,
        label="Max reads",
        default=10000,
        description="Maximum number of reads to load from the BAM file. Helps avoid out-of-memory and/or browser crshes."
    )
    sort_strategy = lore.ValueInput(
        v.SortStrategy,
        label="Sort Strategy",
        default=v.SortStrategy.START,
        description="Strategy for sorting features in the pileup."
    )
    min_window_coverage = lore.ValueInput(
        float,
        label="Min. window coverage (%)",
        default=0,
        min=0, max=100,
        description="Minimum coverage (by percentage) required for a window to be displayed."
    )
    include_metadata = lore.ValueInput(
        bool,
        description="Whether to include metadata in the SVG hover tooltip. Disable to reduce SVG file size.",
        default=True,
        label="Tooltip data",
    )


class BamPileupOutputs:
    """Output data for the BamPileup task"""
    svg = lore.TaskOutput(
        label="BAM Pileup Visualization",
        data_type="svg",
        is_primary=True,
    )


@lore.task(
    "sequence_reads.bam_pileup",
    category="Sequence reads",
    inputs=BamPileupInputs,
    outputs=BamPileupOutputs,
    description="Generate a pileup plot of aligned reads to a reference genome",
    icon="☷",
    preview_mode="full",
)
def bam_pileup(
    ctx: lore.ExecutionContext,
    bam_file: str,
    genome_fasta: Iterator[dict],
    replicon: str,
    start_bp: int,
    end_bp: int,
    max_depth: int,
    max_reads: int,
    overhang_reads: bool,
    include_metadata: bool,
    min_window_coverage: float,
    sort_strategy: v.SortStrategy,
):
    """
    Generate a pileup plot of aligned reads to a reference genome.
    """
    if start_bp >= end_bp:
        ctx.logger.warning("Start position is greater than equal to end position. Reversing the order.")
        start_bp, end_bp = end_bp, start_bp

    window_length = end_bp - start_bp
    bounds = v.TrackBounds(start=start_bp, end=end_bp)
    stack = v.TrackStack(width=1200, track_spacing=10.0)

    # 1. Get the reference sequence
    ref_seq_slice = ""
    ref_start = start_bp

    for record in genome_fasta:
        if record.get("nucleotide_accession") == replicon:
            full_seq = record.get("nucleotide_sequence", "")
            # can't slice sequence that isn't there
            ref_start = max(0, start_bp)
            ref_end = min(len(full_seq), end_bp)
            ref_seq_slice = full_seq[ref_start:ref_end]
            break

    # 2. Add as a Track
    if ref_seq_slice:
        stack.add_track(v.SequenceTrack(
            sequence=ref_seq_slice,
            start=ref_start,
            name="Reference",
        ))
    else:
        ctx.logger.warning(
            f"No FASTA found from '{replicon}'. Check replicon name and adapter keys. "
            f"No reference sequence will be displayed."
        )

    reads = []
    depth_data = []
    avg_depth = 0.0
    mapq_data = []
    avg_mapq_overall = 0.0

    cov_start = max(0, int(bounds.start))
    cov_end = int(bounds.end)

    try:
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            # 3. Load reads within window
            for read in bam.fetch(replicon, int(bounds.start), int(bounds.end)):
                if len(reads) >= max_reads:
                    ctx.logger.warning(
                        f"Reached maximum read limit ({max_reads}). "
                        f"Some reads may not be displayed in the pileup."
                    )
                    break

                if not read.is_unmapped:
                    # 4. Calculate window coverage (lengthwise) and optionally skip reads that are too short
                    r_start = read.reference_start
                    r_end = read.reference_end or (r_start + 1)

                    overlap_start = max(start_bp, r_start)
                    overlap_end = min(end_bp, r_end)
                    overlap_length = max(0, overlap_end - overlap_start)
                    cov_pct = (overlap_length / window_length) * 100 if window_length > 0 else 0
                    if cov_pct < min_window_coverage:
                        continue

                    # 5. Build tooltip metadata
                    read_meta = {}
                    if include_metadata:
                        read_meta = {
                            "Read ID": read.query_name,
                            "MapQ": read.mapping_quality,
                            "Length": read.query_length,
                            "Strand": "Reverse" if read.is_reverse else "Forward",
                            "Coverage %": f"{cov_pct:.2f}",
                        }

                    reads.append(v.Feature(
                        start=read.reference_start,
                        end=read.reference_end or read.reference_start + 1,
                        shape=v.FeatureShape.ARROW_LEFT if read.is_reverse else v.FeatureShape.ARROW_RIGHT,
                        fill=_color_by_mapq(read.mapping_quality),
                        metadata=read_meta,
                    ))

            # 6. Calculate read depth (X-fold coverage) for the window
            if cov_end > cov_start:
                # count_coverage returns 4 arrays (A, C, G, T)
                cov_a, cov_c, cov_g, cov_t = bam.count_coverage(
                    replicon, cov_start, cov_end, quality_threshold=0
                )

                # Sum the bases at each position
                depths = [a + c + g + t for a, c, g, t in zip(cov_a, cov_c, cov_g, cov_t)]
                depth_data = [(cov_start + i, d) for i, d in enumerate(depths)]

                if depths:
                    avg_depth = sum(depths) / len(depths)

            # 7. Calculate MapQ (Phred-scaled mapping quality) for the window
            for column in bam.pileup(replicon, int(bounds.start), int(bounds.end), truncate=True):
                quals = column.get_mapping_qualities()
                if quals:
                    avg_q = sum(quals) / len(quals)
                    mapq_data.append((column.reference_pos, avg_q))

            if mapq_data:
                avg_mapq_overall = sum(d[1] for d in mapq_data) / len(mapq_data)

    except (ValueError, OSError) as e:
        ctx.logger.error(f"Error reading BAM file: {e}")
        raise RuntimeError(f"Failed to read BAM file: {e}")

    # === Rendering ===

    # 8. Add the pileup track
    pileup_meta = {}
    if include_metadata:
        pileup_meta = {
            "Replicon": replicon,
            "Start": start_bp,
            "End": end_bp,
            "Length": end_bp - start_bp,
            "Max Depth": max_depth,
        }

    stack.add_track(v.PileupTrack(
        name=f"Coverage ({len(reads)} reads)",
        features=reads,
        max_lanes=max_depth if max_depth > 0 else None,
        packing_gap=1.0,
        lane_padding_ratio=0.1,
        sort_strategy=sort_strategy,
        metadata=pileup_meta,
    ))

    # 9. Add read depth (X-fold coverage) track
    depth_meta = {}
    if include_metadata:
        depth_meta = {
            "Replicon": replicon,
            "Start": start_bp,
            "End": end_bp,
            "Length": end_bp - start_bp,
            "Average depth": avg_depth,
        }

    stack.add_track(v.PlotTrack(
        name="Depth",
        data=depth_data,
        type="line",
        axis=v.AxisConfig(
            y_ticks=3,
            y_tick_format=".0f",
            show_y_gridlines=True,
            y_min=0,
        ),
        metadata=depth_meta,
    ))

    # 10. Add MapQ plot track (Phred-scaled mapping quality)
    mapq_meta = {}
    if include_metadata:
        mapq_meta = {
            "Replicon": replicon,
            "Start": start_bp,
            "End": end_bp,
            "Length": end_bp - start_bp,
            "Average MapQ": f"{avg_mapq_overall:.1f}",
        }

    stack.add_track(v.PlotTrack(
        name="Average MapQ",
        data=mapq_data,
        type="area",
        axis=v.AxisConfig(
            y_min=0,
            y_max=60,
            y_ticks=[0, 30, 60],
            y_tick_format=".0f",
            show_y_gridlines=True,
        ),
        metadata=mapq_meta,
    ))

    # 11. Render and materialize
    if overhang_reads:
        overhang_start = min(r.start for r in reads)
        overhang_end = max(r.end for r in reads)
        bounds.start = min(bounds.start, overhang_start)
        bounds.end = max(bounds.end, overhang_end)

    svg_string = stack.render(bounds)
    ctx.materialize_content(
        svg_string,
        data_type="svg",
        name=f"Pileup_{replicon}_{start_bp}_{end_bp}",
        label="BAM Pileup Visualization",
        extension="svg",
    )


def _color_by_mapq(mapq: int) -> str | None:
    """
    Returns a CSS color based on Phred-scaled mapping quality.
    Returns None for high-quality reads so they fall back to the TrackTheme default.
    """
    if mapq <= 1:
        return "#ef4444"  # Red for multi-mapping/zero confidence
    elif mapq < 10:
        return "#f97316"  # Orange for very low confidence
    elif mapq < 20:
        return "#eab308"  # Yellow for moderate confidence
    return None  # Let the TrackTheme handle the good reads!
