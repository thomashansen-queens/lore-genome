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
    "alignment.bam_pileup",
    inputs=BamPileupInputs,
    outputs=BamPileupOutputs,
    description="Generate a pileup plot of aligned reads to a reference genome",
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
    stack = v.TrackStack(width=1200)

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

    # 3. Load reads within window
    reads = []
    try:
        with pysam.AlignmentFile(bam_file, "rb") as bam:
            for read in bam.fetch(replicon, int(bounds.start), int(bounds.end)):
                if len(reads) >= max_reads:
                    ctx.logger.warning(
                        f"Reached maximum read limit ({max_reads}). "
                        f"Some reads may not be displayed in the pileup."
                    )
                    break

                if not read.is_unmapped:
                    # 4. Calculate window coverage and optionally skip reads that are too short
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
                        metadata=read_meta,
                    ))

    except (ValueError, OSError) as e:
        ctx.logger.error(f"Error reading BAM file: {e}")
        raise RuntimeError(f"Failed to read BAM file: {e}")

    # 6. Add the pileup track
    track_meta = {}
    if include_metadata:
        track_meta = {
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
        metadata=track_meta,
    ))

    # 7. Render and materialize
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
