"""
Plugin to visualize where BAM reads were mapped to a reference genome.
"""
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
        load_as="path",
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
    max_reads = lore.ValueInput(
        int,
        label="Max Reads to Draw",
        default=1000,
        description="Safeguard to prevent SVG DOM crashes in extremely deep coverage regions.",
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
    genome_fasta: str,
    replicon: str,
    start_bp: int,
    end_bp: int,
    max_reads: int,
):
    """
    Generate a pileup plot of aligned reads to a reference genome.
    """
    if start_bp >= end_bp:
        ctx.logger.warning("Start position is greater than equal to end position. Reversing the order.")
        start_bp, end_bp = end_bp, start_bp

    bounds = v.TrackBounds(start=start_bp, end=end_bp)
    stack = v.TrackStack(width=1200)

    # 1. Add Reference Track
    stack.add_track(ReferenceSequenceTrack(
        fasta_path=genome_fasta, 
        replicon=replicon,
    ))

    # 2. Add BAM Pileup Track
    stack.add_track(BamAlignmentTrack(
        bam_path=bam_file, 
        replicon=replicon, 
        max_reads=max_reads
    ))

    # 3. Render and materialize
    svg_string = stack.render(bounds)
    ctx.materialize_content(
        svg_string,
        data_type="svg",
        name=f"Pileup_{replicon}_{start_bp}_{end_bp}",
        label="BAM Pileup Visualization",
        extension="svg",
    )


############################################################
# TRACKS FOR VISUALIZATION
############################################################

class ReferenceSequenceTrack(v.BaseTrack):
    """Draws the reference genome backbone (ACTG) if zoomed in enough"""
    def __init__(self, fasta_path: str, replicon: str):
        super().__init__(name="Reference", height=20, label_pos=v.LabelPosition.LEFT)
        self.fasta_path = fasta_path
        self.replicon = replicon

    def render_payload(self, bounds: v.TrackBounds, width: float, theme: v.TrackTheme) -> v.SvgGroup:
        group = v.SvgGroup()
        y_center = self.height / 2

        # Draw the backbone
        group.add(v.SvgLine(
            x1=0, y1=y_center, x2=width, y2=y_center,
            style=v.SvgStyle(stroke=theme.color_backbone, stroke_width=2.0),
        ))

        # If zoomed in closer than 150bp, draw the actual letters
        if bounds.length <= 150:
            with pysam.FastaFile(self.fasta_path) as fasta:
                # pysam is 0-indexed, half-open
                seq = fasta.fetch(self.replicon, max(0, int(bounds.start)), int(bounds.end))

            bp_width = width / bounds.length

            for i, nucleotide in enumerate(seq):
                x_pos = (i * bp_width) + (bp_width / 2)
                group.add(v.SvgText(
                    x=x_pos, 
                    y=y_center + (theme.font_size * 0.35),
                    text=nucleotide,
                    style=v.SvgStyle(
                        text_anchor="middle", 
                        font_size=theme.font_size, 
                        font_family=theme.font_family,
                        fill="#111111"
                    ),
                ))
        return group


class BamAlignmentTrack(v.BaseTrack):
    """Greedy read-packing pileup for BAM alignments."""
    def __init__(self, bam_path: str, replicon: str, max_reads: int):
        super().__init__("Aligned Reads", height=300.0, label_pos=v.LabelPosition.LEFT)
        self.bam_path = bam_path
        self.replicon = replicon
        self.max_reads = max_reads

    def render_payload(self, bounds: v.TrackBounds, width: float, theme: v.TrackTheme) -> v.SvgGroup:
        group = v.SvgGroup()

        def _xscale(bp: float) -> float:
            return ((bp - bounds.start) / bounds.length) * width

        # 1. Safely fetch reads
        reads = []
        try:
            with pysam.AlignmentFile(self.bam_path, "rb") as bam:
                for read in bam.fetch(self.replicon, int(bounds.start), int(bounds.end)):
                    if len(reads) >= self.max_reads:
                        break
                    if not read.is_unmapped:
                        reads.append({
                            "start": read.reference_start,
                            "end": read.reference_end,
                            "name": read.query_name,
                            "is_forward": not read.is_reverse
                        })
        except ValueError:
            group.add(v.SvgText(x=width/2, y=self.height/2, text="Invalid Replicon or Missing .bai Index"))
            return group

        if not reads:
            group.add(v.SvgText(x=width/2, y=self.height/2, text="No reads mapped in this window."))
            return group

        # 2. Greedy Packing Algorithm (Pileup)
        reads.sort(key=lambda r: r["start"])
        levels = [] # Tracks the furthest right-hand coordinate for each row

        padding_bp = bounds.length * 0.01 # 1% visual margin between reads

        for read in reads:
            placed = False
            for i, level_end in enumerate(levels):
                if read["start"] > level_end + padding_bp:
                    levels[i] = read["end"]
                    read["row"] = i
                    placed = True
                    break

            if not placed:
                levels.append(read["end"])
                read["row"] = len(levels) - 1

        # 3. Dynamic Height Scaling
        total_rows = len(levels)
        # Cap row height at 12px, but shrink it if coverage is insanely deep
        actual_row_h = min(12.0, self.height / total_rows)

        # 4. Draw Read Rectangles
        for read in reads:
            x1 = _xscale(max(bounds.start, read["start"]))
            x2 = _xscale(min(bounds.end, read["end"]))
            y = read["row"] * actual_row_h

            color = theme.color_primary_fill if read["is_forward"] else theme.color_secondary_fill

            rect = v.SvgRect(
                x=x1, y=y, width=max(1.0, x2 - x1), height=actual_row_h * 0.8,
                style=v.SvgStyle(fill=color, stroke="none")
            )

            rect_group = v.SvgGroup()
            rect_group.add(v.SvgTitle(text=f"{read['name']}\n{read['start']:,} - {read['end']:,}"))
            rect_group.add(rect)

            group.add(rect_group)

        return group
