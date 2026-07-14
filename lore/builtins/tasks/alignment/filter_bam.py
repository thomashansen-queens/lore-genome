"""
Task to extract a subset of a BAM file by genomic region and/or read properties.

Uses pysam's indexed `fetch(replicon, start, stop)` to seek directly to a region
via the .bai index — O(region) rather than a full scan — and writes the matching
reads losslessly (full AlignedSegment, header preserved) to a new, re-indexed BAM
bundle. Read-property constraints (mapping quality, mapped/primary/paired, etc.)
are applied on top, mirroring `samtools view -F/-q` flag filtering.
"""
from enum import StrEnum
from pathlib import Path

import lore
import pysam


class ReadFilter(StrEnum):
    """Opt-in per-read constraints. A read is kept only if every selected one holds."""
    MAPPED_ONLY = "mapped_only"                # drop unmapped reads
    PRIMARY_ONLY = "primary_only"              # drop secondary + supplementary alignments
    PROPER_PAIRS_ONLY = "proper_pairs_only"    # keep only properly paired reads
    EXCLUDE_DUPLICATES = "exclude_duplicates"  # drop PCR/optical duplicates
    EXCLUDE_QCFAIL = "exclude_qcfail"          # drop reads failing QC


# Each filter maps to a predicate that returns True when the read should be KEPT.
_FILTER_PREDICATES = {
    ReadFilter.MAPPED_ONLY: lambda r: not r.is_unmapped,
    ReadFilter.PRIMARY_ONLY: lambda r: not (r.is_secondary or r.is_supplementary),
    ReadFilter.PROPER_PAIRS_ONLY: lambda r: r.is_proper_pair,
    ReadFilter.EXCLUDE_DUPLICATES: lambda r: not r.is_duplicate,
    ReadFilter.EXCLUDE_QCFAIL: lambda r: not r.is_qcfail,
}


class FilterBamInputs:
    """Inputs for the Filter BAM Task"""
    bam_file = lore.ArtifactInput(
        accepted_data=["bam"],
        select="single",
        load_as="path",
        label="Input BAM file",
        description="An indexed BAM+BAI file of aligned reads to subset.",
    )
    replicon = lore.ValueInput(
        str | None,
        default=None,
        label="Target replicon/chromosome",
        description="Contig to isolate (e.g. NC_000913.3). Leave blank to include all replicons.",
    )
    start_bp = lore.ValueInput(
        int,
        default=0,
        min=0,
        label="Start Position (bp)",
        description="Left edge of the region.",
    )
    end_bp = lore.ValueInput(
        int | None,
        default=None,
        label="End Position (bp)",
        description="Right edge of the region. Blank extends to the end of the replicon.",
    )
    min_mapping_quality = lore.ValueInput(
        int,
        default=0,
        min=0,
        label="Minimum mapping quality (MAPQ)",
        description="Drop reads below this MAPQ. 0 keeps everything.",
    )
    read_filters = lore.ValueInput(
        list[ReadFilter] | None,
        default=None,
        label="Read property filters",
        description="Optional additional filters to apply.",
    )


class FilterBamOutputs:
    """Outputs for the Filter BAM Task"""
    bam = lore.TaskOutput(
        data_type="bam",
        label="Filtered BAM file",
        description="A new BAM (with .bai index) containing only the reads that passed the filters.",
        is_primary=True,
        yields="single",
    )


def _build_read_filter(min_mapping_quality, read_filters):
    """Compose the selected constraints into a single keep(read) -> bool predicate."""
    selected = [ReadFilter(f) for f in (read_filters or [])]
    predicates = [_FILTER_PREDICATES[f] for f in selected]

    def keep(read) -> bool:
        if read.mapping_quality < min_mapping_quality:
            return False
        return all(pred(read) for pred in predicates)

    return keep


@lore.task(
    "alignment.filter_bam",
    inputs=FilterBamInputs,
    outputs=FilterBamOutputs,
    name="Filter BAM",
    category="Alignment",
    icon="✂",
    preview_mode="dry_run",
)
def filter_bam_handler(
    ctx: lore.ExecutionContext,
    bam_file: str,
    replicon: str | None = None,
    start_bp: int = 0,
    end_bp: int | None = None,
    min_mapping_quality: int = 0,
    read_filters: list[ReadFilter] | None = None,
):
    """
    Extract a subset of a BAM file by genomic region and/or read properties into a
    new, indexed BAM. Give a replicon (optionally with start/end) to isolate a
    region via a fast indexed seek. Omit the replicon to scan all replicons; any
    start/end you provide is still honored, applied as a coordinate window across
    every replicon. Read-property filters (min MAPQ, mapped-only, primary-only, ...)
    are applied on top.
    """
    bam_path = Path(bam_file)
    keep = _build_read_filter(min_mapping_quality, read_filters)

    # Output pathing
    base_name = bam_path.name
    while base_name.endswith((".bam", ".bai", ".gz")):
        base_name = Path(base_name).stem

    with pysam.AlignmentFile(str(bam_path), "rb") as bam_in:
        # 1. Process header
        total_len = sum(bam_in.lengths)
        total_reads_msg = ""
        if bam_in.has_index():
            total_reads = bam_in.mapped + bam_in.unmapped
            total_reads_msg = f", {total_reads:,} total indexed reads"

        ctx.logger.info(
            f"Input BAM header: {bam_in.nreferences} replicons, "
            f"{total_len:,} bp total length{total_reads_msg}"
        )

        # 2. Echo filter parameters
        filter_names = [f.value for f in (read_filters or [])]
        ctx.logger.info(
            f"Active filters: replicon={replicon or 'all'}, "
            f"start={start_bp}, end={end_bp}, "
            f"min_MAPQ={min_mapping_quality}, "
            f"read_filters={', '.join(filter_names) if filter_names else 'none'}"
        )

        # 3. Check start/end
        if not replicon and (start_bp > 0 or end_bp is not None):
            if bam_in.references == 1:
                replicon = bam_in.references[0]
                ctx.logger.info("Auto-selected the only replicon: %s", replicon)
            elif bam_in.nreferences > 1:
                longest_idx = max(range(bam_in.nreferences), key=lambda i: bam_in.lengths[i])
                replicon = bam_in.references[longest_idx]
                ctx.logger.info(
                    f"No replicon specified. Auto-selected the largest replicon: {replicon} "
                    f"({bam_in.lengths[longest_idx]:,} bp)"
                )

        # 4. Check start/end sanity
        start = max(0, start_bp)
        end = end_bp
        if end is not None and end < start:
            ctx.logger.warning(f"End ({end}) is before start ({start}); swapping values.")
            start, end = end, start

        # 5. Build the optimized read generator
        def _get_read_source():
            if replicon:
                if not bam_in.has_index():
                    raise RuntimeError(
                        f"Cannot subset by region: '{bam_path}' has no index (.bai). "
                        f"Please index it first (e.g. `samtools index {bam_path}`) and try again."
                    )
                if replicon not in bam_in.references:
                    available = ", ".join(bam_in.references[:10])
                    suffix = "..." if len(bam_in.references) > 10 else ""
                    raise ValueError(
                        f"Replicon '{replicon}' not found in BAM. Available: {available}{suffix}"
                    )

                yield from bam_in.fetch(replicon, start, end)
            else:
                # True full-file scan (no coords, just flags)
                ctx.logger.info("Scanning entire BAM file for read properties.")
                yield from bam_in.fetch(until_eof=True)

        region_tag = f"{replicon or 'all'}_{start}_{end if end is not None else 'end'}"
        bam_out_path = ctx.get_temp_path(f"{base_name}_{region_tag}.bam")

        # 6. Stream through the filter into a new BAM
        scanned = 0
        kept = 0
        with pysam.AlignmentFile(str(bam_out_path), "wb", template=bam_in) as bam_out:
            for read in _get_read_source():
                scanned += 1
                if keep(read):
                    bam_out.write(read)
                    kept += 1

    if scanned == 0:
        ctx.logger.warning("No reads found in the specified coordinate region.")
    else:
        pass_rate = (kept / scanned) * 100
        ctx.logger.info(
            f"Spatial fetch yielded {scanned:,} reads. "
            f"Property filters kept {kept:,} ({pass_rate:.1f}%)."
        )

    if kept == 0:
        ctx.logger.warning("Output BAM is completely empty (contains only a header).")

    # 7. Index the new BAM
    try:
        pysam.index(str(bam_out_path))
    except pysam.SamtoolsError as e:
        ctx.logger.error("Failed to index the filtered BAM. Details: %s", str(e))
        raise RuntimeError(
            "Indexing failed. Input BAM must be coordinate-sorted before filtering."
        )

    bai_out_path = Path(f"{bam_out_path}.bai")
    if not bai_out_path.exists() or bai_out_path.stat().st_size == 0:
        raise FileNotFoundError(f"Expected BAI index not found at {bai_out_path}")

    # 8. Materialize
    ctx.materialize_file(
        source={"main": bam_out_path, "index": bai_out_path},
        name=bam_out_path.name,
        output_key="bam",
        move=True,
    )
