"""
Convert an aligned BAM file back into paired FASTQ files.
"""
from pathlib import Path
import lore
import pysam


class BamToFastqInputs:
    bam = lore.ArtifactInput(
        accepted_data=["bam"],
        select="single",
        load_as="path",
        label="Input BAM file",
    )


class BamToFastqOutputs:
    fastq_bundle = lore.TaskOutput(
        data_type="fastq",
        label="FastQ",
        yields="single",
        description="A bundle of FASTQ files generated from the BAM.",
    )


@lore.task(
    key="alignment.bam_to_fastq",
    name="BAM to FASTQ",
    description="Extracts sequences from a BAM file into synchronized Paired-End FASTQ files.",
    category="Alignment",
    inputs=BamToFastqInputs,
    outputs=BamToFastqOutputs,
)
def bam_to_fastq_handler(
    ctx: lore.ExecutionContext,
    bam: str,
):
    """
    Convert an aligned BAM file back into paired FASTQ files.
    """
    bam_path = Path(bam)
    base = bam_path.stem

    # 1. Name-sort the BAM so R1 and R2 are perfectly synchronized
    ctx.logger.info("Name-sorting BAM file to synchronize pairs...")
    ns_bam_path = ctx.get_temp_path(f"{base}_namesorted.bam")

    try:
        # -n: Sort by read name, -o: output file
        pysam.sort("-n", "-o", str(ns_bam_path), str(bam_path))
    except pysam.SamtoolsError as e:
        raise RuntimeError(f"Failed to name-sort BAM. Details: {e}")

    # 1. Prepare output paths
    r1_path = ctx.get_temp_path(f"{base}_R1.fastq")
    r2_path = ctx.get_temp_path(f"{base}_R2.fastq")
    singles_path = ctx.get_temp_path(f"{base}_singletons.fastq")
    unpaired_path = ctx.get_temp_path(f"{base}_unpaired.fastq")  # For any unpaired reads
    bleed_path = ctx.get_temp_path(f"{base}_bleed.fastq")  # For any reads that bleed into the other file

    ctx.logger.info("Extracting FASTQ reads from BAM...")

    # 2. Let the pysam's C-backend handle the heavy lifting
    # -1: Output for first-in-pair
    # -2: Output for second-in-pair
    # -s: Output for singletons/orphans
    # -n: Print read names exactly as they are (don't append /1 and /2)
    # -N: Append /1 and /2 to read names
    try:
        pysam.fastq(
            "-1", str(r1_path),
            "-2", str(r2_path),
            "-s", str(singles_path),
            "-0", str(unpaired_path),
            "-N", 
            str(ns_bam_path),
            save_stdout=str(bleed_path),  # Capture any anything that would otherwise go to stdout
        )
    except pysam.SamtoolsError as e:
        ctx.logger.error(f"Failed to extract FASTQ: {e}")
        raise RuntimeError(f"BAM to FASTQ conversion failed. Details: {e}")

    # 3. Quick sanity check on the outputs
    for p, label in [(r1_path, "R1"), (r2_path, "R2"), (singles_path, "Singletons")]:
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        ctx.logger.info(f"Generated {label}: {size_kb:.1f} KB")

    # 4. Materialize outputs as a single bundle
    bundle_source = {}
    output_map = [
        ("Read 1 (Pairs)", r1_path, "main"),
        ("Read 2 (Pairs)", r2_path, "paired"),
        ("Singletons", singles_path, "singleton"),
        ("Unpaired", unpaired_path, "unpaired"),
        ("Stdout Bleed", bleed_path, "extra"),
    ]

    for label, path, output_key in output_map:
        if path.exists() and path.stat().st_size > 0:
            size_kb = path.stat().st_size / 1024
            ctx.logger.info(f"Materializing {label}: {size_kb:.1f} kB")
            bundle_source[output_key] = path
        else:
            ctx.logger.debug(f"No {label} generated; skipping.")

    if bundle_source:
        ctx.materialize_file(
            source=bundle_source,
            output_key="fastq_bundle",
            name=base,
            move=True,
        )
    else:
        ctx.logger.debug(f"No valid reads were extracted from the BAM. Output will be empty.")
