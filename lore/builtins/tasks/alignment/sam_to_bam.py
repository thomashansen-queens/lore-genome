"""
Task to convert a SAM file to BAM format.
"""
import lore
from pathlib import Path
import subprocess


@lore.config(key="samtools", title="SAM Tools")
class SamToolsConfig:
    """Global settings for SAM Tools"""
    binary_path = lore.ValueInput(
        str | None,
        default=None,
        examples=["Leave blank if you have added samtools to your PATH"],
        label="Path to SAM Tools binary",
        description="Provide the full path to the SAM Tools binary (e.g. C:/samtools-1.10/samtools).",
    )


class SamToBamInputs:
    """Inputs for SAM to BAM Task"""
    sam_file = lore.ArtifactInput(
        accepted_data=["sam"],
        label="Input SAM File",
        select="single",
        load_as="path",
        description="The SAM file to convert to BAM format.",
    )


class SamToBamOutputs:
    """Outputs for SAM to BAM Task"""
    bam = lore.TaskOutput(
        data_type="bam",
        label="Output BAM File",
        is_primary=True,
        yields="single",
    )


@lore.task(
    "alignment.sam_to_bam",
    inputs=SamToBamInputs,
    outputs=SamToBamOutputs,
    name="SAM to BAM Conversion",
    category="Alignment",
    preview_mode="dry_run",
    icon="🔄",
)
def sam_to_bam_handler(
    ctx: lore.ExecutionContext,
    sam_file: str,
):
    """
    Convert a plaint-text Sequence Alignment/Map (SAM) file to a compressed
    BAM (binary) format. Generates the required .bai index for instant
    lookup.
    """
    # 1. Config extraction
    config_model = ctx.get_config("samtools")
    samtools_config = config_model.model_dump() if config_model else {}

    samtools_binary = "samtools"
    if samtools_config.get("binary_path"):
        samtools_binary = samtools_config["binary_path"]

    threads = "4"

    # 2. Output pathing
    sam_path = Path(sam_file)
    base_name = sam_path.name

    # Strip extensions
    while base_name.endswith((".sam", ".bam", ".bai", ".gz")):
        base_name = Path(base_name).stem
    base_name = base_name.removesuffix("_aligned").removesuffix("_mapped").removesuffix("_bowtied")

    bam_out_path = ctx.get_temp_path(f"{base_name}.bam")
    bai_out_path = ctx.get_temp_path(f"{base_name}.bai")

    # --- Phase 1: Sort and compress ---
    cmd_sort = [
        samtools_binary,
        "sort",
        "-@", threads,
        "-o", str(bam_out_path),
        str(sam_path),
    ]
    ctx.logger.info(f"Running SAM to BAM conversion: {' '.join(cmd_sort)}")

    result_sort = subprocess.run(cmd_sort, capture_output=True, text=True)
    if result_sort.returncode != 0:
        ctx.logger.error(f"Error in sort: {result_sort.stderr}")
        raise RuntimeError(f"SAM sort failed: {result_sort.stderr}")

    if not bam_out_path.exists() or bam_out_path.stat().st_size == 0:
        raise FileNotFoundError(f"Expected BAM output not found at {bam_out_path}")

    # --- Phase 2: Index the BAM file ---
    cmd_index = [
        samtools_binary,
        "index",
        "-@", threads,
        str(bam_out_path),
        str(bai_out_path),
    ]
    ctx.logger.info(f"Running BAM indexing: {' '.join(cmd_index)}")

    result_index = subprocess.run(cmd_index, capture_output=True, text=True)
    if result_index.returncode != 0:
        ctx.logger.error(f"Error in index: {result_index.stderr}")
        raise RuntimeError(f"BAM indexing failed: {result_index.stderr}")

    if not bai_out_path.exists() or bai_out_path.stat().st_size == 0:
        raise FileNotFoundError(f"Expected BAI index not found at {bai_out_path}")

    # --- Phase 3: Build the artifact bundle ---
    output_bundle = {
        "main": bam_out_path,
        "index": bai_out_path,
    }

    # --- Phase 3: Materialize ---
    ctx.materialize_file(
        source=output_bundle,
        name=bam_out_path.name,
        output_key="bam",
        move=True,
    )
