"""
Plugin for Bowtie2

https://github.com/benlangmead/bowtie2
"""
from pathlib import Path
import subprocess

import lore


@lore.config(key="bowtie2", title="Bowtie2")
class Bowtie2Config:
    """Global settings for Bowtie2"""
    binary_path = lore.ValueInput(
        str | None,
        default=None,
        examples=["Leave blank if you have added bowtie2 to your PATH"],
        label="Path to Bowtie2 binary",
        description="Provide the full path to the Bowtie2 binary (e.g. C:/bowtie2-2.5.5-linux-x86_64/bowtie2).",
    )


class Bowtie2Inputs:
    """Inputs for Bowtie2 Task"""
    target = lore.ArtifactInput(
        accepted_data=["genome_fasta", "nucleotide_fasta", "fna", "fa"],
        label="Reference Sequence",
        select="single",
        load_as="path",
        description="The reference to which reads will be aligned.",
    )
    fastq_paths = lore.ArtifactInput(
        accepted_data=["fastq", "fq"],
        label="Reads to Align",
        select="multiple",
        load_as="path",
        description=(
            "The sequencing reads to align in FASTQ format. Can be single-end "
            "or paired-end FASTQ files. gzip'd files are also accepted."
        ),
    )


class Bowtie2Outputs:
    """Outputs for Bowtie2 Task"""
    alignment_file = lore.TaskOutput(
        data_type="sam",
        label="Alignment SAM File",
        is_primary=True,
        yields="single",
    )


@lore.task(
    key="alignment.bowtie2",
    inputs=Bowtie2Inputs,
    outputs=Bowtie2Outputs,
    name="Bowtie2 Alignment",
    category="Alignment",
    preview_mode="dry_run",
    icon="B",
)
def bowtie2_alignment_handler(
    ctx: lore.ExecutionContext,
    target: str,
    fastq_paths: list[lore.PathBundle],
):
    """
    Aligns short sequencing reads to a reference database using Bowtie2.
    Requires an indexing step, then performs base-level mapping.
    """
    ctx.logger.info("DEBUG fastq_paths: %r", fastq_paths)
    # 1. Flatten fastq_paths to a list of Path objects
    unique_paths = {}
    for bundle in fastq_paths:
        for path in bundle.get_all_paths():
            unique_paths[str(path)] = path
    all_files = list(unique_paths.values())

    # 2. Validation
    if not all_files:
        raise ValueError("No FASTQ files provided for alignment.")
    if len(all_files) > 2:
        ctx.logger.warning(
            "This task plugin currently only supports up to 2 FASTQ files (Paired-End). "
            "Using the first two files for alignment."
        )

    # 1. Config extraction
    config_model = ctx.get_config("bowtie2")
    bowtie2_config = config_model.model_dump() if config_model else {}

    bowtie2_binary = "bowtie2"
    if bowtie2_config.get("binary_path"):
        bowtie2_binary = bowtie2_config["binary_path"]
    bowtie2_build_binary = f"{bowtie2_binary}-build"

    # 2. Output pathing
    base_name = all_files[0].stem.replace("_1", "").replace("_R1", "")

    idx_dir = ctx.get_temp_dir("bt2_index")
    idx_prefix = idx_dir / "target_index"

    # 3. STEP 1: Build the Bowtie2 index
    cmd = [
        bowtie2_build_binary,
        str(target),
        str(idx_prefix),
    ]

    ctx.logger.info("Running Bowtie2 with command: " + " ".join(cmd))
    build_process = subprocess.Popen(
        cmd,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if build_process.stdout:
        for line in build_process.stdout:
            ctx.logger.info(line.strip())
    build_process.wait()

    if build_process.returncode != 0:
        ctx.logger.error("Bowtie2 index building failed. Check the logs for details.")
        raise RuntimeError("Bowtie2 index building failed. Check the logs for details.")

    # 4. STEP 2: Execute Bowtie2 mapping
    sam_out_path = ctx.get_temp_path(f"{base_name}_aligned.sam")
    map_cmd = [
        bowtie2_binary,
        "-x", str(idx_prefix),
        "-S", str(sam_out_path),
        "--threads", "4",
        "--seed", "42",  # Fixed seed for reproducibility
    ]

    # Handle single-end vs paired-end reads
    if len(all_files) == 1:
        ctx.logger.info("Detected single-end reads. %s", all_files[0])
        map_cmd.extend(["-U", str(all_files[0])])
    elif len(all_files) >= 2:
        ctx.logger.info("Detected paired-end reads. %s (forward) and %s (reverse)", all_files[0], all_files[1])
        map_cmd.extend(["-1", str(all_files[0]), "-2", str(all_files[1])])
        if len(all_files) > 2:
            ctx.logger.warning(">2 FASTQ files: only first two are used for paired-end alignment.")

    ctx.logger.info("Running Bowtie2 with command: " + " ".join(map_cmd))
    map_process = subprocess.Popen(
        map_cmd,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if map_process.stdout:
        for line in map_process.stdout:
            ctx.logger.info(line.strip())

    map_process.wait()

    if map_process.returncode != 0:
        raise RuntimeError(f"Bowtie2 alignment failed. Check the logs for details.")

    if not sam_out_path.exists() or sam_out_path.stat().st_size == 0:
        raise FileNotFoundError(f"Expected output SAM file not found at {sam_out_path}.")

    # 5. Materialize output
    ctx.materialize_file(
        source=sam_out_path,
        output_key="alignment_file",
        name=sam_out_path.name,
        metadata={
            "description": f"Bowtie2 alignment of {base_name} to {target}.",
            "reads_type": "Paired-End" if len(fastq_paths) == 2 else "Single-End",
        },
        move=True,
    )
