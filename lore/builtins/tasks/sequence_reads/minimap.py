"""
Plugin for Minimap2

https://github.com/lh3/minimap2
"""
from enum import StrEnum
import lore
from pathlib import Path
import subprocess


@lore.config(key="minimap2", title="Minimap 2")
class Minimap2Config:
    """Global settings for Minimap2"""
    binary_path = lore.ValueInput(
        str | None,
        default=None,
        examples=["Leave blank if you have added minimap2 to your PATH"],
        label="Path to Minimap2 binary",
        description="Provide the full path to the Minimap2 binary (e.g. C:/minimap2-2.24_x64-linux/minimap2).",
    )


class Minimap2PresetOptions(StrEnum):
    """Presets for Minimap2 based on sequencing technology."""
    PACBIO_OLD = "map-pb"
    PACBIO_NEW = "map-hifi"
    ONT = "map-ont"
    LONG_READ_HIGH_QUALITY = "lr:hq"
    SHORT_READ = "sr"
    ILLUMINA_CLR = "map-iclr"


class Minimap2Inputs:
    """Inputs for Minimap2 Task"""
    target = lore.ArtifactInput(
        accepted_data=["genome_fasta", "nucleotide_fasta", "fna", "fa"],
        label="Reference Sequence",
        select="single",
        load_as="path",
        description="The reference to which reads will be aligned.",
    )
    query = lore.ArtifactInput(
        accepted_data=["fastq", "fq"],
        label="Reads to Align",
        select="multiple",
        load_as="path",
        description=(
            "The sequencing reads to align in FASTQ format. Can be single-end "
            "or paired-end FASTQ files. gzip'd files are also accepted."
        ),
    )
    preset = lore.ValueInput(
        Minimap2PresetOptions,
        label="Sequencing Technology",
        description=(
            "Minimap2 preset based on sequencing technology.\n"
            "Pacbio Old: PacBio CLR reads (for very old PacBio data)\n"
            "Pacbio New: PacBio HiFi reads\n"
            "ONT: Noisy long reads of ~10% error rate (default)\n"
            "Long read (high quality): Accurate long reads of <1% error rate\n"
            "Short read: Illumina short reads\n"
            "Illumina ICLR: Illumina complete long reads (ICLR)\n"
        ),
        default=Minimap2PresetOptions.ONT,
    )


class Minimap2Outputs:
    """Outputs for Minimap2 Task"""
    alignment_file = lore.TaskOutput(
        data_type="sam",
        label="Alignment SAM File",
        is_primary=True,
        yields="single",
    )


@lore.task(
    key="alignment_minimap2",
    inputs=Minimap2Inputs,
    outputs=Minimap2Outputs,
    name="Minimap2 Alignment",
    category="Sequence reads",
    preview_mode="dry_run",
    icon="M",
)
def minimap2_alignment_handler(
    ctx: lore.ExecutionContext,
    target: str,
    query: list[str],
    preset: Minimap2PresetOptions,
):
    """
    Aligns sequencing reads to a reference database using Minimap2.

    minimap2 takes a reference database and a query sequence file as
    input and produce approximate mapping, without base-level alignment
    (i.e. coordinates are only approximate and no CIGAR in output).
    """
    # 1. Config extraction
    config_model = ctx.get_config("minimap2")
    minimap2_config = config_model.model_dump() if config_model else {}

    minimap2_binary = "minimap2"
    if minimap2_config.get("binary_path"):
        minimap2_binary = minimap2_config["binary_path"]
    threads = "4"

    # 2. Output pathing
    if not query:
        raise ValueError("No query FASTQ files provided for Minimap2 alignment.")

    first_query = Path(query[0])
    base_name = first_query.stem.replace("_1", "")  # _1 common in paired-end reads
    sam_out_path = ctx.get_temp_path(f"{base_name}_aligned.sam")

    # 3. Build and execute Minimap2 command
    # Should we index the target sequences? minimap2 -d target.mmi target.fa
    cmd = [
        minimap2_binary,
        "-a",  # Output in SAM format (defaults to PAF)
        "-x", preset.value,
        "-t", threads,
        "--seed", "42",  # Fixed seed for reproducibility
        "-o", str(sam_out_path),
        str(target),
    ]
    cmd.extend([str(q) for q in query])

    ctx.logger.info("Running Minimap2 with command: " + " ".join(cmd))

    # 4. Execute
    process = subprocess.Popen(
        cmd,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if process.stdout:
        for line in process.stdout:
            ctx.logger.info(line.strip())

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"Minimap2 alignment failed. Check the logs for details.")

    if not sam_out_path.exists() or sam_out_path.stat().st_size == 0:
        raise FileNotFoundError(f"Expected output SAM file not found at {sam_out_path}.")

    # 5. Materialize output
    ctx.materialize_file(
        source=sam_out_path,
        output_key="alignment_file",
        name=sam_out_path.name,
        metadata={
            "description": f"Minimap2 alignment of {base_name} to {target}.",
        },
        move=True,
    )
