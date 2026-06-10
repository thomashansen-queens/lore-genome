"""
Plugin for fasterq-dump, sratoolkit's multi-threaded SRA extraction tool.
"""
import subprocess
from pathlib import Path

import lore
from .config import get_sra_binary, isolated_vdb_env


class FasterqDumpInputs:
    """Inputs for fasterq-dump Task"""
    srr_accession = lore.ArtifactInput(
        accepted_data=["sra_accession", "srr_accession"],
        label="SRR Accession",
        select=lore.SINGLE,
        description="The SRR accession to download and extract.",
        examples=["SRR000123 with no suffix"],
    )


class FasterqDumpOutputs:
    """Outputs for fasterq-dump Task"""
    fastq_files = lore.TaskOutput(
        data_type="fastq",
        label="Extracted FASTQ Reads",
        description="The extracted FASTQ files from the SRA run.",
        is_primary=True,
        yields=lore.MULTIPLE,
    )


@lore.task(
    "sra.fasterq_dump",
    inputs=FasterqDumpInputs,
    outputs=FasterqDumpOutputs,
    name="SRA Toolkit fasterq-dump",
    category="SRA Toolkit",
    icon="⬇️",
)
def fasterq_dump_handler(
    ctx: lore.ExecutionContext,
    srr_accession: str,
):
    """
    Downloads and extracts FASTQ files from the specified SRR accession in the 
    sequence read archive (SRA). Be aware that these files can be extremely large, 
    on the order of tens to hundreds of gigabytes, so ensure you have sufficient 
    disk space, bandwidth, time, and battery life before running this task.
    """
    # 1. Config extraction
    config_model = ctx.get_config("sra_tools")
    sra_config = config_model.model_dump() if config_model else {}
    
    fasterq_dump_binary = get_sra_binary(sra_config, "fasterq-dump")
    threads = str(sra_config.get("default_threads", 4))
    clean_accession = srr_accession.strip().split(".")[0]

    # 2. Set up output directory for massive FASTQ files
    out_dir = ctx.get_temp_path(f"{clean_accession}_fastq")
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx.logger.info(f"Initiating fasterq-dump for {clean_accession}...")

    # 3. Execute
    # --split-files means paired-end reads will be split into _1 and _2 files
    cmd = [
        fasterq_dump_binary,
        clean_accession,
        "--split-files",
        "--outdir", str(out_dir),
        "--temp", str(out_dir),
        "--threads", threads,
        "--force",
    ]

    with isolated_vdb_env(sra_config, ctx) as safe_env:
        try:
            subprocess.run(cmd, env=safe_env, check=True)
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"fasterq-dump failed for {clean_accession}: {e}") from e

    # 4. Gather and materialize outputs
    extracted_files = list(out_dir.glob("*.fastq"))
    if not extracted_files:
        raise FileNotFoundError(f"No FASTQ files were extracted for {clean_accession} in {out_dir}")

    for fastq_file in extracted_files:
        ctx.materialize_file(
            source_path=fastq_file,
            output_key="fastq_files",
            name=fastq_file.name,
            description=f"Extracted FASTQ file for {clean_accession}",
            move=True,  # Defaults to True, but be explicit to avoid copying large files
        )
