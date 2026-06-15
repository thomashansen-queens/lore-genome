"""
Plugin for fasterq-dump, sratoolkit's multi-threaded SRA extraction tool.
"""
import subprocess

import lore
from .config import get_sra_binary, isolated_vdb_env


class FasterqDumpInputs:
    """Inputs for fasterq-dump Task"""
    srr_accession = lore.ArtifactInput(
        accepted_data=["sra_accession", "srr_accession"],
        label="SRR Accession",
        select="single",
        description="The SRR accession to download and extract.",
        examples=["SRR000123 with no suffix"],
    )
    sra_lite = lore.ValueInput(
        bool,
        description=(
            "Save space and bandwidth by getting 'sra-lite' files instead of full FASTQ. "
            "have compressed quality scores (not Phred). Useful if you do not need full "
            "quality information, but not compatible with all downstream tools."
        ),
        default=False,
    )


class FasterqDumpOutputs:
    """Outputs for fasterq-dump Task"""
    fastq_files = lore.TaskOutput(
        data_type="fastq",
        label="Extracted FASTQ Reads",
        description="The extracted FASTQ files from the SRA run.",
        is_primary=True,
        yields="multiple",
    )


@lore.task(
    "sra.fasterq_dump",
    inputs=FasterqDumpInputs,
    outputs=FasterqDumpOutputs,
    name="SRA Toolkit fasterq-dump",
    category="SRA Toolkit",
    icon="⬇️",
    preview_mode="dry_run",
)
def fasterq_dump_handler(
    ctx: lore.ExecutionContext,
    srr_accession: list[str],
    sra_lite: bool,
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
    threads = str(sra_config.get("default_threads", 6))

    fasterq_dump_binary = get_sra_binary(sra_config, "fasterq-dump")
    prefetch_binary = get_sra_binary(sra_config, "prefetch")

    # 2. The materializer hands the handler a list of accessinos
    if not srr_accession:
        raise ValueError("No SRR accession provided for fasterq-dump.")
    if len(srr_accession) > 1:
        ctx.logger.warning(
            f"Multiple accessions provided ({len(srr_accession)}). "
            "fasterq-dump will only process the first one: "
            f"{srr_accession[0]}"
        )
    clean_accession = srr_accession[0].strip().split(".")[0]

    # 2. Set up output directory for massive FASTQ files
    out_dir = ctx.get_temp_dir(f"{clean_accession}_fastq")

    # 3. Pre-fetch compressed file locally
    cmd_prefetch = [
        prefetch_binary,
        clean_accession,
        "--max-size", "100G",
        "--verbose",
    ]
    if sra_lite:
        ctx.logger.warning("SRA Lite mode enabled. Original quality scores will be discarded.")
        cmd_prefetch.append("--eliminate-quals")

    # 3. Execute
    # --split-3 means paired-end reads will be split into _1 and _2 files, while any orphaned reads are _3
    # cmd = [
    #     fasterq_dump_binary,
    #     clean_accession,
    #     "--split-3",
    #     "--outdir", str(out_dir),
    #     "--temp", str(out_dir),
    #     "--threads", threads,
    #     "--force",
    # ]

    with isolated_vdb_env(sra_config, ctx) as safe_env:
        try:
            ctx.logger.info(f"Executing: {' '.join(cmd_prefetch)}")
            subprocess.run(cmd_prefetch, env=safe_env, check=True)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SRA toolkit failed for '{clean_accession}': {e}") from e
            # raise RuntimeError(f"fasterq-dump failed for {clean_accession}: {e}") from e

    # 4. Gather and materialize outputs
    # extracted_files = list(out_dir.glob("*.fastq"))
    # if not extracted_files:
    #     raise FileNotFoundError(f"No FASTQ files were extracted for {clean_accession} in {out_dir}")

    # for fastq_file in extracted_files:
    #     ctx.materialize_file(
    #         source_path=fastq_file,
    #         output_key="fastq_files",
    #         name=fastq_file.name,
    #         description=f"Extracted FASTQ file for {clean_accession}",
    #         move=True,  # Defaults to True, but be explicit to avoid copying large files
    #     )
