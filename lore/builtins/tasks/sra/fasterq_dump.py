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
        label="SRA Lite mode",
        description=(
            "Save space and bandwidth by getting 'sra-lite' files instead of full FASTQ. "
            "have compressed quality scores (not Phred). Useful if you do not need full "
            "quality information, but not compatible with all downstream tools."
        ),
        default=False,
    )
    resume = lore.ValueInput(
        bool,
        label="Resume previous run",
        description=(
            "If a previous run was interrupted or failed, enable this to attempt to resume "
            "from where it left off. This will check for existing prefetch files and "
            "partially extracted FASTQ files, and skip re-downloading/re-extracting those."
        ),
        default=True,
    )
    force = lore.ValueInput(
        bool,
        label="Force re-download",
        description=(
            "Force re-download and extraction even if output files already exist. Uses "
            "time and bandwidth, but helpful in case of previous failed/interrupted runs."
        ),
        default=False,
    )


class FasterqDumpOutputs:
    """Outputs for fasterq-dump Task"""
    fastq_files = lore.TaskOutput(
        data_type="fastq",
        label="Extracted FASTQ Reads",
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
    resume: bool,
    force: bool
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

    prefetch_binary = get_sra_binary(sra_config, "prefetch")
    fasterq_dump_binary = get_sra_binary(sra_config, "fasterq-dump")

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

    # 3. Set up output directory for massive FASTQ files
    # prefetch downloads as ERR12345/ERR12345.sra
    global_cache = sra_config.get("cache_dir") or ctx.runtime.cache_dir

    fastq_out_dir = ctx.get_temp_dir(f"{clean_accession}_fastq")

    # --- Phase 1: Pre-fetch compressed file locally ---
    # 4. Build prefetch command
    cmd_prefetch = [
        prefetch_binary,
        clean_accession,
        "--max-size", "100G",
        "--output-directory", str(global_cache),
    ]

    if sra_lite:
        ctx.logger.warning("SRA Lite mode enabled. Original quality scores will be discarded.")
        cmd_prefetch.append("--eliminate-quals")
    if force:
        cmd_prefetch.extend(["--force", "all"])
    if resume:
        cmd_prefetch.extend(["--resume", "yes"])

    ctx.logger.info("Phase 1 (Prefetch): Downloading SRA data with prefetch...")
    ctx.logger.info("Command: " + " ".join(cmd_prefetch))

    # 5. Execute prefetch
    with isolated_vdb_env(sra_config, ctx) as safe_env:
        prefetch_process = subprocess.Popen(
            cmd_prefetch,
            bufsize=1,  # Line-buffered output,
            env=safe_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if prefetch_process.stdout:
            for line in prefetch_process.stdout:
                ctx.logger.info(line.strip())

        prefetch_process.wait()

        if prefetch_process.returncode != 0:
            raise RuntimeError(
                f"SRA prefetch failed for '{clean_accession}'. Check logs for details."
            )

        sra_files = list((global_cache / clean_accession).rglob("*.sra"))
        if not sra_files:
            raise FileNotFoundError(
                f"No .sra files found in cache directory {global_cache} after prefetch."
            )
        target_sra = sra_files[0]

        # --- Phase 2: Pre-fetch compressed file locally ---
        # 6. Build fasterq-dump command
        cmd_fasterq = [
            fasterq_dump_binary,
            str(target_sra),
            "--split-3",
            "--outdir", str(fastq_out_dir),
            "--temp", str(fastq_out_dir),
            "--threads", threads,
        ]
        if force:
            cmd_fasterq.extend(["--force", "all"])

        ctx.logger.info("Phase 2 (Fasterq-dump): Converting SRA data to FASTQ...")
        ctx.logger.info("Command: " + " ".join(cmd_fasterq))
        fasterq_process = subprocess.Popen(
            cmd_fasterq,
            bufsize=1,  # Line-buffered output
            env=safe_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if fasterq_process.stdout:
            for line in fasterq_process.stdout:
                ctx.logger.info(line.strip())

        fasterq_process.wait()

        if fasterq_process.returncode != 0:
            raise RuntimeError(
                f"SRA fasterq-dump failed for '{clean_accession}'. Check logs for details."
            )

    # --- Phase 3: Prepare outputs ---
    # 7. Materialize output fastq files
    extracted_files = list(fastq_out_dir.glob("*.fastq"))
    if not extracted_files:
        raise FileNotFoundError(
            f"No FASTQ files were extracted for {clean_accession} in {fastq_out_dir}"
        )

    extracted_files.sort()
    for fastq_file in extracted_files:
        ctx.materialize_file(
            source_path=fastq_file,
            output_key="fastq_files",
            name=fastq_file.name,
            metadata={
                "description": f"Extracted FASTQ file for {clean_accession}",
            },
            move=True,  # Defaults to True, but be explicit to avoid copying large files
        )
