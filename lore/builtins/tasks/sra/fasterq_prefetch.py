"""
fasterq-dump is massively faster than fastq-dump, and even faster if you
'prefetch' the SRA files first. This plugin wraps the fasterq-dump prefetch
command.
"""
import lore
import subprocess
from .config import get_sra_binary, isolated_vdb_env


class FasterqDumpInputs:
    """Inputs for fasterq-dump Task"""
    srr_accession = lore.ArtifactInput(
        accepted_data=["sra_accession", "srr_accession"],
        label="SRA Accession",
        select="multiple",
        load_as="adapted",
        description="The SRA accession to download and extract (e.g. SRR390728).",
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
    sra_files = lore.TaskOutput(
        data_type="sra_file",
        label="Downloaded SRA File",
        is_primary=True,
        yields="multiple",
    )


@lore.task(
    "sra.fasterq_prefetch",
    inputs=FasterqDumpInputs,
    outputs=FasterqDumpOutputs,
    name="SRA Toolkit fasterq-dump prefetch",
    category="SRA Toolkit",
    icon="①⬇",
    preview_mode="dry_run",
)
def fasterq_prefetch_handler(
    ctx: lore.ExecutionContext,
    srr_accession: list[str],
    sra_lite: bool,
    resume: bool,
    force: bool,
):
    """Handler for fasterq-dump prefetch Task"""
    if not srr_accession:
        raise ValueError("No SRR accession provided for fasterq-dump prefetch.")

    # 1. Config extraction
    config_model = ctx.get_config("sra_tools")
    sra_config = config_model.model_dump() if config_model else {}
    prefetch_binary = get_sra_binary(sra_config, "prefetch")

    # 3. Set up output directory for massive FASTQ files
    # prefetch downloads as ERR12345/ERR12345.sra
    global_cache = sra_config.get("cache_dir") or ctx.runtime.cache_dir

    with isolated_vdb_env(sra_config, ctx) as safe_env:
        for acc in srr_accession:
            # 4. Clean up accession to remove any version suffixes
            clean_accession = acc.strip().split(".")[0]
            if not clean_accession:
                ctx.logger.warning(f"Skipping empty accession: {acc}")
                continue

            # 5. Build prefetch command
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

            # 6. Execute prefetch
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

            # 7. Verify that the SRA file was downloaded
            sra_files = list((global_cache / clean_accession).rglob("*.sra"))
            if not sra_files:
                ctx.logger.error(f"No .sra files found in cache directory {global_cache} after prefetch.")
                continue

            target_sra = sra_files[0]

            # 8. Materialize the downloaded SRA file as an artifact
            ctx.materialize_file(
                source=target_sra,
                output_key="sra_files",
                name=clean_accession,
                move=True,
            )
