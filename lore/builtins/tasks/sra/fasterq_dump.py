"""
Dumps compressed FASTQ files from prefetched SRA files using the SRA Toolkit's
fasterq-dump command. This task is designed to work in conjunction with the
fasterq-prefetch task, which downloads the SRA files from NCBI.
"""
import lore
from pathlib import Path
import subprocess
from .config import get_sra_binary, isolated_vdb_env


class FasterqDumpInputs:
    """Inputs for fasterq-dump Task"""
    sra_files = lore.ArtifactInput(
        accepted_data=["sra_file"],
        label="SRA Files",
        select="multiple",
        load_as="path",
        description="The SRA file(s) to extract FASTQ from (e.g. SRR390728.sra).",
    )


class FasterqDumpOutputs:
    """Outputs for fasterq-dump Task"""
    fastq_bundles = lore.TaskOutput(
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
    icon="②⮕",
    preview_mode="dry_run",
)
def fasterq_dump_handler(
    ctx: lore.ExecutionContext,
    sra_files: list[str],
):
    """
    Handler for fasterq-dump Task. Takes one or more compressed SRA files and
    extracts its FASTQ reads using the SRA Toolkit's fasterq-dump command.
    """
    if not sra_files:
        raise ValueError("No SRA files provided for extraction.")

    for sra_file in sra_files:
        if not Path(sra_file).exists():
            raise FileNotFoundError(f"SRA file not found: {sra_file}")

    # 1. Get SRA Toolkit configuration and binary path
    sra_config = ctx.get_config("sra_tools").model_dump() if ctx.get_config("sra_tools") else {}
    threads = str(sra_config.get("default_threads", 6))
    fasterq_dump_binary = get_sra_binary(sra_config, "fasterq-dump")

    with isolated_vdb_env(sra_config, ctx) as safe_env:
        for sra_file in sra_files:
            sra_path = Path(sra_file)
            clean_accession = sra_path.stem

            fastq_out_dir = ctx.get_temp_dir(f"{clean_accession}_fastq")
            temp_scratch = ctx.get_temp_dir(f"{clean_accession}_scratch")

            # 2. Build and run the fasterq-dump command
            cmd_fasterq = [
                fasterq_dump_binary,
                str(sra_path),
                "--split-3",  # Split paired reads, leave unpaired as single
                "--outdir", str(fastq_out_dir),
                "--temp", str(temp_scratch),
                "--threads", str(threads),
            ]

            ctx.logger.info(f"Extracting {clean_accession} to FASTQ...")
            process = subprocess.Popen(
                cmd_fasterq,
                env=safe_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output to logger
            if process.stdout:
                for line in process.stdout:
                    ctx.logger.info(line.strip())

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"fasterq-dump failed for {clean_accession}. Check logs.")

            # 3. Collect the output FASTQ files and build into an Artifact bundle
            extracted_files = list(fastq_out_dir.glob("*.fastq"))
            if not extracted_files:
                raise FileNotFoundError(f"No FASTQ files were extracted for {clean_accession}.")

            bundle_source = {}
            base_fastq = fastq_out_dir / f"{clean_accession}.fastq"
            read1_fastq = fastq_out_dir / f"{clean_accession}_1.fastq"
            read2_fastq = fastq_out_dir / f"{clean_accession}_2.fastq"

            # A. Paired reads
            if read1_fastq.exists():
                bundle_source["main"] = read1_fastq
                if read2_fastq.exists():
                    bundle_source["paired"] = read2_fastq
                if base_fastq.exists():
                    bundle_source["unpaired"] = base_fastq

            # B. Single reads
            elif base_fastq.exists():
                bundle_source["main"] = base_fastq

            # C. Unknown case?
            else:
                ctx.logger.warning(
                    f"Unexpected FASTQ naming schema for {clean_accession}. "
                    f"Found files: {[f.name for f in extracted_files]}"
                )
                extracted_files.sort()
                bundle_source["main"] = extracted_files[0]
                for i, f in enumerate(extracted_files[1:], start=1):
                    bundle_source[f"extra_{i}"] = f

            # 4. Materialize this bundle as a Fastq artifact
            ctx.materialize_file(
                source=bundle_source,
                output_key="fastq_bundles",
                name=clean_accession,
                move=True,
            )
