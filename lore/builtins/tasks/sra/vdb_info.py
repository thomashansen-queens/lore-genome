"""
Plugin for vdb-dump, one of sratoolkit's core utilities for looking
at the contents of SRA files.
"""
import json
import lore
import subprocess
from time import sleep

from .config import get_sra_binary, isolated_vdb_env


class VdbDumpInputs:
    """Inputs for vdb-dump Task"""
    sra_accession = lore.ArtifactInput(
        accepted_data=["sra_accession", "srr_accession"],
        label="SRA Accession",
        description="The SRA accession to dump (e.g. SRR390728).",
        examples=["SRR000123 with no suffix"],
    )


class VdbDumpOutputs:
    """Outputs for vdb-dump Task"""
    run_metadata = lore.TaskOutput(
        data_type="sra_metadata",
        label="Run Metadata",
        description="Metadata about the SRA run, including sample info, instrument, library prep, etc.",
        is_primary=True,
    )


@lore.task(
    "sra.vdb_dump",
    inputs=VdbDumpInputs,
    outputs=VdbDumpOutputs,
    name="SRA Toolkit vdb-dump (--info)",
    category="SRA Toolkit",
    icon="📦",
)
def vdb_dump_handler(
    ctx: lore.ExecutionContext,
    sra_accession: list[str],
):
    """Handler for vdb-dump Task"""
    sra_config = ctx.get_config("sra_tools")
    vdb_dump_binary = get_sra_binary(sra_config.model_dump(), "vdb-dump")

    clean_accession = [acc.strip().split(".")[0] for acc in sra_accession if acc]
    table_data = []

    # Execute within our safely isolated VDB cache env
    with isolated_vdb_env(sra_config.model_dump(), ctx) as safe_env:
        for accession in clean_accession:
            ctx.logger.info("Running vdb-dump for accession: %s", accession)
            
            # Construct the CLI command
            cmd = [
                vdb_dump_binary,
                "--info",
                accession,
                "--format",
                "json",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    env=safe_env,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                if result.stdout:
                    metadata = json.loads(result.stdout)
                    table_data.append(metadata)
                else:
                    ctx.logger.warning(f"vdb-dump returned empty stdout for {accession}")

            except subprocess.CalledProcessError as e:
                ctx.logger.error("vdb-dump failed: %s", e.stderr)
                raise RuntimeError(f"vdb-dump failed with error: {e.stderr}") from e
            except json.JSONDecodeError as e:
                ctx.logger.error("Failed to parse vdb-dump output as JSON: %s", e)
                raise ValueError(f"vdb-dump output is not valid JSON: {e}") from e

            # Be a good citizen of the NCBI servers
            sleep(1.0)

    if not table_data:
        raise ValueError(f"vdb-dump returned empty or unparsable metadata")

    ctx.materialize_content(
        content=json.dumps(table_data, indent=2),
        output_key="run_metadata",
        name=f"VDB Run Metadata",
        extension="json",
        metadata={"columns": list(table_data[0].keys())},
    )
