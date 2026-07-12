"""
Plugin for local, offline use of InterProScan.

https://interproscan-docs.readthedocs.io/en/v5/HowToDownload.html
"""
from enum import StrEnum
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import lore


@lore.config(key="interproscan", title="InterProScan")
class InterproscanConfig:
    """Global settings for InterProScan"""
    script_path = lore.ValueInput(
        str | None,
        default=None,
        examples=["Leave blank if you have added interproscan.sh to your PATH"],
        label="Path to InterProScan shell script",
        description="Provide the full path to the InterProScan shell script (e.g. /thomas/interproscan-5.78/interproscan.sh).",
    )
    default_threads = lore.ValueInput(
        int,
        default=8,
        min=1, max=32, step=1,
        label="Default CPU Threads",
        description="More threads makes more faster.",
        widget="slider",
    )


class InterproscanApplication(StrEnum):
    """InterProScan applications/databases"""
    CDD = "CDD"
    COILS = "COILS"
    GENE3D = "Gene3D"
    HAMAP = "HAMAP"
    MOBIDB = "MobiDBLite"
    PANTHER = "PANTHER"
    PFAM = "Pfam"
    PIRSF = "PIRSF"
    PRINTS = "PRINTS"
    PROSITEPATTERNS = "PROSITEPATTERNS"
    PROSITEPROFILES = "PROSITEPROFILES"
    SFLD = "SFLD"
    SMART = "SMART"
    SUPERFAMILY = "SUPERFAMILY"
    NCBIFAM = "NCBIFAM"


class InterproscanInputs:
    """Inputs for InterProScan Task"""
    input_file = lore.ArtifactInput(
        accepted_data=["protein_fasta", "faa", "fasta"],
        label="Protein Sequences",
        select="single",
        load_as="path",
        description="The protein sequences to scan in FASTA format.",
    )
    disable_precalc = lore.ValueInput(
        bool,
        default=False,
        label="Disable Precalculated Matches",
        description=(
            "EBI has provided a web API with precalculated matches to more than 500 million protein sequences, "
            "including all of the sequence in UniProtKB. Check this if you do not have internet access, or "
            "your firewall prevents access to http://www.ebi.ac.uk. This will force InterProScan to "
            "perform a full scan. This may take significantly longer. "
            "For details: https://interproscan-docs.readthedocs.io/en/v5/HowToDownload.html#using-the-local-pre-calculated-match-lookup-service-optional"
        ),
    )
    seq_type = lore.ValueInput(
        Literal["protein", "nucleotide"],
        default="protein",
        label="Sequence Type",
        description="InterProScan can analyze translated nucleotide sequences.",
    )
    appl = lore.ValueInput(
        list[InterproscanApplication],
        default=[a for a in InterproscanApplication],
        description="The database(s) to scan against. By default, all databases are used.",
        label="InterProScan Applications",
        widget="checkbox_group",
    )


class InterproscanOutputs:
    """Outputs for InterProScan Task"""
    annotations_json = lore.TaskOutput(
        data_type="interproscan_json",
        label="InterProScan Output JSON File",
        is_primary=True,
        yields="single",
    )
    annotations_xml = lore.TaskOutput(
        data_type="interproscan_xml",
        label="InterProScan Output XML File",
        is_primary=False,
        yields="single",
    )


def _preflight_check(ctx: lore.ExecutionContext, ips_binary_path: str):
    """Check dependencies and paths before attempting to run InterProScan."""
    dependencies = ["java", "perl", "python3"]
    missing = []

    for dep in dependencies:
        if shutil.which(dep) is None:
            missing.append(dep)

    if missing:
        ctx.logger.error(f"Missing required system dependencies: {', '.join(missing)}")
        raise RuntimeError(
            f"InterProScan requires {', '.join(missing)} to be installed and available on the system PATH."
        )

    if "/" in ips_binary_path or "\\" in ips_binary_path:
        if not Path(ips_binary_path).exists():
            raise FileNotFoundError(f"InterProScan script not found at: {ips_binary_path}")
    else:
        if shutil.which(ips_binary_path) is None:
            raise FileNotFoundError(f"InterProScan command '{ips_binary_path}' not found in PATH.")


@lore.task(
    key="annotation.interproscan",
    inputs=InterproscanInputs,
    outputs=InterproscanOutputs,
    name="InterProScan",
    category="Annotation",
    preview_mode="dry_run",
    icon="I",
)
def interproscan_handler(
    ctx: lore.ExecutionContext,
    input_file: str,
    disable_precalc: bool = False,
    seq_type: Literal["protein", "nucleotide"] = "protein",
    appl: list[InterproscanApplication] = [a for a in InterproscanApplication],
):
    """
    Scan protein sequences for functional domains and motifs using InterProScan.
    """
    # 1. Config & Pre-flight
    config_model = ctx.get_config("interproscan")
    ipr_config = config_model.model_dump() if config_model else {}

    iprscan_bin = ipr_config.get("script_path") or "interproscan.sh"
    threads = str(ipr_config.get("default_threads", 8))

    _preflight_check(ctx, iprscan_bin)

    fasta_path = Path(input_file)
    base_name = fasta_path.stem

    # 2. Setup Output Directory
    # InterProScan can leave massive temp folders. Confining it is safe.
    out_dir = ctx.get_temp_dir("ips_output")
    out_prefix = out_dir / f"{base_name}_ips"
    ips_temp_dir = ctx.get_temp_dir("ips_temp")

    # 3. Build Command
    cmd = [
        iprscan_bin,
        "-i", str(fasta_path),
        "-b", str(out_prefix),
        "-t", "n" if seq_type == "nucleotide" else "p",
        "-f", "JSON,XML",
        "-cpu", threads,
        "--tempdir", str(ips_temp_dir),
        "-appl", ",".join(appl),
    ]

    if disable_precalc:
        cmd.append("-dp")
        ctx.logger.info("Offline mode enabled: Pre-calculated lookup service disabled (-dp).")

    ctx.logger.info(f"Running InterProScan: {' '.join(cmd)}")

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
            # IPRScan is very chatty. Stream it so the user knows it hasn't hung.
            ctx.logger.info(line.strip())

    process.wait()

    if process.returncode != 0:
        raise RuntimeError("InterProScan analysis failed. Check the logs for details.")

    # 5. Locate outputs
    out_json = out_dir / f"{base_name}_ips.json"
    out_xml = out_dir / f"{base_name}_ips.xml"

    if not out_json.exists() or not out_xml.exists():
        raise FileNotFoundError("InterProScan completed, but expected output files are missing.")

    # 6. Materialize
    ctx.materialize_file(
        source=out_json,
        output_key="annotations_json",
        name=out_json.name,
        metadata={"description": "InterProScan JSON Annotations"},
        move=True,
    )

    ctx.materialize_file(
        source=out_xml,
        output_key="annotations_xml",
        name=out_xml.name,
        metadata={"description": "InterProScan XML Annotations"},
        move=True,
    )
