"""
Plugin to run FAL_prediction to find bacterial fribrillar adhesins-like (FA-Like) proteins
by Monzon, V., Lafita, A., and Bateman, A.
https://github.com/VivianMonzon/FAL_prediction/tree/main

This tool has a number of pre-requisites that need to be installed before it will run.
IUPred2A: https://github.com/doszilab/iupred2a
Clone this repo and add the appropriate path in the configuration settings.

T-REKS: https://github.com/lafita/treks-hpc
I have rebuilt the project from source (commit 6d807c9) and vendored the JAR file with this
plugin. Look for 'T-ReksHPC.jar'.

MUSCLE v3: conda install -c bioconda muscle=3.8.1551
Is used by T-REKS and is not accepted by the FAL_prediction script, so the easiest way is
to install it via conda so that it is available in your PATH. Alternatively, you could
download the binary (https://www.drive5.com/muscle3/) and add it to your PATH.

HMMER: conda install -c bioconda hmmer
Same as the above, install via conda or download the binary (http://hmmer.org/) and add
it to your PATH.

scikit-learn: pip install scikit-learn
Used by the FAL_prediction script for its machine learning model. Install via pip or conda.

TODO: The T-Reks output is currently discarded, but it would be quite interesting to keep it
and visualize it using the built-in visualization system used for InterProScan.
"""
import lore
from pathlib import Path
import shutil
import subprocess


@lore.config(key="fal_prediction", title="FAL_prediction")
class FALPredictionConfig:
    """
    FAL_prediction by Monzon, V., Lafita, A., and Bateman, A.
    https://github.com/VivianMonzon/FAL_prediction/tree/main
    """
    script_path = lore.ValueInput(
        str | None,
        default=None,
        examples=["Leave blank if you have added ML_predict.py to your PATH"],
        label="Path to FAL_prediction python script",
        description="Provide the full path to the FAL_prediction python script (e.g. /thomas/FAL_prediction/ML_predict.py).",
    )
    treks_dir = lore.ValueInput(
        str | None,
        default=None,
        examples=["Path to the directory containing T-ReksHPC.jar"],
        label="Path to T-REKS directory",
        description="Provide the full path to the directory containing T-ReksHPC.jar (e.g. /thomas/FAL_prediction/).",
    )
    iupred_dir = lore.ValueInput(
        str | None,
        default=None,
        examples=["Path to the directory containing iupred2a.py"],
        label="Path to IUPred2A directory",
        description="Provide the full path to the IUPred2A directory (e.g. /home/thomas/Applications/iupred2a/iupred2a).",
    )


class FALPredictionInputs:
    """Inputs for FAL_prediction Task"""
    input_file = lore.ArtifactInput(
        accepted_data=["protein_fasta", "faa", "fasta"],
        label="Protein Sequences",
        select="single",
        load_as="path",
    )


class FALPredictionOutputs:
    """Outputs for FAL_prediction Task"""
    output_file = lore.TaskOutput(
        data_type="tabular",
        label="FAL_prediction Output",
        description="A tabular file containing the FAL_prediction results.",
        is_primary=True,
    )


def _check_system_deps(ctx: lore.ExecutionContext):
    """Fail fast if dependencies are missing. Let the user know what to install."""
    # import_name -> pip package name
    pkg_map = {
        "sklearn": "scikit-learn",
        "Bio": "biopython",
    }
    required_bins = ["java", "muscle", "hmmsearch"]
    missing_pkgs = []
    missing_bins = []

    # 1. Check python packages
    for import_name, pkg_name in pkg_map.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_pkgs.append(pkg_name)

    # 2. Check system binaries
    for bin_name in required_bins:
        if shutil.which(bin_name) is None:
            missing_bins.append(bin_name)

    # 3. Raise errors if any dependencies are missing
    if missing_pkgs:
        raise RuntimeError(
            f"Missing required Python dependencies: {', '.join(missing_pkgs)}. "
            "Ensure they are installed in your conda environment."
        )
    if missing_bins:
        raise RuntimeError(
            f"Missing required system dependencies: {', '.join(missing_bins)}. "
            "Ensure they are installed and available in your PATH or conda env."
        )


@lore.task(
    "FAL_prediction",
    description="Run FAL_prediction to find bacterial fribrillar adhesins-like (FA-Like) proteins",
    inputs=FALPredictionInputs,
    outputs=FALPredictionOutputs,
    icon="🦠",
)
def run_fal_prediction(
    ctx: lore.ExecutionContext,
    input_file: lore.PathBundle,
):
    """Run FAL_prediction to find bacterial fribrillar adhesins-like (FA-Like) proteins"""
    # 1. Check system dependencies and build config
    _check_system_deps(ctx)

    config = ctx.get_config("fal_prediction")
    script_path = config.script_path or "ML_predict.py"
    fal_root_dir = Path(script_path).resolve().parent

    if not config.treks_dir:
        raise ValueError("T-REKS directory is not set in the configuration. T-ReksHPC.jar is required.")
    if not config.iupred_dir:
        raise ValueError("IUPred2A directory is not set in the configuration. iupred2a.py is required.")

    # 2. Define outputs
    tmp_dir = ctx.get_temp_path("fal_prediction_tmp")
    out_dir = ctx.get_temp_path("fal_prediction_out")

    # 3. Build and run the command
    command = [
        "python", script_path, "predict",
        "--fasta_seqs", input_file.main,
        "--treks_dir", config.treks_dir,
        "--iupred_dir", config.iupred_dir,
        "--analysisfolder", tmp_dir,
        "--resultsfolder", out_dir,
        "--jobname", "FAL_prediction",
    ]
    ctx.logger.info(f"Running FAL_prediction with command: {' '.join(map(str, command))}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(fal_root_dir),
    )
    if process.stdout:
        for line in process.stdout:
            ctx.logger.info(line.strip())

    process.wait()

    if process.returncode != 0:
        ctx.logger.error("FAL_prediction failed! See the logs for details.")
        raise ValueError("FAL_prediction failed! See the logs for details.")

    # 4. Process output - keep result, throw away temp files
    result_file = Path(out_dir) / "results_FAL_prediction.tsv"
    if not result_file.exists():
        raise FileNotFoundError(f"Expected output file not found: {result_file}")

    ctx.materialize_file(
        result_file,
        data_type="FAL_prediction_output",
        label="FAL_prediction Output",
        description="A tabular file containing the FAL_prediction results.",
    )
