"""
SPAdes assembly task.
"""
from pathlib import Path
import shutil
import subprocess

import lore


@lore.config(key="spades", title="SPAdes de novo assembler")
class SpadesConfig:
    """Global settings for the SPAdes assembler."""
    script_path = lore.ValueInput(
        str,
        default="spades.py",
        label="Path to SPAdes Python script",
        description="Provide the full path if not in your system PATH.",
    )
    default_threads = lore.ValueInput(
        int,
        default=4,
        min=1, max=32, step=1,
        label="Default CPU Threads",
        description="Global default for all SPAdes tasks.",
        widget="slider",
    )
    default_memory = lore.ValueInput(
        int,
        default=16,
        min=1,
        label="Memory (GB)",
        description="Amount of memory to allocate for assembly.",
    )


class SpadesInputs:
    reads = lore.ArtifactInput(
        accepted_data=["fastq"],
        select="multiple",
        load_as="path",
        label="Input FASTQ files",
    )
    kmer_sizes = lore.ValueInput(
        list[int] | None,
        default=None,
        examples=["21, 33, 55 (leave blank to use SPAdes defaults)"],
        label="K-mer sizes",
        description="List of k-mer sizes to use for assembly.",
    )
    error_correction = lore.ValueInput(
        bool,
        default=False,
        label="Perform read error correction",
        description="BayesHammer error correction. Takes time, but can improve assembly quality.",
    )


class SpadesOutputs:
    contigs = lore.TaskOutput(
        data_type="nucleotide_fasta",
        label="Assembled contigs",
        description="FASTA file of assembled contigs.",
        is_primary=True,
        yields="single",
    )
    scaffolds = lore.TaskOutput(
        data_type="nucleotide_fasta",
        label="Assembled scaffolds",
        description="FASTA file of assembled scaffolds.",
        yields="single",
    )
    gfa = lore.TaskOutput(
        data_type="gfa",
        label="Assembly graph (GFA 1.2)",
        description="The De Brujin graph structure in GFA format.",
        yields="single",
    )
    fastg = lore.TaskOutput(
        data_type="fastg",
        label="Assembly graph (FASTG)",
        description="The De Brujin graph structure in FASTG format.",
        yields="single",
    )


@lore.task(
    key="assembly.spades",
    name="SPAdes Assembly",
    description="Assemble genomes from short reads using SPAdes.",
    category="Assembly",
    inputs=SpadesInputs,
    outputs=SpadesOutputs,
)
def spades_handler(
    ctx: lore.ExecutionContext,
    reads: list[lore.PathBundle],
    kmer_sizes: list[int] | None,
    error_correction: bool = False,
):
    """
    Run SPAdes assembly on the provided FASTQ files.
    """
    # 1. Validate config
    spades_config = ctx.get_config("spades")
    raw_path = str(spades_config.script_path).strip("\"'")

    spades_script = shutil.which(raw_path)
    if spades_script is None:
        raise RuntimeError(
            f"SPAdes script not found at '{raw_path}'. Either add it to PATH "
            f"or set spades_path to the full script location in Settings."
        )
    threads = spades_config.default_threads or 4
    memory = spades_config.default_memory or 16

    # Prepare output paths
    output_dir = ctx.get_temp_path("spades_output")
    contigs_path = Path(output_dir) / "contigs.fasta"
    scaffolds_path = Path(output_dir) / "scaffolds.fasta"
    gfa_path = Path(output_dir) / "assembly_graph_with_scaffolds.gfa"
    fastg_path = Path(output_dir) / "assembly_graph.fastg"

    # 2. Build SPAdes command
    cmd = [
        "python", spades_script,
        "-o", str(output_dir),
        "--threads", str(threads),
        "--memory", str(memory),
    ]

    # Format k-mers correctly: -k 21,33,55
    if kmer_sizes:
        k_str = ",".join(map(str, kmer_sizes))
        cmd.extend(["-k", k_str])

    # Skip error correction if not requested
    if not error_correction:
        cmd.append("--only-assembler")

    # 3. Flawless Bundle Routing
    lib_idx = 1
    for bundle in reads:
        if "paired" in bundle:
            # It's a paired-end bundle! Route R1 and R2
            cmd.extend([f"--pe{lib_idx}-1", str(bundle["main"])])
            cmd.extend([f"--pe{lib_idx}-2", str(bundle["paired"])])

            # If bam_to_fastq generated singletons, pass them alongside their library!
            if "singletons" in bundle:
                cmd.extend([f"--pe{lib_idx}-s", str(bundle["singletons"])])
        else:
            # It's an orphan/single-end bundle
            cmd.extend([f"--s{lib_idx}", str(bundle["main"])])

        lib_idx += 1

    # 4. Run SPAdes
    ctx.logger.info(f"Running SPAdes with command: {' '.join(cmd)}")
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
        raise RuntimeError(f"SPAdes assembly failed. Check the logs for details.")

    # 5. Materialize outputs
    output_map = {
        "contigs": contigs_path,
        "scaffolds": scaffolds_path,
        "gfa": gfa_path,
        "fastg": fastg_path,
    }

    for output_key, path in output_map.items():
        if path.exists() and path.stat().st_size > 0:
            ctx.logger.info(f"Materializing {output_key}: {path}")
            ctx.materialize_file(path, name=path.name, output_key=output_key, move=True)
        else:
            ctx.logger.warning(f"{output_key} file not found or empty: {path}")
