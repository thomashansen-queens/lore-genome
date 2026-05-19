"""
Task for running MMseqs2 sequence search/clustering
"""
import subprocess
import shutil
from pathlib import Path

from lore.builtins.adapters.fasta import FastaAdapter
import lore.dsl as lore


@lore.config(key="mmseqs2", title="MMseqs2 suite")
class MmseqsConfig:
    """Global settings for the MMMseqs2 clustering suite."""
    binary_path = lore.ValueInput(
        str,
        default="mmseqs",
        label="Path to MMseqs2 binary",
        description="Provide the full path if not in your system PATH.",
    )
    default_threads = lore.ValueInput(
        int,
        default=4,
        min=1, max=32, step=1,
        label="Default CPU Threads",
        description="Global default for all MMseqs2 tasks.",
        widget="slider",
    )


class Mmseqs2ClusterInputs:
    """Inputs for MMseqs2 clustering Task"""
    source_fasta = lore.ArtifactInput(
        label="Protein FASTA",
        accepted_data=["protein_fasta", "fasta"],
        select=lore.SINGLE,
        load_as=lore.PATH,
    )
    min_seq_id = lore.ValueInput(
        float,
        default=0.9, min=0.0, max=1.0, step=0.01,
        label="Minimum sequence identity",
        description="List matches above this sequence identity (0.0 to 1.0)",
        widget="slider",
    )
    coverage = lore.ValueInput(
        float,
        default=0.8, min=0.0, max=1.0, step=0.01,
        label="Minimum coverage",
        description="List matches above this fraction of aligned (covered) residues (0.0 to 1.0)",
        widget="slider",
    )
    sensitivity = lore.ValueInput(
        float,
        default=4.0, min=1.0, max=8.0, step=0.5,
        label="Sensitivity",
        description="1.0 is fast but less sensitive. 7.5 is highly sensitive for distant homologs but slower.",
        examples=[4.0],
        widget="slider",
    )
    cluster_window = lore.ValueInput(
        str | None,
        default="",
        label="Cluster Window (optional)",
        description="Optional window to cluster on, in the format `start, end`. Leave empty for start/end e.g. (, 50) which clusters based on the first 50 residues. Negative numbers can be used to count backward from the C terminus e.g. (-50, -20), which will cluster by a 30-residue region starting 50 residues from the C terminus. If not provided, clusters will be based on full-length sequences (which is typical).",
        examples=["-800,"],
    )
    strict_window = lore.ValueInput(
        bool,
        default=False,
        label="Strict Window",
        description="If a region was specified, only include sequences that fully span the requested window.",
    )
    keep_representative_fasta = lore.ValueInput(
        bool,
        default=False,
        label="Keep representative FASTA",
        description="Save the representative sequences of each cluster (with optional truncation) as an Artifact.",
    )
    keep_truncated = lore.ValueInput(
        bool,
        default=False,
        label="Keep truncated FASTA",
        description="If a region was specified, save all truncated sequences as an Artifact.",
    )


class Mmseqs2ClusterOutputs:
    """Outputs for MMseqs2 clustering Task"""
    cluster_tsv = lore.TaskOutput(
        data_type="mmseqs2_cluster_map",
        label="Cluster Mapping",
        description="TSV file mapping all sequences to their representative cluster head.",
        is_primary=True,
    )
    representative_fasta = lore.TaskOutput(
        data_type="protein_fasta",
        label="Representative Sequences",
        description="FASTA file containing only the representative sequence from each cluster.",
        yields=lore.OPTIONAL,
    )
    truncated_fasta = lore.TaskOutput(
        data_type="protein_fasta",
        label="Truncated Sequences",
        description="The sliced FASTA used as input for clustering (if requested).",
        yields=lore.OPTIONAL,
    )

# --- Helpers ---

def _parse_window(cluster_window: str) -> tuple[int | None, int | None]:
    """
    Parse the cluster_window string into start and end integers.
    - "start, end" -> (start, end)
    - "1000" -> (None, 1000) (first 1000 residues)
    - "-50" -> (-50, None) (last 50 residues)
    """
    if not cluster_window or not cluster_window.strip():
        return None, None

    if "," in cluster_window:
        start_str, end_str = cluster_window.split(",")
        end = int(end_str.strip()) if end_str.strip() else None
        start = int(start_str.strip()) if start_str.strip() else None
        return start, end

    try:
        truncate_int = int(cluster_window.strip())
        if truncate_int >= 0:
            return None, truncate_int
        else:
            return truncate_int, None

    except ValueError:
        raise ValueError(
            f"Cluster window '{cluster_window}' is not a valid format. "
            f"Use 'start, end' or a single integer."
        )


def _sliced_fasta(
    ctx: lore.ExecutionContext,
    source_path: Path,
    cluster_window: str,
    strict_window: bool,
    sandbox_dir: Path,
) -> Path:
    """
    Create a sliced FASTA file based on a specified window.
    """
    start, end = _parse_window(cluster_window)
    tmp_out = sandbox_dir / "sliced.fasta"
    adapter = FastaAdapter()

    with (
        open(source_path, "r", encoding="utf-8") as f_in,
        open(tmp_out, "w", encoding="utf-8", newline="\n") as f_out
    ):
        kept_count = 0
        dropped_count = 0
        for record in adapter.adapt_stream(f_in):

            seq = record["sequence"]
            sliced = seq[start:end]
            if not sliced:
                dropped_count += 1
                continue

            if strict_window:
                # Positive start out of bounds
                if start is not None and abs(start) > len(seq):
                    dropped_count += 1
                    continue
                # Negative start out of bounds
                if end is not None and abs(end) > len(seq):
                    dropped_count += 1
                    continue

            name_part = f" {record['name']}" if record.get("name") else ""
            f_out.write(f">{record['accession']}{name_part}\n{sliced}\n")
            kept_count += 1

    ctx.logger.info(f"Sliced FASTA created with {kept_count} sequences kept and {dropped_count} sequences dropped based on the specified window.")
    return tmp_out

# --- Main Task ---

@lore.task(
    'mmseqs.easy_cluster',
    inputs=Mmseqs2ClusterInputs,
    outputs=Mmseqs2ClusterOutputs,
    name="MMseqs2 easy-cluster",
    category="Clustering",
    icon="🐈︎",
)
def mmseqs2_handler(
    ctx: lore.ExecutionContext,
    source_fasta: str,
    min_seq_id: float = 0.9,
    coverage: float = 0.8,
    sensitivity: float = 4.0,
    cluster_window: str | None = None,
    strict_window: bool = False,
    keep_truncated: bool = False,
    keep_representative_fasta: bool = False,
):
    """
    Run MMSeqs2 on the given input FASTA file. MMSeqs2 is an ultra-fast and 
    sensitive sequence search and clustering suite. This task uses the 
    'easy-cluster' workflow.
    """
    # 1. Validate
    mmseqs_config = ctx.get_config("mmseqs2")
    source_path = Path(source_fasta)
    raw_path = str(mmseqs_config.binary_path).strip("\"'")

    mmseqs_bin = shutil.which(raw_path)
    if mmseqs_bin is None:
        raise RuntimeError(
            f"MMseqs2 binary not found at '{raw_path}'. Either add it to PATH "
            f"or set mmseqs_path to the full binary location in Settings."
        )

    # 2. Set up auto-cleaning sandbox for intermediate files and DBs
    sandbox = ctx.get_temp_path(filename="mmseqs_sandbox")
    sandbox.mkdir(parents=True, exist_ok=True)
    out_prefix = sandbox / "cluster_res"
    tmp_dir = sandbox / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    # 3. Parse cluster_window and make new FASTA in tmp dir if needed
    if cluster_window and cluster_window.strip():
        target_fasta = _sliced_fasta(ctx, source_path, cluster_window, strict_window, sandbox)
    else:
        target_fasta = sandbox / "input.fasta"
        shutil.copy(source_path, target_fasta)

    # 4. Build CLI command
    cmd = [
        mmseqs_bin, "easy-cluster",
        target_fasta.name,
        "cluster_res",  # automatic prefix in sandbox
        "tmp",          # automatic prefix in sandbox
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "-s", str(sensitivity),
        "--threads", str(mmseqs_config.default_threads),
    ]
    ctx.logger.info("Running MMseqs2 with command: %s", " ".join(cmd))

    # 5. Execute and capture results
    try:
        subprocess.run(
            cmd,
            cwd=sandbox,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        # Handle errors (e.g., log them, re-raise, etc.)
        ctx.logger.error("MMseqs2 failed: %s", e.stderr)
        raise RuntimeError(f"MMseqs2 failed with error: {e.stderr}") from e

    # 6. Harvest specific output before the sandbox is cleaned up
    rep_path = Path(f"{out_prefix}_rep_seq.fasta")
    tsv_path = Path(f"{out_prefix}_cluster.tsv")

    if not rep_path.exists() or not tsv_path.exists():
        raise RuntimeError("Expected MMseqs2 output files not found.")

    ctx.materialize_file(
        source_path=tsv_path,
        output_key="cluster_tsv",
        data_type="mmseqs2_cluster_map",
    )

    if keep_representative_fasta:
        ctx.materialize_file(
            source_path=rep_path,
            output_key="representative_fasta",
            data_type="protein_fasta",
        )

    if keep_truncated and cluster_window and cluster_window.strip():
        ctx.materialize_file(
            source_path=target_fasta,
            output_key="truncated_fasta",
            data_type="protein_fasta",
        )
