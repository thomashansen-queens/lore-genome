"""
Task for running MMseqs2 sequence search/clustering
"""
import subprocess
import tempfile
import shutil
from pathlib import Path
from lore.core.tasks import task_registry, Cardinality, Materialization, ArtifactInput, ValueInput, TaskOutput
from lore.core.executor import ExecutionContext


class Mmseqs2ClusterInputs:
    """Inputs for MMseqs2 clustering Task"""
    source_fasta = ArtifactInput(
        label="Protein FASTA",
        accepted_data=["protein_fasta", "fasta"],
        cardinality=Cardinality.SINGLE,
        load_as=Materialization.PATH,
    )
    min_seq_id = ValueInput(
        float,
        default=0.9,
        label="Minimum Sequence Identity",
        description="List matches above this sequence identity (0.0 to 1.0)",
    )
    coverage = ValueInput(
        float,
        default=0.8,
        label="Minimum Coverage",
        description="List matches above this fraction of aligned (covered) residues (0.0 to 1.0)",
    )
    clustered_region = ValueInput(
        str,
        default="",
        label="Clustered Region (optional)",
        description="Optional region to cluster on, in the format `start, end`. Leave empty for start/end e.g. (, 50) which clusters based on the first 50 residues. Negative numbers can be used to count backward from the C terminus e.g. (-50, -20), which will cluster by a 30-residue region starting 50 residues from the C terminus. If not provided, clusters will be based on full-length sequences (which is typical).",
        examples=["-800,"]
    )
    keep_representative_fasta = ValueInput(
        bool,
        default=False,
        label="Keep Representative FASTA",
        description="Save the (optionally truncated) representative sequences of each cluster as an Artifact.",
    )
    keep_truncated = ValueInput(
        bool,
        default=False,
        label="Keep Truncated FASTA",
        description="If a region was specified, save all truncated sequences as an Artifact."
    )


class Mmseqs2ClusterOutputs:
    """Outputs for MMseqs2 clustering Task"""
    cluster_tsv = TaskOutput(
        data_type="mmseqs2_cluster_map",
        label="Cluster Mapping",
        description="TSV file mapping all sequences to their representative cluster head.",
        is_primary=True,
    )
    representative_fasta = TaskOutput(
        data_type="protein_fasta",
        label="Representative Sequences",
        description="FASTA file containing only the representative sequence from each cluster.",
    )
    truncated_fasta = TaskOutput(
        data_type="protein_fasta",
        label="Truncated Sequences",
        description="The sliced FASTA used as input for clustering (if requested).",
    )


@task_registry.register(
    'mmseqs.easy_cluster',
    inputs=Mmseqs2ClusterInputs,
    outputs=Mmseqs2ClusterOutputs,
    name="MMseqs2 easy-cluster",
    category="Clustering",
    icon="🐈︎",
)
def mmseqs2_handler(
    ctx: ExecutionContext,
    source_fasta: str,
    min_seq_id: float = 0.9,
    coverage: float = 0.8,
    clustered_region: str = "",
    keep_truncated: bool = False,
    keep_representative_fasta: bool = False,
):
    """
    Run MMseqs2 on the given input FASTA file and output prefix.
    """
    # 1. Validate
    source_path = Path(source_fasta)
    mmseqs_bin = ctx.runtime.settings.mmseqs_path
    if not shutil.which(mmseqs_bin):
        raise RuntimeError(f"MMseqs2 binary not found at '{mmseqs_bin}'. Please check your settings.")

    # 2. Create temporary sandbox
    with tempfile.TemporaryDirectory() as sandbox_dir:
        sandbox = Path(sandbox_dir)
        target_fasta = source_path

        # 3. Parse clustered_region and make new FASTA in tmp dir if needed
        if clustered_region.strip():
            if "," not in clustered_region:
                ctx.logger.warning("Invalid format for clustered_region. Expected 'start,end'. Clustering will proceed on full-length sequences.")
                start = end = None
            # Extract start and end positions from clustered_region
            start, end = clustered_region.split(",")
            start = int(start.strip()) if start.strip() else None
            end = int(end.strip()) if end.strip() else None

            # Create a new FASTA file with the specified region
            target_fasta = sandbox / "clustered_region.fasta"
            with open(source_path, "r", encoding="utf-8") as f_in, open(target_fasta, "w", encoding="utf-8") as f_out:
                current_header = ""
                current_seq = []

                def _write_seq():
                    if not current_header or not current_seq:
                        return
                    seq_str = "".join(current_seq)
                    seq_len = len(seq_str)

                    s_idx = 0 if start is None else (start if start >= 0 else max(0, seq_len + start))
                    e_idx = seq_len if end is None else (end if end >= 0 else seq_len + end)

                    if e_idx > s_idx:
                        f_out.write(f"{current_header}\n{seq_str[s_idx:e_idx]}\n")

                for line in f_in:
                    line = line.strip()
                    if line.startswith(">"):
                        _write_seq()  # flush previous seq
                        current_header = line
                        current_seq = []
                    elif line:
                        current_seq.append(line)

                _write_seq()  # flush last seq

        # Define where the outputs and temp DBs should go
        out_prefix = sandbox / "cluster_res"
        tmp_dir = sandbox / "tmp"
        tmp_dir.mkdir()

        # 3. Build CLI command
        cmd = [
            mmseqs_bin, "easy-cluster",
            str(target_fasta),
            str(out_prefix),
            str(tmp_dir),
            "--min-seq-id", str(min_seq_id),
            "-c", str(coverage),
        ]
        ctx.logger.info("Running MMseqs2 with command: %s", " ".join(cmd))

        # 4. Execute and capture results
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            # Handle errors (e.g., log them, re-raise, etc.)
            ctx.logger.error("MMseqs2 failed: %s", e.stderr)
            raise RuntimeError(f"MMseqs2 failed with error: {e.stderr}") from e

        # 5. Harvest specific output before the sandbox is cleand up
        print("DEBUG: Checking for MMseqs2 output files...")
        print(list(sandbox.glob("*")))  # List all files in the sandbox for debugging
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

        if keep_truncated and clustered_region.strip():
            ctx.materialize_file(
                source_path=target_fasta,
                output_key="truncated_fasta",
                data_type="protein_fasta",
            )
