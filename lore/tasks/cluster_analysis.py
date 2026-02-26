"""
Synthesizing and anaylzing protein clusters
"""
import pandas as pd

from lore.adapters import adapter_registry
from lore.core.tasks import ArtifactInput, TaskOutput, ValueInput, task_registry, Cardinality, Materialization
from lore.core.executor import ExecutionContext
from lore.utils.parse import fasta_lookup

def _load_and_merge_cluster_data(
    cluster_map_path: str,
    genome_annotations_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper function to do the big load and merge of the cluster map and genome annotations
    """
    cluster_df = pd.read_csv(
        cluster_map_path,
        sep="\t",
        header=None,
        names=["mmseqs_cluster_id", "protein_accession"]
    )
    # NOTE: annotations_df is an adapted table! We need to use the adapter to 
    # get a list of dicts, then convert it to a dataframe
    # (could be very large, <= 500 MB)
    adapter = adapter_registry["NcbiGenomeAnnotationsAdapter"]
    annotations_df = adapter.to_dataframe(genome_annotations_path)
    annotations_df[["begin", "end", "protein_length"]] = (
        annotations_df[["begin", "end", "protein_length"]].astype("Int64")
    )

    # Count the prevalence of each protein accession across the annotations
    protein_counts = annotations_df["protein_accession"].value_counts().reset_index()
    protein_counts.columns = ["protein_accession", "occurrence_count"]

    cluster_df = cluster_df.merge(
        protein_counts,
        on="protein_accession",
        how="left",
    ).fillna({"occurrence_count": 1})

    # Intrinsic protein metadata
    metadata_lookup = annotations_df[
        ["protein_accession", "name", "symbol", "protein_length"]
    ].drop_duplicates(subset=["protein_accession"])

    cluster_df = cluster_df.merge(
        metadata_lookup,
        on="protein_accession",
        how="left",
    )

    return cluster_df, annotations_df

class BaseClusterInputs:
    """Inputs for summarizing the origins of protein clusters"""
    cluster_map = ArtifactInput(
        label="MMSeqs2 cluster map",
        accepted_data=["mmseqs2_cluster_map"],
        cardinality=Cardinality.SINGLE,
        load_as=Materialization.PATH,
    )
    genome_annotations = ArtifactInput(
        label="Genome annotations",
        accepted_data=["ncbi_annotation_packages"],
        cardinality=Cardinality.SINGLE,
        load_as=Materialization.PATH,
    )
    protein_fasta = ArtifactInput(
        label="Clustered protein sequences",
        accepted_data=["protein_fasta"],
        cardinality=Cardinality.OPTIONAL,
        load_as=Materialization.PATH,
    )

# --- Summarize cluster origins ---

class SummarizeClusterOriginsInputs(BaseClusterInputs):
    """Inputs for summarizing the origins of protein clusters"""
    pass


class SummarizeClusterOriginsOutputs:
    """Outputs for summarizing the origins of protein clusters"""
    clustered_summary = TaskOutput(
        data_type="clustered_summary",
        label="Cluster origins summary",
        description="A tabular report detailing how many and which genomes contribute to each cluster.",
        is_primary=True,
    )


@task_registry.register(
    "analysis.summarize_cluster_origins",
    name="Summarize Cluster Origins",
    inputs=SummarizeClusterOriginsInputs,
    outputs=SummarizeClusterOriginsOutputs,
    category="analysis",
    icon="🖩",
)
def summarize_cluster_origins(
    ctx: ExecutionContext,
    cluster_map: str,
    genome_annotations: str,
    protein_fasta: str | None = None,
):
    """
    Summarize the origins of protein clusters by counting how many and which genomes contribute to each cluster.
    """
    # 1. Load, merge, map
    cluster_df, annotations_df = _load_and_merge_cluster_data(cluster_map, genome_annotations)

    # 2. Find the "True" Representative (most frequent member in each cluster)
    cluster_df = cluster_df.sort_values(by=["mmseqs_cluster_id", "occurrence_count"], ascending=[True, False])
    best_reps_df = cluster_df.groupby("mmseqs_cluster_id").agg(
        best_representative=pd.NamedAgg(column="protein_accession", aggfunc="first"),
        cluster_name=pd.NamedAgg(column="name", aggfunc="first"),
        cluster_symbol=pd.NamedAgg(column="symbol", aggfunc="first"),
        protein_occurences=pd.NamedAgg(column="occurrence_count", aggfunc="sum"),
        cluster_size=pd.NamedAgg(column="protein_accession", aggfunc="count"),
        cluster_members=pd.NamedAgg(column="protein_accession", aggfunc=lambda x: ";".join(x.unique())),
    ).reset_index().copy()

    # 3. Group by representative sequence and summarize the contributing genomes
    cluster_df = cluster_df.merge(
        annotations_df[["protein_accession", "genome_accession"]],
        on="protein_accession",
    )

    cluster_df = cluster_df.groupby("mmseqs_cluster_id").agg(
        num_genomes=pd.NamedAgg(column="genome_accession", aggfunc="nunique"),
        genomes=pd.NamedAgg(column="genome_accession", aggfunc=lambda x: ";".join(sorted(x.unique()))),
        min_protein_length=pd.NamedAgg(column="protein_length", aggfunc="min"),
        max_protein_length=pd.NamedAgg(column="protein_length", aggfunc="max"),
        mean_protein_length=pd.NamedAgg(column="protein_length", aggfunc="mean"),
    ).reset_index()
    cluster_df["mean_protein_length"] = cluster_df["mean_protein_length"].round(1)
    cluster_df[["min_protein_length", "max_protein_length"]] = cluster_df[["min_protein_length", "max_protein_length"]].astype("Int64")

    # Final join: Combine our metadata summary with the genome counts
    final_summary_df = best_reps_df.merge(
        cluster_df,
        on="mmseqs_cluster_id",
    )

    # 5. RAM-safe plucking of representative sequences from the original FASTA
    if protein_fasta:
        ctx.logger.info("Extracting representative sequences from FASTA...")
        valid_accs = set(final_summary_df["best_representative"])
        extracted_seqs = fasta_lookup(
            targets=list(valid_accs),
            fasta_path=protein_fasta,
        )

        # Convert to a DataFrame and merge
        seq_df = pd.DataFrame(list(extracted_seqs.items()), columns=["best_representative", "representative_sequence"])
        final_summary_df = final_summary_df.merge(seq_df, on="best_representative", how="left")

    # 4. Reorder the columns
    final_cols = ["mmseqs_cluster_id", "cluster_name", "cluster_symbol",
        "best_representative", "min_protein_length", "max_protein_length",
        "mean_protein_length", "cluster_size", "protein_occurences",
        "num_genomes", "cluster_members", "genomes"]
    if protein_fasta:
        final_cols.append("representative_sequence")

    final_summary_df = final_summary_df[final_cols]

    out_path = ctx.get_temp_path("cluster_origins_summary.csv")
    final_summary_df.to_csv(out_path, sep=",", index=False)

    ctx.materialize_file(
        output_key="clustered_summary",
        source_path=out_path,
    )

# --- Individual cluster report ---

class InspectClusterInputs(BaseClusterInputs):
    """Inputs for inspecting an individual cluster"""
    protein_accession = ArtifactInput(
        description="Generate a report showing all proteins co-clustered with the input protein(s)",
        label="Protein accession",
        accepted_data=["protein_accession"],
        cardinality=Cardinality.ONE_OR_MORE,
        load_as=Materialization.CONTENT,
        examples=["WP_012345678.1"],
    )
    save_fasta = ValueInput(
        bool,
        description="Whether to write the sequences of the cluster members to a new FASTA file. This is not very useful for LoRē thanks to the semantic typing system, but maybe you want to download the FASTA for use elsewhere?",
        default=True,
        label="Write cluster FASTA",
    )


class InspectClusterOutputs:
    """Outputs for inspecting an individual cluster"""
    cluster_report = TaskOutput(
        data_type="genome_annotations",
        label="Cluster report",
        description="A detailed report on the composition of a single protein cluster.",
        is_primary=True,
    )
    cluster_fasta = TaskOutput(
        data_type="cluster_fasta",
        label="Cluster FASTA",
        description="A FASTA file containing the sequences of all proteins in the cluster.",
    )


@task_registry.register(
    "analysis.inspect_cluster",
    name="Inspect Cluster",
    inputs=InspectClusterInputs,
    outputs=InspectClusterOutputs,
    category="analysis",
    icon="🗏",
)
def inspect_cluster(
    ctx: ExecutionContext,
    protein_accession: list[str],
    cluster_map: str,
    genome_annotations: str,
    save_fasta: bool,
    protein_fasta: str | None = None,
):
    """
    Generate a detailed report on the composition of a single protein cluster, including its member proteins and the genomes they come from.
    """
    # 1. Load, merge, map
    cluster_df, annotations_df = _load_and_merge_cluster_data(cluster_map, genome_annotations)
    if cluster_df.empty:
        raise ValueError(f"No clusters found in cluster map {cluster_map}")

    # 2. Find the cluster(s) that contain any of the input protein accessions
    cluster_ids = cluster_df[cluster_df["protein_accession"].isin(protein_accession)]["mmseqs_cluster_id"].unique()
    if len(cluster_ids) == 0:
        raise ValueError(f"No clusters found containing any of the input protein accessions: {protein_accession}")

    cluster_df = cluster_df[cluster_df["mmseqs_cluster_id"].isin(cluster_ids)]
    cluster_df = cluster_df.merge(
        annotations_df[["protein_accession", "genome_accession", "locus_tag", "begin", "end", "orientation"]],
        on="protein_accession",
        how="left",
    )

    # 3. RAM-safe plucking of representative sequences from the original FASTA
    if protein_fasta:
        ctx.logger.info("Extracting representative sequences from FASTA...")
        valid_accs = set(cluster_df["protein_accession"])
        extracted_seqs = fasta_lookup(
            targets=list(valid_accs),
            fasta_path=protein_fasta,
        )

        if save_fasta:
            fasta_path = ctx.get_temp_path("cluster_members.fasta")
            with open(fasta_path, "w", encoding="utf-8") as f:
                for head, seq in extracted_seqs.items():
                    f.write(f">{head}\n{seq}\n")
            ctx.materialize_file(
                output_key="cluster_fasta",
                source_path=fasta_path,
                metadata={
                    "description": "Cluster(s) containing " + ", ".join(protein_accession),
                }
            )

        # Convert to a DataFrame and merge
        seq_df = pd.DataFrame(list(extracted_seqs.items()), columns=["protein_accession", "representative_sequence"])
        cluster_df = cluster_df.merge(seq_df, on="protein_accession", how="left")

    # 4. Tidy up the DataFrame
    cluster_df = cluster_df.sort_values(by=["mmseqs_cluster_id", "genome_accession", "protein_accession"])
    out_path = ctx.get_temp_path("cluster_report.csv")
    cluster_df.to_csv(out_path, sep=",", index=False)

    ctx.materialize_file(
        name=str(protein_accession[0]) + "_cluster",
        output_key="cluster_report",
        source_path=out_path,
    )
