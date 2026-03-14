"""
Synthesizing and anaylzing protein clusters
"""
import pandas as pd

# from lore.core.adapters import adapter_registry, TableAdapter
import lore.dsl as lore
from lore.utils.parse import fasta_lookup

# --- Helpers ----

def _load_and_merge_cluster_data(
    ctx: lore.ExecutionContext,
    cluster_map: list[dict],
    genome_annotations: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merges streams for genome annotations and cluster maps
    """
    # 1. Tabulate records
    cluster_df = pd.DataFrame(cluster_map)
    annotations_df = pd.DataFrame(genome_annotations)

    # 2. Data hygiene: Ensure consistent column names and types
    clustered_accs = set(cluster_df["protein_accession"])
    annotated_accs = set(annotations_df["protein_accession"])

    missing_annotations = clustered_accs - annotated_accs
    if missing_annotations:
        ctx.logger.warning(
            "%s clustered proteins have no matching annotation records!",
            len(missing_annotations),
        )

    cluster_df = cluster_df.rename(columns={"cluster_rep": "mmseqs_cluster_id"})
    annotations_df[["begin", "end", "protein_length"]] = annotations_df[["begin", "end", "protein_length"]].astype("Int64")

    # 3. Count the prevalence of each protein accession across the annotations
    protein_counts = annotations_df["protein_accession"].value_counts().reset_index()
    protein_counts.columns = ["protein_accession", "occurrence_count"]

    cluster_df = cluster_df.merge(
        protein_counts,
        on="protein_accession",
        how="left",
    ).fillna({"occurrence_count": 1})

    # 4. Intrinsic protein metadata
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
    cluster_map = lore.ArtifactInput(
        label="MMSeqs2 cluster map",
        accepted_data=["mmseqs2_cluster_map"],
        select=lore.SINGLE,
        load_as=lore.ADAPTED,
    )
    genome_annotations = lore.ArtifactInput(
        label="Genome annotations",
        accepted_data=["ncbi_annotation_packages"],
        select=lore.MULTIPLE,
        load_as=lore.ADAPTED,
    )
    protein_fasta = lore.ArtifactInput(
        label="Clustered protein sequences",
        accepted_data=["protein_fasta"],
        select=lore.OPTIONAL,
        load_as=lore.PATH,
    )

# --- Summarize cluster origins ---

class SummarizeClusterOriginsInputs(BaseClusterInputs):
    """Inputs for summarizing the origins of protein clusters"""
    pass


class SummarizeClusterOriginsOutputs:
    """Outputs for summarizing the origins of protein clusters"""
    clustered_summary = lore.TaskOutput(
        data_type="clustered_summary",
        label="Cluster origins summary",
        description="A tabular report detailing how many and which genomes contribute to each cluster.",
        is_primary=True,
    )


@lore.task(
    "analysis.summarize_cluster_origins",
    name="Summarize cluster origins",
    inputs=SummarizeClusterOriginsInputs,
    outputs=SummarizeClusterOriginsOutputs,
    category="Clustering",
    icon="🖩",
    live_preview=True,
)
def summarize_cluster_origins(
    ctx: lore.ExecutionContext,
    cluster_map: list[dict],
    genome_annotations: list[dict],
    protein_fasta: str | None = None,
):
    """
    Summarize the origins of protein clusters by counting how many and which genomes contribute to each cluster.
    """
    # 1. Load, merge, map
    cluster_df, annotations_df = _load_and_merge_cluster_data(ctx, cluster_map, genome_annotations)

    # 2. Find the "True" Representative (most frequent member in each cluster)
    cluster_df = cluster_df.sort_values(by=["mmseqs_cluster_id", "occurrence_count"], ascending=[True, False])
    best_reps_df = cluster_df.groupby("mmseqs_cluster_id").agg(
        best_representative=pd.NamedAgg(column="protein_accession", aggfunc="first"),
        cluster_name=pd.NamedAgg(column="name", aggfunc="first"),
        cluster_symbol=pd.NamedAgg(column="symbol", aggfunc="first"),
        protein_occurences=pd.NamedAgg(column="occurrence_count", aggfunc="sum"),
        cluster_size=pd.NamedAgg(column="protein_accession", aggfunc="count"),
        cluster_members=pd.NamedAgg(column="protein_accession", aggfunc=lambda x: ",".join(x.dropna().astype(str).unique())),
    ).reset_index().copy()
    ctx.logger.debug("Found %s clusters with best representatives", len(best_reps_df))

    # 3. Group by representative sequence and summarize the contributing genomes
    cluster_df = cluster_df.merge(
        annotations_df[["protein_accession", "genome_accession"]],
        on="protein_accession",
    )

    cluster_df = cluster_df.groupby("mmseqs_cluster_id").agg(
        num_genomes=pd.NamedAgg(column="genome_accession", aggfunc="nunique"),
        genomes=pd.NamedAgg(column="genome_accession", aggfunc=lambda x: ",".join(sorted(x.dropna().astype(str).unique()))),
        min_protein_length=pd.NamedAgg(column="protein_length", aggfunc="min"),
        max_protein_length=pd.NamedAgg(column="protein_length", aggfunc="max"),
        mean_protein_length=pd.NamedAgg(column="protein_length", aggfunc="mean"),
    ).reset_index()
    cluster_df["mean_protein_length"] = cluster_df["mean_protein_length"].round(1)
    cluster_df[["min_protein_length", "max_protein_length"]] = cluster_df[["min_protein_length", "max_protein_length"]].astype("Int64")

    ctx.logger.debug("Found %s clusters with genome information", len(cluster_df))

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
        seq_df = pd.DataFrame(list(extracted_seqs.items()), columns=["best_representative", "protein_sequence"])
        final_summary_df = final_summary_df.merge(seq_df, on="best_representative", how="left")

    # 4. Reorder the columns
    final_cols = ["mmseqs_cluster_id", "cluster_name", "cluster_symbol",
        "best_representative", "min_protein_length", "max_protein_length",
        "mean_protein_length", "cluster_size", "protein_occurences",
        "num_genomes", "cluster_members", "genomes"]
    if protein_fasta:
        final_cols.append("protein_sequence")

    final_summary_df = final_summary_df[final_cols]

    out_path = ctx.get_temp_path("cluster_origins_summary.csv")
    final_summary_df.to_csv(out_path, sep=",", index=False)

    ctx.materialize_file(
        output_key="clustered_summary",
        source_path=out_path,
        metadata={
            "columns": final_summary_df.columns.tolist(),
        }
    )

# --- Individual cluster report ---

class InspectClusterInputs(BaseClusterInputs):
    """Inputs for inspecting an individual cluster"""
    protein_accession = lore.ArtifactInput(
        description="Generate a report showing all proteins co-clustered with the input protein(s)",
        label="Protein accession",
        accepted_data=["protein_accession"],
        select=lore.MULTIPLE,
        load_as=lore.ADAPTED,
        examples=["WP_012345678.1"],
    )
    save_fasta = lore.ValueInput(
        bool,
        description="Whether to write the sequences of the cluster members to a new FASTA file. This is not very useful for LoRē thanks to the semantic typing system, but maybe you want to download the FASTA for use elsewhere?",
        default=True,
        label="Write cluster FASTA",
    )


class InspectClusterOutputs:
    """Outputs for inspecting an individual cluster"""
    cluster_report = lore.TaskOutput(
        data_type="genome_annotations",
        label="Cluster report",
        description="A detailed report on the composition of a single protein cluster.",
        is_primary=True,
    )
    cluster_fasta = lore.TaskOutput(
        data_type="protein_fasta",
        label="Cluster FASTA",
        description="A FASTA file containing the sequences of all proteins in the cluster.",
    )


@lore.task(
    "analysis.inspect_cluster",
    name="Inspect cluster",
    inputs=InspectClusterInputs,
    outputs=InspectClusterOutputs,
    category="Clustering",
    icon="🗏",
    live_preview=True,
)
def inspect_cluster(
    ctx: lore.ExecutionContext,
    protein_accession: list[str],
    cluster_map: list[dict],
    genome_annotations: list[dict],
    save_fasta: bool = True,
    protein_fasta: str | None = None,
):
    """
    Generate a detailed report on the composition of a single protein cluster, including its member proteins and the genomes they come from.
    """
    # 1. Load, merge, map
    cluster_df, annotations_df = _load_and_merge_cluster_data(
        ctx=ctx,
        cluster_map=cluster_map,
        genome_annotations=genome_annotations,
    )

    if cluster_df.empty:
        raise ValueError(f"No clusters found in cluster map {cluster_map}")

    # 2. Find the cluster(s) that contain any of the input protein accessions
    cluster_ids = cluster_df[cluster_df["protein_accession"].isin(protein_accession)]["mmseqs_cluster_id"].unique()
    if len(cluster_ids) == 0:
        raise ValueError(f"No clusters found containing any of the input protein accessions: {protein_accession}")

    cluster_df = cluster_df[cluster_df["mmseqs_cluster_id"].isin(cluster_ids)]
    cluster_df = cluster_df.merge(
        annotations_df[["protein_accession", "genome_accession", "locus_tag", "begin", "end", "orientation", "chromosome"]],
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
                    "source_accessions": ", ".join(protein_accession),
                }
            )

        # Convert to a DataFrame and merge
        seq_df = pd.DataFrame(list(extracted_seqs.items()), columns=["protein_accession", "protein_sequence"])
        cluster_df = cluster_df.merge(seq_df, on="protein_accession", how="left")
    elif save_fasta:
        ctx.logger.warning("Cannot save FASTA for cluster members because no source FASTA was provided. Skipping...")

    # 4. Tidy up the DataFrame
    cluster_df = cluster_df.sort_values(by=["mmseqs_cluster_id", "genome_accession", "protein_accession"])
    out_path = ctx.get_temp_path("cluster_report.csv")
    cluster_df.to_csv(out_path, sep=",", index=False)

    ctx.materialize_file(
        name=str(protein_accession[0]) + "_cluster",
        output_key="cluster_report",
        source_path=out_path,
        metadata={
            "columns": cluster_df.columns.tolist(),
            "source_accessions": ", ".join(protein_accession),
        }
    )
