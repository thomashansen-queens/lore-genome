"""
Helper task for gene sequene from protein workflow pipeline
"""
from collections.abc import Iterator
from enum import StrEnum
import lore
import pandas as pd

class Inputs:
    ipg_mapping_table = lore.ArtifactInput(
        accepted_data=["ipg_records"],
        select="single",
        load_as="adapted_stream",
        label="IPG Mapping Table",
        description="Tabular artifact containing the mapping of protein accessions to nucleotide accessions.",
    )

    nucleotide_fasta_table = lore.ArtifactInput(
        accepted_data=["nucleotide_fasta"],
        select="single",
        load_as="adapted_stream",
        label="Nucleotide FASTA Table",
        description="Tabular artifact containing nucleotide accessions and their corresponding FASTA sequences.",
    )

    drop_duplicates = lore.ValueInput(
        bool,
        default=True,
        label="Drop Duplicates",
        description="Whether to drop rows with duplicate nucleotide accessions in the output.",
    )

class Outputs:
    joined_fasta = lore.TaskOutput(
        data_type="nucleotide_fasta",  # inherits the data type of the left table
        label="Joined FASTA Table",
        is_primary=True,
    )

@lore.task(
    "aleyssu.gene_from_protein_workflow.join_tables",
    inputs=Inputs,
    outputs=Outputs,
    name="Join IPG Mapping and Nucleotide FASTA Tables",
    category="Data Utilities",
    preview_mode="full",
    icon="⏵⏴",
)
def join_tables(
    ctx: lore.ExecutionContext,
    ipg_mapping_table: Iterator[dict],
    nucleotide_fasta_table: Iterator[dict],
    drop_duplicates: bool = True,
):
    # 1. Load the input tables into DataFrames
    df_ipg = pd.DataFrame(ipg_mapping_table)
    df_fasta = pd.DataFrame(nucleotide_fasta_table)

    # 2. Drop duplicates in the ipg_mapping_table if requested
    if drop_duplicates:
        df_ipg = df_ipg.drop_duplicates(subset=["Nucleotide Accession"], keep="first")
        df_ipg["Assembly"] = df_ipg["Assembly"].str.replace("GCA_|GCF_", "", regex=True)
        # Drop rows with identical assembly numbers
        df_ipg = df_ipg.drop_duplicates(subset=["Assembly"], keep="first")

    # 3. Perform the join operation
    try:
        df_joined = pd.merge(
            df_ipg[["Nucleotide Accession", "Protein", "Start", "Stop"]],
            df_fasta[["nucleotide_accession", "nucleotide_sequence"]],
            how="inner",
            left_on="Nucleotide Accession",
            right_on="nucleotide_accession",
        ).drop(columns=["nucleotide_accession"])
    except Exception as e:
        ctx.logger.error(f"Error during join operation: {e}")
        raise RuntimeError(f"Error during join operation: {e}") from e

    # 4. Materialize the output table as a new artifact
    out_path = ctx.get_temp_path("nucleotide_fasta.faa")

    with open(out_path, "w") as f:
        for _, row in df_joined.iterrows():
            start = row['Start']
            stop = row['Stop']
            f.write(f">{row['Nucleotide Accession']} {row['Protein']} ({start}-{stop})\n{row['nucleotide_sequence'][int(start)-1:int(stop)]}\n")

    ctx.materialize_file(
        source=out_path,
        output_key="joined_fasta",
    )
