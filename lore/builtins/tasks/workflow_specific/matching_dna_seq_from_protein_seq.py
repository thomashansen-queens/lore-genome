"""
Helper task for gene sequene from protein workflow pipeline
"""
from collections.abc import Iterator
import lore
import pandas as pd
from Bio import Align

class Inputs:
    nucleotide_fasta = lore.ArtifactInput(
        accepted_data=["nucleotide_fasta"],
        select="single",
        load_as="adapted_stream",
        label="Nucleotide Sequence Fasta",
        description="Fasta of all the nucleotide sequences to scan",
    )

    compare_seq = lore.ValueInput(
        str,
        label="Protein Sequence",
        description="The amino acid sequence of the protein to match the nucleotide sequences to.",
    )

    condense_header = lore.ValueInput(
        bool,
        default=True,
        label="Condense Headers",
        description="Condenses the headers of the final fasta.",
    )

class Outputs:
    fasta = lore.TaskOutput(
        data_type="nucleotide_fasta",
        label="Nucleotide FASTA",
        is_primary=True,
    )

@lore.task(
    "aleyssu.gene_from_protein_workflow.match_dna_protein",
    inputs=Inputs,
    outputs=Outputs,
    name="Match DNA Sequences to Protein Subsequence",
    category="Workflow-Specific",
    preview_mode="full",
    icon="⏵⏴",
)
def task(
    ctx: lore.ExecutionContext,
    nucleotide_fasta: Iterator[dict],
    condense_header: bool = True,
):
    headers = []
    sequences = []
    
    for entry in nucleotide_fasta:
        entry['nucleotide_accession']

    # with open(out_path, "w") as f:
    #     for _, row in df_joined.iterrows():
    #         start = row['Start']
    #         stop = row['Stop']
    #         f.write(f">{row['Nucleotide Accession']} {row['Protein']} ({start}-{stop})\n{row['nucleotide_sequence'][int(start)-1:int(stop)]}\n")

    # ctx.materialize_file(
    #     source=out_path,
    #     output_key="fasta",
    # )
