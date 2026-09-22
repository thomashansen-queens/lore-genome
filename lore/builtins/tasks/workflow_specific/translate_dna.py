"""
Helper task for gene sequene from protein workflow pipeline
"""
from collections.abc import Iterator
import lore
from Bio import Seq

class Inputs:
    nucleotide_fasta = lore.ArtifactInput(
        accepted_data=["nucleotide_match_fasta"],
        select="single",
        load_as="adapted_stream",
        label="Nucleotide Sequence Fasta",
        description="Fasta of all the nucleotide sequences to translate",
    )

class Outputs:
    fasta = lore.TaskOutput(
        data_type="protein_fasta",
        label="Protein FASTA",
        is_primary=True,
    )

    cluster_table = lore.TaskOutput(
        data_type="csv",
        label="Translated Protein Cluster Table",
        is_primary=False,
    )

@lore.task(
    "aleyssu.gene_from_protein_workflow.translate",
    inputs=Inputs,
    outputs=Outputs,
    name="Translate DNA to Protein",
    category="Workflow-Specific",
    preview_mode="full",
    icon="",
)
def task(
    ctx: lore.ExecutionContext,
    nucleotide_fasta: Iterator[dict],
):
    """Uses Biopython's translate tool to translate a fasta containing coding nucleotide sequences in the correct reading frame."""
    sequences = dict()

    for entry in nucleotide_fasta:
        accession = entry["representative_nucleotide_accession"]
        desc = entry["description"]
        sequence = Seq.translate(entry['nucleotide_sequence'])
        sequence = sequence[:-1] if sequence[-1] == "*" else sequence  # Get rid of stop codon

        if sequence not in sequences.keys():
            sequences[sequence] = [[], 0]
        cluster_size = entry["cluster_size"]

        sequences[sequence][0].append([accession, desc, cluster_size])
        sequences[sequence][1] += cluster_size

    out_path = ctx.get_temp_path("protein_fasta.faa")
    out_path_csv = ctx.get_temp_path("cluster_table.csv")

    with open(out_path_csv, "w") as f_csv:
        f_csv.write("representative_nucleotide_accession,nucleotide_accession,description,cluster_size_contribution\n")
        with open(out_path, "w") as f:
            for seq, info in sequences.items():
                cluster_size = info[1]
                headers = info[0]
                rep_header = headers[0]
                f.write(f">{rep_header[0]} {rep_header[1]} ({cluster_size} total)\n{seq}\n")

                for header in headers:
                    f_csv.write(f"{rep_header[0]},{header[0]},{header[1]},{header[2]}\n")

    ctx.materialize_file(
        source=out_path,
        output_key="fasta",
    )

    ctx.materialize_file(
        source=out_path_csv,
        output_key="cluster_table",
    )
