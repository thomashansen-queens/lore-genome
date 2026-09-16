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
        label="Sequence to Match",
        description="The DNA sequence to match the nucleotide sequences to.",
    )

    condense_header = lore.ValueInput(
        bool,
        default=False,
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
    name="Match DNA Sequences to Nucleotide Subsequence",
    category="Workflow-Specific",
    preview_mode="full",
    icon="⏵⏴",
)
def task(
    ctx: lore.ExecutionContext,
    nucleotide_fasta: Iterator[dict],
    compare_seq: str,
    condense_header: bool = True,
):
    """Given a fasta containing nucleotide sequences, will match and trim all the sequences to a provided representative subsequence."""
    sequences = dict()

    compare_seq = "".join(compare_seq.strip().split())

    aligner = Align.PairwiseAligner(scoring="blastn")
    aligner.mode = "global"
    
    for entry in nucleotide_fasta:
        seq = entry["nucleotide_sequence"]
        duplicate = False
        for subseq in sequences.keys():
            idx = seq.find(subseq)
            if idx != -1:
                duplicate = True
                break
        if not duplicate:
            alignment = aligner.align(compare_seq, seq)

            # Coordinates are [start, end] for each sequence
            t_start, t_end = alignment[0].aligned[1][0]

            subseq = seq[t_start:t_end]
            sequences[subseq] = []
        else:
            t_start, t_end = idx, idx + len(subseq)

        header = f"{entry['nucleotide_accession']} {entry['nucleotide_description'].replace(" ", "_")}_{t_start}-{t_end}"
        sequences[subseq].append(header)

    out_path = ctx.get_temp_path("nucleotide_fasta.faa")

    with open(out_path, "w") as f:
        for seq, headers in sequences.items():
            if condense_header:
                f.write(f">{headers[0]} (+{len(headers)-1})\n{seq}\n")
            else: 
                f.write(f">{", ".join(headers)} ({len(headers)} total)\n{seq}\n")

    ctx.materialize_file(
        source=out_path,
        output_key="fasta",
    )
