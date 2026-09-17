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

    preserve_length = lore.ValueInput(
        bool,
        default=True,
        label="Preserve Length for Alignment",
        description="If true, tries to extend the length of the aligned subsequence to match the length of the query sequence if the alignment algorithm runs short.",
    )

    match_score = lore.ValueInput(
        float,
        default = 1.0,
        label = "Match score"
    )

    mismatch_score = lore.ValueInput(
        float,
        default = 0.0,
        label = "Mismatch score"
    )

    open_gap_score = lore.ValueInput(
        float,
        default = -7.0,
        label = "Open gap score"
    )

    extend_gap_score = lore.ValueInput(
        float,
        default = -2.0,
        label = "Extend gap score"
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
    icon="",
)
def task(
    ctx: lore.ExecutionContext,
    nucleotide_fasta: Iterator[dict],
    compare_seq: str,
    condense_header: bool = True,
    preserve_length: bool = True,
    extend_gap_score: float = -2.0,
    open_gap_score: float = -7.0,
    mismatch_score: float = 0,
    match_score: float = 1,
):
    """Given a fasta containing nucleotide sequences, will match and trim all the sequences to a provided representative subsequence. Uses BioPython's Pairwise alignment tool with blastn scoring and local alignment mode."""
    sequences = dict()
    full_span_seqs = set()

    compare_seq = "".join(compare_seq.strip().split()).upper()
    compare_seq_len = len(compare_seq)

    aligner = Align.PairwiseAligner(scoring="blastn")
    aligner.mode = "local"
    aligner.extend_gap_score = extend_gap_score
    aligner.open_gap_score = open_gap_score
    aligner.mismatch_score = mismatch_score
    aligner.match_score = match_score
    
    for entry in nucleotide_fasta:
        seq = entry["nucleotide_sequence"]
        # Check to see if a previous full-length match is an exact match within the current sequence 
        duplicate = False
        for subseq in full_span_seqs:
            idx = seq.find(subseq)
            if idx != -1:
                duplicate = True
                break
        if not duplicate:
            alignment = aligner.align(compare_seq, seq)

            # Coordinates are [start, end] for each sequence
            t_start = alignment[0].aligned[1][0][0]
            t_end = alignment[0].aligned[1][-1][1]

            m_start = alignment[0].aligned[0][0][0]
            m_end = alignment[0].aligned[0][-1][1]

            # Extend the match to try to match the length of the query sequence
            if preserve_length:
                subseq = seq[max(0, t_start - m_start): max(t_end, t_start - m_start + compare_seq_len)]
            else:
                subseq = seq[t_start:t_end]
            if subseq not in sequences.keys():
                sequences[subseq] = []

            # Full length matches can be added to a set to scan future sequences for exact matches and skip expensive alignment computations
            if compare_seq_len <= t_end - t_start:
                full_span_seqs.add(subseq)
        else:
            t_start, t_end = idx, idx + len(subseq)

        header = f"{entry['nucleotide_accession']} {entry['nucleotide_description'].replace(" ", "_")}({m_start}-{m_end})"
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
