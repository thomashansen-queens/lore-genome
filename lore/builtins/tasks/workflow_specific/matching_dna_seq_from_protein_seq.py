"""
Helper task for gene sequene from protein workflow pipeline
"""
from collections.abc import Iterator
import lore
import pandas as pd
from Bio import Align

class Inputs:
    nucleotide_fasta = lore.ArtifactInput(
        accepted_data=["nucleotide_fasta", "gene_fasta"],
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

    preserve_length = lore.ValueInput(
        bool,
        default=False,
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
        data_type="nucleotide_match_fasta",
        label="Nucleotide FASTA",
        is_primary=True,
    )

    cluster_table = lore.TaskOutput(
        data_type="csv",
        label="Match Cluster Table",
        is_primary=False,
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

    known_fasta_type = False
    
    for entry in nucleotide_fasta:
        if not known_fasta_type:
            if "gene_locus" in entry.keys():
                fasta_type = "gene_fasta"
            else:
                fasta_type = "nucleotide_fasta"
            known_fasta_type = True

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

        if fasta_type == "gene_fasta":
            header = [entry['nucleotide_accession'], entry['protein_accession'], entry["gene_locus"], entry["nucleotide_sequence"], f"{m_start}-{m_end}", ]
        else:
            header = [entry['nucleotide_accession'], entry['nucleotide_description'], entry["nucleotide_sequence"], f"{m_start}-{m_end}"]
        sequences[subseq].append(header)

    out_path = ctx.get_temp_path("nucleotide_fasta.faa")
    out_path_csv = ctx.get_temp_path("match_table.csv")

    with open(out_path_csv, "w") as f_csv:
        if fasta_type == "gene_fasta":
            f_csv.write("representative_nucleotide_accession,nucleotide_accession,protein_accession,gene_locus,selected_region_in_gene\n")
        else:
            f_csv.write("representative_nucleotide_accession,nucleotide_accession,description,selected_region_in_sequence\n")

        with open(out_path, "w") as f:
            for seq, headers in sequences.items():
                rep_header = headers[0]
                f.write(f">{rep_header[0]} {rep_header[1]} ({len(headers)} total)\n{seq}\n")

                for header in headers:
                    if fasta_type == "gene_fasta":
                        f_csv.write(f"{rep_header[0]},{header[0]},{header[1]},{header[2]},{header[4]}\n")
                    else:
                        f_csv.write(f"{rep_header[0]},{header[0]},{header[1]},{header[3]}\n")

    ctx.materialize_file(
        source=out_path_csv,
        output_key="cluster_table",
    )

    ctx.materialize_file(
        source=out_path,
        output_key="fasta",
    )