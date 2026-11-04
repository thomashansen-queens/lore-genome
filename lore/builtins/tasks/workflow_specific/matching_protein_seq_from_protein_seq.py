"""
Helper task for matching protein sequences by pairwise protein alignment.
"""
from collections.abc import Iterator
from collections import defaultdict
import csv

from Bio import Align
import lore


class Inputs:
    protein_fasta = lore.ArtifactInput(
        accepted_data=["protein_fasta", "fasta"],
        select="single",
        load_as="adapted_stream",
        label="Protein Sequence FASTA",
        description="FASTA of all the protein sequences to scan",
    )

    compare_seq = lore.ValueInput(
        str,
        label="Sequence to Match",
        description="The protein sequence to match against the protein sequences.",
    )

    entrez_cluster_table = lore.ArtifactInput(
        accepted_data=["protein_nuucore_elink_cluster_table"],
        select="optional",
        load_as="adapted",
        label="Protein cluster table"
    )

    preserve_length = lore.ValueInput(
        bool,
        default=False,
        label="Preserve Length for Alignment",
        description="If true, extends the selected subsequence toward the length of the query sequence.",
    )

    open_gap_score = lore.ValueInput(
        float,
        default=-12.0,
        label="Open gap score",
    )

    extend_gap_score = lore.ValueInput(
        float,
        default=-1.0,
        label="Extend gap score",
    )


class Outputs:
    fasta = lore.TaskOutput(
        data_type="protein_match_fasta",
        label="Protein Match FASTA",
        is_primary=True,
    )

    cluster_table = lore.TaskOutput(
        data_type="csv",
        label="Match Cluster Table",
        is_primary=False,
    )

    failed_matches = lore.TaskOutput(
        data_type="csv",
        label="Failed Matches",
        is_primary=False,
    )


@lore.task(
    "aleyssu.gene_from_protein_workflow.match_protein_protein",
    inputs=Inputs,
    outputs=Outputs,
    name="Match Protein Sequences to Protein Subsequence",
    category="Workflow-Specific",
    preview_mode="full",
    icon="",
)
def task(
    ctx: lore.ExecutionContext,
    protein_fasta: Iterator[dict],
    compare_seq: str,
    preserve_length: bool = False,
    extend_gap_score: float = -1.0,
    open_gap_score: float = -12.0,
    entrez_cluster_table = None
):
    """Match and trim protein sequences against a protein query using local BLASTP alignment."""
    sequences = dict()
    failed_matches = []
    full_span_seqs = set()

    e_cluster_table = defaultdict(lambda: 1)  # Default to cluster size of 1 if missing
    if entrez_cluster_table:
        for entry in entrez_cluster_table:
            e_cluster_table[entry["protein_accession"]] = int(entry['cluster_size'])

    compare_seq = "".join(compare_seq.strip().split()).upper()
    if not compare_seq:
        raise ValueError("The protein sequence to match cannot be empty.")
    compare_seq_len = len(compare_seq)

    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "local"
    aligner.extend_gap_score = extend_gap_score
    aligner.open_gap_score = open_gap_score

    for entry in protein_fasta:
        accession = entry.get("protein_accession") or entry.get("accession", "unknown_entry")
        description = entry.get("protein_description") or entry.get("description") or ""
        seq = entry["protein_sequence"]
        seq = "".join(seq.strip().split()).upper()

        duplicate = any(subseq in seq for subseq in full_span_seqs)
        if duplicate:
            idx = next(seq.find(subseq) for subseq in full_span_seqs if subseq in seq)
            subseq = next(candidate for candidate in full_span_seqs if candidate in seq)
            m_start = m_end = 0
        else:
            alignment = aligner.align(compare_seq, seq)
            if not alignment:
                failed_matches.append([accession, e_cluster_table[accession], seq])

            best_alignment = alignment[0]
            target_blocks = best_alignment.aligned[1]
            query_blocks = best_alignment.aligned[0]
            t_start = target_blocks[0][0]
            t_end = target_blocks[-1][1]
            m_start = query_blocks[0][0]
            m_end = query_blocks[-1][1]

            if preserve_length:
                subseq = seq[max(0, t_start - m_start):max(t_end, t_start - m_start + compare_seq_len)]
            else:
                subseq = seq[t_start:t_end]

            if compare_seq_len <= t_end - t_start:
                full_span_seqs.add(subseq)

            idx = t_start

        cluster = sequences.setdefault(subseq, [[], [], []])
        cluster[0].append(accession)
        cluster[1].append(e_cluster_table[accession])
        cluster[2].append(f"{idx}-{idx + len(subseq)}")

    out_path = ctx.get_temp_path("protein_fasta.faa")
    out_path_csv = ctx.get_temp_path("match_table.csv")
    out_path_fails = ctx.get_temp_path("failed_match_table.csv")

    with open(out_path_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["representative_protein_accession", "protein_accession", "cluster_size_contribution", "selected_region_in_protein"])

        with open(out_path, "w", encoding="utf-8") as f:
            for subseq, headers in sequences.items():
                representative_index = max(range(len(headers[0])), key=lambda index: headers[1][index])
                representative = headers[0][representative_index]
                representative_range = headers[2][representative_index]
                cluster_total = sum(headers[1])

                f.write(f">{representative} {representative_range} ({cluster_total} total)\n{subseq}\n")

                for i, accession in enumerate(headers[0]):
                    cluster_size_contribution = headers[1][i]
                    selected_range = headers[2][i]
                    writer.writerow([representative, accession, cluster_size_contribution, selected_range])
    
    ctx.materialize_file(source=out_path_csv, output_key="cluster_table")
    ctx.materialize_file(source=out_path, output_key="fasta")

    if failed_matches:
        with open(out_path_fails, 'w', encoding='utf-8') as f:
            f.write("protein_accession,cluster_size,protein_sequence\n")
            for protein_accession, cluster_size, protein_sequence in failed_matches:
                f.write(f"{protein_accession},{cluster_size},{protein_sequence}\n")

        ctx.materialize_file(source=out_path_fails, output_key="failed_matches")
