"""
Helper task for matching protein sequences by pairwise protein alignment.
"""
import csv
from collections import defaultdict
from collections.abc import Iterator

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

    cluster_table = lore.ArtifactInput(
        accepted_data=["protein_nucleotide_assembly_mapping"],
        select="optional",
        load_as="adapted",
        label="Protein cluster table"
    )

    preserve_length = lore.ValueInput(
        bool,
        default=False,
        label="Preserve Length for Alignment",
        description=(
            "If true, extends the selected subsequence toward the length of the query sequence."
        ),
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

    alignment_scores = lore.TaskOutput(
        label="Protein Match Alignment Scores",
        data_type = "alignment_scores",
        is_primary=False,
    )

    cluster_table = lore.TaskOutput(
        data_type="protein_nucleotide_assembly_mapping",
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
    cluster_table=None
):
    """Match and trim protein sequences against a protein query using local BLASTP alignment."""
    sequences = dict()
    failed_matches = []
    full_span_scores = {}

    cluster_map = defaultdict(lambda: [0, []])
    member_to_representative = {}
    if cluster_table:
        cluster_members = defaultdict(dict)
        for entry in cluster_table:
            representative = str(
                entry.get("representative_protein_accession")
                or entry.get("protein_accession")
                or ""
            ).strip()
            if not representative:
                continue
            members = entry.get("members") or [entry]
            for member_entry in members:
                member = str(member_entry.get("protein_accession") or "").strip()
                if not member:
                    continue
                nucleotide = str(member_entry.get("nucleotide_accession") or "").strip()
                cluster_members[representative][(member, nucleotide)] = member_entry
                member_to_representative[member] = representative

        for representative, members in cluster_members.items():
            cluster_map[representative][0] = len(members)
            cluster_map[representative][1].extend(members.values())

    compare_seq = "".join(compare_seq.strip().split()).upper()
    if not compare_seq:
        raise ValueError("The protein sequence to match cannot be empty.")
    compare_seq_len = len(compare_seq)

    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "local"
    aligner.extend_gap_score = extend_gap_score
    aligner.open_gap_score = open_gap_score

    processed_clusters = set()
    for entry in protein_fasta:
        accession = str(
            entry.get("protein_accession") or entry.get("accession", "unknown_entry")
        ).strip()
        cluster_representative = member_to_representative.get(accession, accession)
        if cluster_map and cluster_representative in processed_clusters:
            continue
        if cluster_map:
            processed_clusters.add(cluster_representative)

        seq = entry["protein_sequence"]
        seq = "".join(seq.strip().split()).upper()
        full_matches = [candidate for candidate in full_span_scores if candidate in seq]
        duplicate = len(full_matches) > 0
        if duplicate:
            idx = next(seq.find(subseq) for subseq in full_span_scores if subseq in seq)
            subseq = full_matches[0]
            score = full_span_scores[subseq]
        else:
            alignment = aligner.align(compare_seq, seq)
            if not alignment:
                failed_matches.append([accession, cluster_map[cluster_representative][0], seq])
                continue

            best_alignment = alignment[0]
            score = best_alignment.score
            target_blocks = best_alignment.aligned[1]
            query_blocks = best_alignment.aligned[0]
            t_start = target_blocks[0][0]
            t_end = target_blocks[-1][1]
            m_start = query_blocks[0][0]
            if preserve_length:
                subseq = seq[
                    max(0, t_start - m_start):max(
                        t_end, t_start - m_start + compare_seq_len
                    )
                ]
            else:
                subseq = seq[t_start:t_end]

            if compare_seq_len <= t_end - t_start:
                full_span_scores[subseq] = score

            idx = t_start

        cluster = sequences.setdefault(subseq, [[], [], [], [], []])
        cluster[0].append(accession)
        cluster[1].append(cluster_map[cluster_representative])
        cluster[2].append(f"{idx}-{idx + len(subseq)}")
        cluster[3].append(score)
        cluster[4].append(cluster_representative)

    out_path = ctx.get_temp_path("protein_fasta.faa")
    out_path_csv = ctx.get_temp_path("match_table.csv")
    out_path_scores = ctx.get_temp_path("alignment_scores.csv")
    out_path_fails = ctx.get_temp_path("failed_match_table.csv")

    with open(out_path_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "representative_protein_accession",
                "protein_accession",
                "nucleotide_accession",
                "assembly_accession",
                "selected_region_in_protein",
            ]
        )

        with open(out_path, "w", encoding="utf-8") as f:
            fasta_records = []
            for subseq, headers in sequences.items():
                representative_index = max(
                    range(len(headers[0])),
                    key=lambda i: cluster_map[headers[4][i]][0],
                )
                representative = headers[4][representative_index]
                representative_range = headers[2][representative_index]
                score = headers[3][representative_index]

                cluster_total = 0
                for i, accession in enumerate(headers[0]):
                    selected_range = headers[2][i]
                    cluster_entries = headers[1][i][1]
                    cluster_size = headers[1][i][0]
                    if cluster_size > 0:
                        cluster_total += cluster_size
                        for entry in cluster_entries:
                            member_accession = entry.get("protein_accession") or accession
                            nucleotide_accession = entry.get("nucleotide_accession", "")
                            assembly_accession = entry.get("assembly_accession", "")
                            writer.writerow(
                                [
                                    representative,
                                    member_accession,
                                    nucleotide_accession,
                                    assembly_accession,
                                    selected_range,
                                ]
                            )
                    else:
                        cluster_total += 1
                        writer.writerow([representative, accession, "", "", selected_range])

                fasta_records.append(
                    (score, representative, representative_range, cluster_total, subseq)
                )

            with open(out_path_scores, "w", encoding="utf-8") as scores_file:
                scores_file.write("accession,score\n")
                for score, representative, representative_range, cluster_total, subseq in sorted(
                    fasta_records, key=lambda record: record[0], reverse=True
                ):
                    f.write(
                        f">{representative} {representative_range} ({cluster_total} total) "
                        f"score={score:g}\n{subseq}\n"
                    )
                    scores_file.write(f"{representative},{score:g}\n")
    
    ctx.materialize_file(source=out_path_csv, output_key="cluster_table")
    ctx.materialize_file(source=out_path, output_key="fasta")
    ctx.materialize_file(source=out_path_scores, output_key="alignment_scores")

    if failed_matches:
        with open(out_path_fails, 'w', encoding='utf-8') as f:
            f.write("protein_accession,cluster_size,protein_sequence\n")
            for protein_accession, cluster_size, protein_sequence in failed_matches:
                f.write(f"{protein_accession},{cluster_size},{protein_sequence}\n")

        ctx.materialize_file(source=out_path_fails, output_key="failed_matches")
