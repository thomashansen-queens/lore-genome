"""Resolve protein sequence and cluster provenance inputs into match outputs."""
from collections import defaultdict
from collections.abc import Iterator
import csv

import lore


class Inputs:
    entrez_cluster_table = lore.ArtifactInput(
        accepted_data=["protein_nuucore_elink_cluster_table"],
        select="single",
        load_as="adapted",
        label="Protein-to-nuucore cluster table",
    )
    mmseqs_cluster_map = lore.ArtifactInput(
        accepted_data=["mmseqs2_cluster_map"],
        select="single",
        load_as="adapted",
        label="MMSeqs2 cluster map",
    )
    protein_fasta = lore.ArtifactInput(
        accepted_data=["protein_fasta"],
        select="single",
        load_as="adapted_stream",
        label="Protein FASTA",
    )


class Outputs:
    protein_match_fasta = lore.TaskOutput(
        data_type="protein_match_fasta",
        label="Protein match FASTA",
        is_primary=True,
    )
    cluster_table = lore.TaskOutput(
        data_type="csv",
        label="Protein cluster resolution table",
        is_primary=False,
    )


@lore.task(
    "workflow_specific.resolve_protein_clusters",
    inputs=Inputs,
    outputs=Outputs,
    name="Resolve protein clusters",
    category="Workflow-Specific",
    preview_mode="live",
)
def resolve_protein_clusters(
    ctx: lore.ExecutionContext,
    entrez_cluster_table: list[dict],
    mmseqs_cluster_map: list[dict],
    protein_fasta: Iterator[dict],
):
    """Combine MMSeqs2 clusters, nuccore provenance, and protein sequences."""
    provenance = {}
    for row in entrez_cluster_table or []:
        accession = str(row.get("protein_accession", "")).strip()
        if not accession:
            continue
        try:
            contribution = max(int(row.get("cluster_size", 1)), 0)
        except (TypeError, ValueError):
            contribution = 1
        raw_uids = row.get("nuucore_uids", "")
        if isinstance(raw_uids, str):
            uids = [uid.strip() for uid in raw_uids.split(",") if uid.strip()]
        elif isinstance(raw_uids, (list, tuple, set)):
            uids = [str(uid).strip() for uid in raw_uids if str(uid).strip()]
        else:
            uids = []
        provenance[accession] = (contribution, ",".join(uids))

    sequence_by_accession = {}
    for record in protein_fasta or []:
        accession = str(record.get("protein_accession") or record.get("accession") or "").strip()
        if accession:
            sequence_by_accession[accession] = str(record.get("protein_sequence") or record.get("sequence") or "")

    clusters = defaultdict(list)
    for row in mmseqs_cluster_map or []:
        representative = str(row.get("cluster_rep", "")).strip()
        member = str(row.get("protein_accession", "")).strip()
        if representative and member:
            clusters[representative].append(member)

    fasta_path = ctx.get_temp_path("protein_match_fasta.faa")
    csv_path = ctx.get_temp_path("protein_cluster_resolution.csv")

    missing_sequences = set()
    with open(fasta_path, "w", encoding="utf-8", newline="") as fasta_file, open(
        csv_path, "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow([
            "representative_protein_accession",
            "protein_accession",
            "cluster_size_contribution",
            "nuucore_uids",
        ])

        for representative, members in clusters.items():
            unique_members = list(dict.fromkeys(members))
            total_size = 0
            representative_sequence = sequence_by_accession.get(representative)

            if representative_sequence is None:
                representative_sequence = next(
                    (sequence_by_accession[member] for member in unique_members if member in sequence_by_accession),
                    None,
                )
            if not representative_sequence:
                missing_sequences.add(representative)

            for member in unique_members:
                contribution, nuucore_uids = provenance.get(member, (1, ""))
                total_size += contribution
                writer.writerow([
                    representative,
                    member,
                    contribution,
                    nuucore_uids,
                ])

            if representative_sequence is not None:
                fasta_file.write(f">{representative} protein ({total_size} total)\n")
                fasta_file.write(f"{representative_sequence}\n")

    if missing_sequences:
        ctx.logger.warning(
            "Could not find FASTA sequences for %s cluster representatives; they were omitted from the output FASTA.",
            len(missing_sequences),
        )

    ctx.materialize_file(source=csv_path, output_key="cluster_table")
    ctx.materialize_file(source=fasta_path, output_key="protein_match_fasta")
