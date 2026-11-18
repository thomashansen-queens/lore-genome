"""Resolve protein sequence and cluster provenance inputs into match outputs."""
import csv
from collections import defaultdict
from collections.abc import Iterator

import lore
from lore.builtins.adapters.protein_nucleotide_assembly import ProteinNucleotideAssemblyMappingAdapter
from lore.builtins.adapters.mmseqs import Mmseqs2ClusterAdapter


class Inputs:
    protein_cluster_table = lore.ArtifactInput(
        accepted_data=[
            "protein_nucleotide_assembly_mapping",
        ],
        select="single",
        load_as="adapted",
        label="Protein cluster table",
        description=(
            "Protein cluster provenance or NCBI protein-to-nucleotide/assembly mappings."
        ),
    )
    alignment_scores = lore.ArtifactInput(
        accepted_data=["alignment_scores"],
        select="single",
        load_as="adapted",
        label="Protein match alignment scores",
    )

    mmseqs_cluster_map = lore.ArtifactInput(
        accepted_data=["mmseqs2_cluster_map"],
        adapt_as=[Mmseqs2ClusterAdapter],
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
        data_type="protein_nucleotide_assembly_mapping",
        label="Protein cluster resolution table",
        is_primary=False,
    )


@lore.task(
    "workflow_specific.resolve_protein_clusters",
    inputs=Inputs,
    outputs=Outputs,
    name="Resolve protein clusters",
    category="Workflow-Specific",
    preview_mode="full",
)
def resolve_protein_clusters(
    ctx: lore.ExecutionContext,
    protein_cluster_table: list[dict],
    mmseqs_cluster_map: list[dict],
    protein_fasta: Iterator[dict],
    alignment_scores: list[dict],
):
    """Combine MMSeqs2 clusters, nuccore provenance, and protein sequences."""
    # Establish mapping for alignment scores
    score_by_accession = {
        str(record.get("accession", "")).strip(): float(record.get("score", 0))
        for record in alignment_scores
    }

    # Establish mapping from representative protein accession to its cluster members
    cluster_mapping = defaultdict(list)
    for row in protein_cluster_table or []:
        representative = str(row.get("representative_protein_accession", "")).strip()
        protein_accession = str(row.get("protein_accession", "")).strip()
        nucleotide_accession = str(row.get("nucleotide_accession", "")).strip()
        assembly_accession = str(row.get("assembly_accession", "")).strip()
        if representative and protein_accession:
            cluster_mapping[representative].append({
                "protein_accession": protein_accession,
                "nucleotide_accession": nucleotide_accession,
                "assembly_accession": assembly_accession,
                "score": score_by_accession[representative]
            })

    # Combine MMSeqs2 clusters with the protein cluster mapping to create a final cluster representation
    clusters = defaultdict(lambda: [0, []])
    for row in mmseqs_cluster_map or []:
        representative = row.get("cluster_rep", "").strip()
        protein_accession = row.get("protein_accession", "").strip()
        clusters[representative][1].extend(cluster_mapping[protein_accession])

    # Sort clusters by average score
    score_maps = []
    for representative, (_, cluster) in clusters.items():
        avg_score = sum(map(lambda entry: entry["score"], cluster)) / len(cluster)
        score_maps.append((avg_score, representative))
    score_maps.sort(key=lambda x: x[0], reverse=True)

    sequence_by_accession = {
        str(record.get("protein_accession", "")).strip(): str(record.get("protein_sequence", "")).strip()
        for record in protein_fasta
    }

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
            "nucleotide_accession",
            "assembly_accession"
        ])

        for score, representative in score_maps: 
            protein_info_list = clusters[representative][1]
            for protein_info in protein_info_list:
                writer.writerow([
                    representative,
                    protein_info["protein_accession"],
                    protein_info["nucleotide_accession"],
                    protein_info["assembly_accession"]
                ])

            representative_sequence = sequence_by_accession[representative]
            if representative_sequence is not None:
                fasta_file.write(f">{representative} ({len(protein_info_list)} total) (avg_score={int(score)})\n")
                fasta_file.write(f"{representative_sequence}\n")

    if missing_sequences:
        ctx.logger.warning(
            "Could not find FASTA sequences for %s cluster representatives; they were omitted from the output FASTA.",
            len(missing_sequences),
        )

    ctx.materialize_file(source=csv_path, output_key="cluster_table")
    ctx.materialize_file(source=fasta_path, output_key="protein_match_fasta")
