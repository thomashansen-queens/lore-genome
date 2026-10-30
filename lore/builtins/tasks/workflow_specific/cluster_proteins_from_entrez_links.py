"""
Workflow-specific task skeleton for clustering protein accessions from Entrez links.
"""
from collections.abc import Iterator

import lore


class Inputs:
    protein_accessions = lore.ArtifactInput(
        accepted_data=["uid", "target_uid"],
        select="multiple",
        load_as="adapted_stream",
        label="Protein Accessions",
        description="Protein accession list used to build the cluster table.",
    )

    entrez_link_table = lore.ArtifactInput(
        accepted_data=["entrez_link"],
        select="multiple",
        load_as="adapted_stream",
        label="Entrez Link Table",
        description="Table containing Entrez links associated with the protein accessions.",
    )


class Outputs:
    cluster_table = lore.TaskOutput(
        data_type="csv",
        label="Cluster Table",
        is_primary=True,
    )


@lore.task(
    "workflow_specific.cluster_proteins_from_entrez_links",
    inputs=Inputs,
    outputs=Outputs,
    name="Cluster Proteins from Entrez Links",
    category="Workflow-Specific",
    preview_mode="live",
)
def cluster_proteins_from_entrez_links(
    ctx: lore.ExecutionContext,
    protein_accessions: Iterator[dict],
    entrez_link_table: Iterator[dict],
):
    clusters = dict()
    reverse_mappings = dict()
    conflicts = set()
    prev_source_uid = None
    for entry in entrez_link_table:
        source_uid = entry["source_uid"]
        target_uid = entry["target_uid"]
        if prev_source_uid != source_uid:
            protein_acc = next(protein_accessions)
            if protein_acc not in clusters.keys():
                clusters[protein_acc] = set()
            if conflicts:
                superset = protein_acc
                for uid in conflicts:
                    if len(clusters[superset]) < len(clusters[uid]):
                        for _target_uid in clusters.pop(superset):
                            reverse_mappings[_target_uid].remove(superset)
                        superset = uid
                    else:
                        for _target_uid in clusters.pop(uid):
                            reverse_mappings[_target_uid].remove(uid)
                conflicts = set()

            prev_source_uid = source_uid
        clusters[protein_acc].add(target_uid)
        if target_uid in reverse_mappings.keys():
            conflicts = conflicts.union(reverse_mappings[target_uid])
            reverse_mappings[target_uid].add(protein_acc)
        else:
            reverse_mappings[target_uid] = set([protein_acc])
    
    output_path = ctx.get_temp_path("cluster_table.csv")

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("protein_acc,cluster_size,nuucore_uids\n")
        for protein_acc in clusters.keys():
            target_uids = clusters[protein_acc]
            output_file.write(f"{protein_acc},{len(target_uids)},{";".join(target_uids)}\n")

    ctx.materialize_file(
        source=output_path,
        output_key="cluster_table",
    )
