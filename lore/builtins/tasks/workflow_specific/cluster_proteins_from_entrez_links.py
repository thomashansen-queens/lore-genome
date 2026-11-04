"""
Workflow-specific task skeleton for clustering protein accessions from Entrez links.
"""
from collections.abc import Iterator
from collections import defaultdict

import lore


class Inputs:
    entrez_link_table = lore.ArtifactInput(
        accepted_data=["entrez_link"],
        select="multiple",
        load_as="adapted_stream",
        label="Entrez Link Table",
        description="Table containing Entrez links associated with the protein accessions.",
    )


class Outputs:
    cluster_table = lore.TaskOutput(
        data_type="protein_nuucore_elink_cluster_table",
        label="Cluster Table",
        is_primary=True,
    )

    merged_accessions = lore.TaskOutput(
        data_type="tsv",
        label="Merged Accession Table"
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
    entrez_link_table: Iterator[dict],
):
    clusters = defaultdict(set)
    reverse_mappings = dict()
    conflicts = set()
    merged_accessions = dict()
    reverse_merged_accession_map = dict()
    prev_query_uid = None

    def resolve_conflict(protein_acc_merge, protein_acc_target):
        merged_uids = clusters[protein_acc_merge]
        merged_accessions[protein_acc_merge] = [protein_acc_target, merged_uids]
        if protein_acc_merge in reverse_merged_accession_map.keys():
            remerge_acc = reverse_merged_accession_map[protein_acc_merge]
            merged_accessions[remerge_acc][0] = protein_acc_target
        reverse_merged_accession_map[protein_acc_target] = protein_acc_merge

        for _target_uid in merged_uids:
            reverse_mappings[_target_uid].remove(protein_acc_merge)

    for entry in entrez_link_table:
        query_uid = entry["query_uid"]
        target_uid = entry["target_uid"]
        if prev_query_uid != query_uid:
            if conflicts:
                superset = prev_query_uid
                for uid in conflicts:
                    if len(clusters[superset]) < len(clusters[uid]):
                        resolve_conflict(superset, uid)
                        superset = uid
                    else:
                        resolve_conflict(uid, superset)
                conflicts = set()

            prev_query_uid = query_uid
        clusters[query_uid].add(target_uid)
        if target_uid in reverse_mappings.keys():
            reverse_mappings[target_uid].add(query_uid)
            conflicts = conflicts.union(reverse_mappings[target_uid])
        else:
            reverse_mappings[target_uid] = set([query_uid])
    
    output_path = ctx.get_temp_path("cluster_table.tsv")

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("protein_accession\tcluster_size\tnuucore_uids\n")
        for query_uid, target_uids in clusters.items():
            output_file.write(f"{query_uid}\t{len(target_uids)}\t{",".join(target_uids)}\n")

    ctx.materialize_file(
        source=output_path,
        output_key="cluster_table",
    )

    if merged_accessions:
        ctx.logger.info(f"Found {len(merged_accessions)} merged accessions. ({merged_accessions})")
        merge_acc_output_path = ctx.get_temp_path("merged_accession_table.tsv")
        with open(merge_acc_output_path, 'w', encoding="utf-8") as f:
            f.write("protein_accession\tcluster_size\tmerged_to\tnuucore_uids\n")
            for query_uid, (merged_to, uids) in merged_accessions.items():
                f.write(f"{query_uid}\t{len(uids)}\t{merged_to}\t{','.join(uids)}\n")

        ctx.materialize_file(
            source=merge_acc_output_path,
            output_key="merged_accessions"
        )
