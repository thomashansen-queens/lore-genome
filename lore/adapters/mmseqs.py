"""
Adapter for MMseqs2 two-column TSV output
"""
from collections import Counter
from lore.core.adapters import adapter_registry, TableAdapter


class Mmseqs2ClusterAdapter(TableAdapter):
    """
    Adapter for parsing MMseqs2 cluster output in TSV format.
    Expects two columns: "representative" and "cluster_member", where the representative column contains the cluster head (representative sequence) and the cluster_member column contains all sequences in the cluster (including the representative).
    """
    accepted_formats = {"tsv"}
    accepted_types = {"mmseqs2_cluster_map", "tsv"}

    @property
    def schema(self):
        return {
            "cluster_rep": "cluster_rep",
            "protein_accession": "protein_accession",
            "cluster_size": "cluster_size",
        }

    def parse(self, raw_data) -> list[dict]:
        """
        Parse MMseqs2 cluster TSV data into a list of dictionaries.
        """
        if not raw_data.strip():
            return []
        records = []

        # Pass 1: Extract valid lines into a list of (representative, member) pairs
        valid_pairs = []
        for line in raw_data.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                valid_pairs.append((parts[0], parts[1]))

        # Pass 2: Count cluster sizes
        cluster_counts = Counter(rep for rep, member in valid_pairs)

        # Pass 3: Build records
        records = []
        for rep, member in valid_pairs:
            records.append({
                "cluster_rep": rep,
                "protein_accession": member,
                "cluster_size": cluster_counts[rep]
            })

        return records


adapter_registry.register(Mmseqs2ClusterAdapter())
