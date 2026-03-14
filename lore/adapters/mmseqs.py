"""
Adapter for MMseqs2 two-column TSV output
"""
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from lore.core.adapters import adapter_registry, TableAdapter


class Mmseqs2ClusterAdapter(TableAdapter):
    """
    Adapter for parsing MMseqs2 cluster output in TSV format.
    Two columns: representative sequence and cluster member. No header row!
    Cluster sizes are calculated in a second pass.
    """
    accepted_formats = {"tsv"}
    accepted_types = {"mmseqs2_cluster_map"}
    version = "1.0.0"

    @property
    def schema(self):
        return {
            "cluster_rep": "cluster_rep",
            "protein_accession": "protein_accession",
            "cluster_size": "cluster_size",
        }

    def adapt(self, raw_data: Any, config: dict | None = None) -> list[dict]:
        """
        Intercepts the Engine's request to parse raw content into a list of dicts.
        Handles raw strings, string lists, and mangled dictionaries from generic readers.
        """
        if not raw_data:
            return []

        # --- 1. Normalization Layer ---
        if isinstance(raw_data, str):
            lines = raw_data.strip().splitlines()
        elif isinstance(raw_data, bytes):
            lines = raw_data.decode("utf-8").strip().splitlines()
        elif isinstance(raw_data, list):
            if isinstance(raw_data[0], str):
                lines = raw_data
            elif isinstance(raw_data[0], dict):
                # Is it already correctly formatted?
                if "cluster_rep" in raw_data[0]:
                    return raw_data 
                else:
                    # The generic reader swallowed the first row as a header!
                    # Reconstruct the raw lines to rescue the data.
                    lines = ["\t".join(str(k) for k in raw_data[0].keys())]
                    for row in raw_data:
                        lines.append("\t".join(str(v) for v in row.values()))
            else:
                return super().adapt(raw_data, config)
        else:
            return super().adapt(raw_data, config)

        # --- 2. Pass 1 & Pass 2 (Safely working with strings now) ---
        cluster_counts = Counter()
        parsed_pairs = []

        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 2: # Use >= to safely ignore trailing empty tabs
                rep = parts[0].strip()
                member = parts[1].strip()
                cluster_counts[rep] += 1
                parsed_pairs.append((rep, member))

        return [
            {
                "cluster_rep": rep,
                "protein_accession": member,
                "cluster_size": cluster_counts[rep],
            }
            for rep, member in parsed_pairs
        ]

    def stream(self, path: Path, config: dict | None = None) -> Iterator[dict]:
        """
        Parse MMseqs2 cluster TSV from a file path.
        Two passes: first to extract pairs, second to count cluster sizes.
        """
        cluster_counts = Counter()

        # Pass 1: Stream to tally cluster sizes
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    cluster_counts[parts[0]] += 1

        # Pass 2: Stream again to yield fully-formed records
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    rep, member = parts[0], parts[1]
                    yield {
                        "cluster_rep": rep,
                        "protein_accession": member,
                        "cluster_size": cluster_counts[rep],
                    }


adapter_registry.register(Mmseqs2ClusterAdapter())
