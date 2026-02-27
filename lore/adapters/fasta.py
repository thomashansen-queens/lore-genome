"""
Adapter for handling FASTA files in LoRe Genome. This module provides utilities 
for reading, writing, and manipulating FASTA files
"""
from lore.core.adapters import adapter_registry, TableAdapter


class FastaAdapter(TableAdapter):
    """Adapter for FASTA files"""
    accepted_formats = {"fasta", "faa", "fa"}
    accepted_types = {"protein_fasta", "nucleotide_fasta"}

    @property
    def schema(self):
        return {
            "accession": "accession",
            "name": "name",
            "length": lambda x: len(x["sequence"]) if "sequence" in x else None,
            "sequence": "sequence",
        }

    def parse(self, raw_data) -> list[dict]:
        """
        Parse FASTA to tabular format with columns: accession, name, sequence
        """
        records = []
        current_record = {"accession": None, "name": None, "sequence": []}
        for line in raw_data.splitlines():
            line = line.strip()
            if line.startswith(">"):
                if current_record["accession"] is not None:
                    current_record["sequence"] = "".join(current_record["sequence"])
                    records.append(current_record)
                header_parts = line[1:].split(None, 1)
                current_record = {
                    "accession": header_parts[0],
                    "name": header_parts[1] if len(header_parts) > 1 else None,
                    "sequence": [],
                }
            elif line:
                current_record["sequence"].append(line)

        if current_record["accession"] is not None:  # final record
            current_record["sequence"] = "".join(current_record["sequence"])
            records.append(current_record)

        return records


adapter_registry.register(FastaAdapter())
