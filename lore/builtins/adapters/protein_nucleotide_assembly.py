"""Adapters for NCBI protein-to-nucleotide and assembly mapping data."""

import lore


@lore.adapter()
class ProteinNucleotideAssemblyMappingAdapter(lore.CsvAdapter):
    """Summarize IPG rows as representative protein accession and cluster size."""

    accepted_formats = {"csv"}
    accepted_types = {"protein_nucleotide_assembly_mapping"}
    version = "1.0.0"
    priority = -1

    def adapt(self, raw_data, config=None, **kwargs):
        records = super().parse(raw_data, config=config, **kwargs)
        members_by_representative: dict[str, dict[tuple[str, str], dict]] = {}

        for record in records:
            representative = str(record.get("representative_protein_accession") or "").strip()
            member = str(record.get("protein_accession") or "").strip()
            if representative and member:
                nucleotide = str(record.get("nucleotide_accession") or "").strip()
                members_by_representative.setdefault(representative, {})[
                    (member, nucleotide)
                ] = record

        return [
            {
                "protein_accession": representative,
                "cluster_size": len(members),
                "members": list(members.values()),
            }
            for representative, members in members_by_representative.items()
        ]
