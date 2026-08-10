"""
Adapter for InterProScan integration.
"""
import lore


def _count_db_matches(record, db_name):
    """Helper to count matches for a specific database from the record."""
    return sum(
        1 for match in record.get("matches", [])
        if match.get("signature", {}).get("signatureLibraryRelease", {}).get("library", "").upper() \
        == db_name.upper()
    )


@lore.adapter()
class InterproscanAdapter(lore.JsonAdapter):
    """
    Tabulates some of the most useful fields from InterProScan JSON
    output. The data itself hierarchical and does not lend itself to flat
    tabulation, so this adapter is designed to provide a human-readable
    summary.
    """
    accepted_types = {"interproscan_json"}

    @property
    def schema(self):
        return {
            "protein_accession": "xref[0].id",
            "name": "xref[0].name",
            "length": lambda x: len(x.get("sequence", "")),
            "total_matches": lambda x: len(x.get("matches", [])),
            # Subset of highly useful domain databases
            "pfam_domains": lambda x: _count_db_matches(x, "PFAM"),
            "panther_domains": lambda x: _count_db_matches(x, "PANTHER"),
            "cdd_domains": lambda x: _count_db_matches(x, "CDD"),
            "gene3d_domains": lambda x: _count_db_matches(x, "GENE3D"),
        }

    def parse(self, raw_data, config=None, **kwargs):
        """
        Parse InterProScan JSON output. Strips the outer metadata and returns only
        the list of 'results' records. JsonReader wraps dicts into a list, so we
        must unwrap here.
        """
        if isinstance(raw_data, dict):
            raw_data = [raw_data]

        if isinstance(raw_data, list):
            records = []
            for item in raw_data:
                if isinstance(item, dict) and "results" in item:
                    records.extend(item["results"])
                else:
                    records.append(item)
            return super().parse(records, config, **kwargs)

        # Fallback
        return super().parse(raw_data, config, **kwargs)

    def parse_stream(self, raw_stream, config=None, **kwargs):
        """
        Yield records from InterProScan JSON output. Strips outer metadata and yields
        only the list of 'results'.
        """
        for record in super().parse_stream(raw_stream, config, **kwargs):
            if isinstance(record, dict) and "results" in record:
                yield from record["results"]
            else:
                yield record
