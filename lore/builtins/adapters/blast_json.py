"""
Adapter for NCBI BLAST JSON2_S integration.
"""
import lore


@lore.adapter()
class BlastJsonAdapter(lore.JsonAdapter):
    """
    Extracts and tabulates hit-level data from NCBI BLAST JSON2_S output.
    Flattens the deeply nested BlastOutput2 structure into a clean, row-per-hit format.
    """
    accepted_types = {"blast_results"}

    @property
    def schema(self):
        return {
            "accession": "description[0].accession",
            "title": "description[0].title",
            "species": "description[0].sciname",
            "subject_length": "len", 
            # Take top high-scoring pair (HSP) for each hit
            "evalue": "hsps[0].evalue",
            "alignment_length": "hsps[0].align_len",
            "identity": "hsps[0].identity",
            "bit_score": "hsps[0].bit_score",
        }

    def _extract_hits(self, raw_data):
        """Helper to tunnel through the NCBI JSON nesting and extract the hits list."""
        if isinstance(raw_data, dict):
            raw_data = [raw_data]

        records = []
        for item in raw_data:
            if isinstance(item, dict) and "BlastOutput2" in item:
                # NCBI returns BlastOutput2 as an array
                for output in item["BlastOutput2"]:
                    search = output.get("report", {}).get("results", {}).get("search", {})
                    records.extend(search.get("hits", []))
            else:
                # Fallback if it's already stripped or malformed
                records.append(item)
        return records

    def parse(self, raw_data, config=None, **kwargs):
        """
        Parse BLAST JSON output. Strips the outer metadata and returns only
        the list of 'hits' records.
        """
        hits = self._extract_hits(raw_data)
        return super().parse(hits, config, **kwargs)

    def parse_stream(self, raw_stream, config=None, **kwargs):
        """
        Yield records from BLAST JSON output. Strips outer metadata and yields
        only the list of 'hits'.
        """
        for record in super().parse_stream(raw_stream, config, **kwargs):
            hits = self._extract_hits(record)
            yield from hits
