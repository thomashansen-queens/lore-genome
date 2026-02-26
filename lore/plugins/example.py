"""
End goal: A user should be able to write a simple Adapter class like this in a
pluging and it "just works" with the rest of the LoRe system (Workbench, Artifacts, etc.)
"""
from lore.core.adapters import BaseAdapter, adapter_registry

class InterProScanAdapter(BaseAdapter):
    """
    Community Plugin for InterProScan TSV/JSON output.
    """
    accepted_types = {"interproscan_results"}
    accepted_formats = {"tsv", "json"}

    @property
    def field_map(self):
        # The user just maps the external tool's jargon to your standard names
        return {
            "protein_accession": "protein_accession",
            "md5": "md5",
            "length": lambda r: int(r.get("sequence_length", 0)),
            "analysis": "analysis",
            "signature_accession": "signature_accession",
            "description": "signature_description",
            "e_value": lambda r: float(r.get("e_value")) if r.get("e_value") != '-' else None,
        }

    def load(self, path):
        # They define a simple loader for the tool's specific format
        # (InterProScan JSON is often a list of complex objects)
        import json
        with path.open() as f:
            data = json.load(f)
            # Flatten the nested structure slightly to make it a list of hits
            return [hit for protein in data['results'] for hit in protein['matches']]

# The "Plug" - registering it with your backend
adapter_registry.register(InterProScanAdapter())