"""
General purpose adapters for JSON
"""

from lore.core.adapters import BaseAdapter, adapter_registry

class GenericJsonAdapter(BaseAdapter):
    """
    Adapter for undefined JSON data.
    """
    accepted_formats = {"json", "jsonl", "ndjson"}
    accepted_types = {"*"}

    @property
    def schema(self):
        return {} # Triggers the "Generic Mode" pass-through


adapter_registry.register(GenericJsonAdapter())
