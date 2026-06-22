"""
An Adapter is a class that converts and flattens domain-specific raw data into
and analysis-ready format. Think of it like a lens through which you view your
data.

Every Adapter subclasses BaseAdapter and is registered in the adapter_registry.

To define an adapter, use the decorator provided by the registry:
    @adapter()
    class MyAdapter(BaseAdapter):
        ...
"""
from .base import BaseAdapter, AdapterPreview
from .svg import SvgAdapter
from .tabular import TabularAdapter
from .registry import AdapterRegistry
from .parsers.csv import CsvAdapter
from .parsers.json import JsonAdapter
from .parsers.text import TextAdapter

# Instantiate the global registry and register base adapters.
adapter_registry = AdapterRegistry()

adapter_registry._register_core(CsvAdapter())
adapter_registry._register_core(JsonAdapter())
adapter_registry._register_core(SvgAdapter())
adapter_registry._register_core(TextAdapter())


__all__ = [
    # Adapter classes
    "BaseAdapter",
    "TabularAdapter",
    # UI preview class
    "AdapterPreview",
    # Registry class and global singleton instance
    "AdapterRegistry",
    "adapter_registry",
    # Built-in base type adapters
    "CsvAdapter",
    "JsonAdapter",
    "SvgAdapter",
    "TextAdapter",
]
