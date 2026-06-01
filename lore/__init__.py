"""
LoRē Genome : A Long-Read alignment and classification toolkit.

This package provides a centralized RuntimeContext for managing 
genomic data artifacts across both CLI and Web interfaces.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lore-genome")
except PackageNotFoundError:
    __version__ = "unknown"
__app_name__ = "LoRē Genome"

# Public API is defined in core/dsl.py, a façade module that collects and 
# re-exports all the core components of the LoRē framework.
from .core.dsl import *
from .core.dsl import __all__ as _dsl_all

__all__ = ["__version__", "__app_name__"] + _dsl_all  # pyright: ignore[reportUnsupportedDunderAll]
