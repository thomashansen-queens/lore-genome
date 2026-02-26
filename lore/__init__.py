"""
LoRe Genome: A Long-Read alignment and classification toolkit.

This package provides a centralized RuntimeContext for managing 
genomic data artifacts across both CLI and Web interfaces.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lore-genome")
except PackageNotFoundError:
    __version__ = "unknown"
__app_name__ = "LoRe Genome"
