"""
Module for Artifacts
"""

from .manager import TransferMode, ArtifactManager, normalize_sources
from .artifact import Artifact, ArtifactFile


__all__ = [
    "Artifact",
    "ArtifactFile",
    "TransferMode",
    "ArtifactManager",
    "normalize_sources",
]
