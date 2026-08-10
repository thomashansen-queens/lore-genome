"""
Module for Artifacts
"""

from .manager import TransferMode, ArtifactManager, normalize_sources
from .artifact import Artifact, ArtifactPathBundle, ArtifactFile


__all__ = [
    "Artifact",
    "ArtifactPathBundle",
    "ArtifactFile",
    "TransferMode",
    "ArtifactManager",
    "normalize_sources",
]
