"""
Configuration for the protein classification pipeline.
"""

from pathlib import Path
import logging
import yaml
from pydantic import BaseModel, Field, field_validator
from ncbi.datasets.openapi.configuration import Configuration as NcbiConfig
from ncbi.datasets.openapi.models.v2reports_assembly_level import V2reportsAssemblyLevel

class PipelineConfig(BaseModel):
    """
    Configuration for the protein classification pipeline. Contains all the
    necessary parameters to fetch and process genome data from NCBI.
    """
    download_dir: Path
    # NCBI Datasets API
    ncbi: NcbiConfig
    page_size: int = Field(50, description="Number of records to fetch per API call")
    # Genome reports
    taxons: list[str]
    assembly_level: list[V2reportsAssemblyLevel]
    refseq_only: bool = Field(False, description="Whether to fetch only 'refseq' source records")
    search_terms: list[list[str]] = Field([['']], description="Terms to include in genome search.")
    sampling_strategy: list[str] = Field(
        default_factory=lambda: ["default"],
        description="List of columns to use for stratified sampling. "
                    "Can use 'default' for geographic location and collection date, "
                    "an empty list or 'none' to ignore diveristy (random sampling)."
    )
    # MMSeqs2
    cluster_residues: int = Field(0, description="Number of residues to consider for clustering (negative for tail)")
    genome_limit: int = Field(0, description="Limit the number of genomes to fetch annotations for (0 for no limit)")
    mmseqs: str = Field('mmseqs', description="Path to mmseqs2 executable")
    min_seq_id: float = Field(0.9, description="Minimum sequence identity for clustered proteins")

    @field_validator('taxons', 'assembly_level', 'sampling_strategy', mode='before')
    @classmethod
    def ensure_list(cls, value) -> list[str]:
        """Convert strings to lists in the config."""
        if isinstance(value, str):
            values = value.split(',')
            return [sub.strip() for sub in values]
        return value

    @field_validator('search_terms', mode='before')
    @classmethod
    def parse_search_terms(cls, value) -> list:
        """Converts search_terms to a list of lists."""
        if isinstance(value, str):  # only one set of search terms
            search_term_set = [term.strip() for term in value.split(",")]
            return [search_term_set]
        elif isinstance(value, list):
            # Convert each set of search terms to a list
            search_term_sets = []
            for search_term_set in value:
                if isinstance(search_term_set, str):
                    search_term_sets.append([sub.strip() for sub in search_term_set.split(",")])
                elif isinstance(search_term_set, list):
                    search_term_sets.append(search_term_set)
                else:
                    raise ValueError("search_terms must be a string or list of strings.")
            return search_term_sets
        raise ValueError("search_terms must be a list or a string")

    @field_validator('ncbi', mode='before')
    @classmethod
    def validate_ncbi(cls, value):
        """Convert ncbi config to NcbiConfig object from Datasets openapi."""
        if isinstance(value, dict):
            # retries_value = v.pop("retries", None)
            obj = NcbiConfig(**value)
            # obj.retries = retries_value if retries_value is not None else 3
            return obj
        return value

    model_config = {'arbitrary_types_allowed': True}


def load_config(config_path="config.yaml"):
    """Load configuration from a YAML file."""
    logging.info("Loading config object from %s...", config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
