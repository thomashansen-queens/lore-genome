"""
Generates a report of clusters from pre-fetched data.
"""
import ast
from pathlib import Path
import pandas as pd
from pipeline.utils import sci_namer
from pipeline.summary import make_clusters_report
from pipeline.config import load_config, PipelineConfig

def csv_cell_to_list(cell: str) -> list:
    """
    Turn a string cell (like "['ACC1','ACC2']") into a Python list,
    or return [] for missing/blank cells.
    """
    if pd.isna(cell) or not cell.strip():
        return []
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        # Optionally log a warning here
        return []

def generate_clusters_report():
    """Entry point for generating the protein report from pre-fetched data."""
    # Load the configuration
    config_dict = load_config('config.yaml')
    config = PipelineConfig(**config_dict)
    organism = sci_namer(config.taxons[0])
    cache_dir = Path(config.download_dir.expanduser()) / organism
    # Load the data
    annotations = pd.read_csv(cache_dir / 'gene_annotations_filtered.csv')
    annotations['assemblies'] = annotations['assemblies'].apply(csv_cell_to_list)
    annotations['proteins__length'] = annotations['proteins__length'].fillna(0).astype(int)
    fasta = Path(cache_dir / 'proteins' / 'unique_proteins.faa').read_text(encoding='utf-8')
    clusters = pd.read_csv(cache_dir / 'proteins' / 'clustered.csv', names=["cluster","member"])
    # Generate the protein report
    summary_df = make_clusters_report(annotations, fasta, clusters)
    summary_df.to_csv(cache_dir / 'cluster_report.csv', index=False)
    print("Cluster report generated and saved!")
