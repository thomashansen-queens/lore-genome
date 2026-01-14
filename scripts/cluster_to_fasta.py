#!/usr/bin/env python
"""
Given a protein accession, collect it and all its co-clustered proteins from
MMseqs2 output and write them to a FASTA file.
"""
from pathlib import Path
import sys
import logging
import pandas as pd
from pipeline.summary import get_cluster_members

# from pipeline.config import load_config, PipelineConfig
from pipeline.utils import parse_fasta

def load_protein_report(report_path: Path) -> pd.DataFrame:
    """Protein report CSV from the pipeline."""
    try:
        df = pd.read_csv(report_path)
    except Exception as e:
        sys.exit(f"Error loading report file {report_path}: {e}")
    return df

# def get_cluster_members(report_df: pd.DataFrame, accession: str) -> list:
#     """
#     Given a protein accession, find its cluster and return all co-clustered accessions.
#     Parses comma-separated accessions (e.g., "WP_001, WP_002, WP_003").
#     """
#     cluster_row = report_df[report_df['protein_accessions'].astype(str).str.contains(accession, na=False)]
#     if cluster_row.empty:
#         sys.exit(f"Error: {accession} not found in the clusters report.")
#     # Parse comma-separated accessions
#     accessions_str = cluster_row['protein_accessions'].values[0]
#     accessions = [acc.strip() for acc in accessions_str.split(',')]
#     return accessions

def load_fasta(fasta_path: Path) -> dict:
    """FASTA of unique protein sequences."""
    try:
        with open(fasta_path, 'r', encoding='utf-8') as fasta_file:
            fasta_str = fasta_file.read()
        fasta_dict = parse_fasta(fasta_str)
    except Exception as e:
        sys.exit(f"Error loading FASTA file {fasta_path}: {e}")
    return fasta_dict

def write_cluster_fasta(
    accession: str,
    report_df: pd.DataFrame,
    fasta_dict: dict[str, str],
    output_path: Path,
) -> int:
    """
    Write the FASTA of the cluster containing the given accession to the output path.
    
    :param accession: Protein accession to find in the report.
    :param report_df: DataFrame containing protein clusters.
    :param fasta_dict: Dictionary mapping accessions to sequences.
    :param output_path: Path to write the output FASTA file.
    :return: Number of sequences written.
    """
    cluster_members = get_cluster_members(accession, report_df)
    cluster_dict = {acc: fasta_dict[acc] for acc in cluster_members if acc in fasta_dict}

    if not cluster_dict:
        sys.exit(f"No clustered accessions found for {accession}.")

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for header, sequence in cluster_dict.items():
            output_file.write(f">{header}\n{sequence}\n")
    members = get_cluster_members(report_df=report_df, accession=accession)
    custer_dict = {acc: fasta_dict[acc] for acc in members if acc in fasta_dict}
    if not custer_dict:
        raise ValueError(f"No cluster members found for accession: {accession}.")
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for header, sequence in custer_dict.items():
            output_file.write(f">{header}\n{sequence}\n")
    return len(custer_dict)

# def main(accession: str):
#     """Main entry point."""
#     # Load config from pipeline
#     config_dict = load_config('config.yaml')
#     config = PipelineConfig(**config_dict)
#     cache_dir = Path(config.download_dir.expanduser()) / sci_namer(config.taxons[0])
#     # Allow user input but default to config
#     report_path = Path(args.report) if args.report else cache_dir / "protein_report.csv"
#     protein_report = load_protein_report(report_path)
#     # load fasta, subset, and write to file
#     fasta_path = cache_dir / "proteins" / "unique_proteins.faa"
#     fasta = load_fasta(fasta_path)
#     # subset the fasta dictionary to only include the cluster members
#     logging.info("Loaded data, looking for cluster containing %s", args.accession)
#     cluster_members = get_cluster_members(protein_report, args.accession)
#     cluster_dict = {acc: fasta[acc] for acc in cluster_members if acc in fasta}
#     if not cluster_dict:
#         sys.exit("No clustered accessions found for the input accession.")
#     # write the cluster fasta to the output file
#     output_path = Path(args.output) if args.output else cache_dir / "proteins" / f"{args.accession}_cluster.faa"
#     with open(output_path, 'w', encoding='utf-8') as output_file:
#         for header, sequence in cluster_dict.items():
#             output_file.write(f">{header}\n{sequence}\n")
#     logging.info("%s sequences written to %s", len(cluster_dict), output_path)
