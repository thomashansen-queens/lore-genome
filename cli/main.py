#!/usr/bin/env python
"""
LoRe: A streamlined tool for filtering genomes and extracting protein data from NCBI.

This script provides a rudimentary command-line interface (CLI) for running the
LoRe pipeline.
"""
import logging
from pathlib import Path
import sys
import click
import pandas as pd

from pipeline.caching import load_or_fetch_df
from pipeline.config import load_config, PipelineConfig
from pipeline.pipeline import GenomePipeline
from pipeline.utils import sci_namer, parse_fasta
from scripts import write_cluster_fasta

def get_config(config_path):
    """Uses a common configuration file."""
    config_dict = load_config(config_path)
    config_obj = PipelineConfig(**config_dict)
    return config_obj


@click.group()
def cli():
    """
    LoRe: A tool to filter and get protein information from genomes via NCBI.
    """
    # Setup basic logging if needed.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@cli.command()
@click.option(
    "-f", "--force",
    is_flag=True,
    default=False,
    help="Force re-fetching every step, ignoring any existing cache."
)
@click.option(
    "--config-path", default="config.yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to pipeline config."
)
def pipeline(force: bool, config_path: Path):
    """
    Recommended: Run the full genome processing pipeline.
    """
    config_obj = get_config(config_path)
    pipeline_obj = GenomePipeline(config_obj)
    pipeline_obj.run(force=force)
    click.echo("Full pipeline execution completed.")


@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to configuration file.")
def genomes(config_path):
    """
    Step 1: Fetch and filter genome reports from NCBI.
    """
    config_obj = get_config(config_path)
    pipeline_obj = GenomePipeline(config_obj)
    genome_reports = pipeline_obj.fetch_and_filter_genomes()
    logging.info("Fetched %s genome reports.", len(genome_reports))


@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to configuration file.")
def sample(config_path):
    """
    Step 1b: Make representative sample from genomes.
    """
    config_obj = get_config(config_path)
    pipeline_obj = GenomePipeline(config_obj)
    genome_reports = pipeline_obj.fetch_and_filter_genomes(allow_fetch=False)
    sampled_genomes = pipeline_obj.sample_genomes(genome_reports)
    logging.info("Sample contains %s genome reports.", len(sampled_genomes))


@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to configuration file.")
def proteins(config_path):
    """
    Step 2: Fetch unique proteins from a set of genomes.
    """
    config_obj = get_config(config_path)
    pipeline_obj = GenomePipeline(config_obj)
    genome_cache = pipeline_obj.cache_dir / "genome_reports_sampled.csv"
    if not genome_cache.exists():
        logging.info("No sampled genomes found at %s; using filtered list.", genome_cache)
        genome_cache = pipeline_obj.cache_dir / "genome_reports_filtered.csv"
    if not genome_cache.exists():
        logging.error("No filtered genomes found at %s.", genome_cache)
        sys.exit("Couldn't find genome_reports. Exiting.")
    genome_reports = load_or_fetch_df(cache_path=genome_cache, allow_fetch=False)
    genome_accessions = genome_reports["accession"].tolist()
    fasta_str = pipeline_obj.fetch_protein_fasta(genome_accessions)
    logging.info("Fetched %s unique protein sequences.", len(fasta_str.split(">")) - 1)


@cli.command()
@click.argument("accession", type=str)
@click.option(
    "-r", "--report",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to the pipeline's protein_report.csv file (defaults to cache).",
)
@click.option(
    "-o", "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Where to write the cluster FASTA (default to <accesison>_cluster.faa in cache).",
)
@click.option("--config-path", default="config.yaml", help="Path to configuration file.")
def cluster2fasta(accession: str, report: Path, output: Path, config_path: Path):
    """
    Writes a FASTA file of co-clustered proteins.
    """
    config_obj = get_config(config_path)
    basepath = Path(config_obj.download_dir).expanduser() / sci_namer(config_obj.taxons[0])
    # Use the protein report to find clusters
    # Note for the future: This could also just use mmseqs2 output
    if report is None:
        report = basepath / "protein_report.csv"
    if not report.exists():
        logging.error("Protein report not found at %s. Run the pipeline first.", report)
        sys.exit(1)
    report_df = pd.read_csv(report)
    if output is None:
        output = basepath / f"{accession}_cluster.faa"
    # Load big fasta file
    fasta_path = basepath / "proteins" / "unique_proteins.faa"
    if not fasta_path.exists():
        logging.error("Protein FASTA not found at %s. Run the pipeline first.", fasta_path)
        sys.exit(1)
    fasta_dict = parse_fasta(fasta_path.read_text(encoding="utf-8"))
    logging.info("Looking up cluster for accession: %s", accession)
    size = write_cluster_fasta(
        accession=accession,
        report_df=report_df,
        fasta_dict=fasta_dict,
        output_path=output.expanduser(),
    )
    logging.info("Wrote %s sequences to %s", size, output)


@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to configuration file.")
def config(config_path):
    """
    Show the current configuration.
    """
    logging.info("This is the current config.yaml. Edit it before running the pipeline.")
    config_obj = get_config(config_path)
    logging.info(
        config_obj.model_dump_json(indent=2, exclude={"ncbi"}, exclude_defaults=True)
    )  # doesn't work with NCBI object


if __name__ == "__main__":
    cli()
