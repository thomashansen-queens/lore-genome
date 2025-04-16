#!/usr/bin/env python
"""
LoRe: A streamlined tool for filtering genomes and extracting protein data from NCBI.

This script provides a rudimentary command-line interface (CLI) for running the
LoRe pipeline.
"""
import logging
import sys
import click

from pipeline.config import load_config, PipelineConfig
from pipeline.pipeline import GenomePipeline
from pipeline.io_functions import load_or_fetch_df
from scripts import cluster_to_fasta

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
    # pass

@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to configuration file.")
def pipeline(config_path):
    """
    Recommended: Run the full genome processing pipeline.
    """
    config_obj = get_config(config_path)
    pipeline_obj = GenomePipeline(config_obj)
    pipeline_obj.run()
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
    genome_reports = load_or_fetch_df(genome_cache)
    genome_accessions = genome_reports["accession"].tolist()
    fasta_str = pipeline_obj.fetch_protein_fasta(genome_accessions)
    logging.info("Fetched %s unique protein sequences.", len(fasta_str.split(">")) - 1)

@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to configuration file.")
def cluster2fasta(config_path):
    """
    Writes a FASTA file of co-clustered proteins.
    """
    logging.info("This is the current config.yaml. Edit it before running the pipeline.")
    config_obj = get_config(config_path)
    logging.info(config_obj.model_dump_json(indent=2, exclude={"ncbi"}, exclude_defaults=True))  # doesn't work with NCBI object

@cli.command()
@click.option("--config-path", default="config.yaml", help="Path to configuration file.")
def config(config_path):
    """
    Show the current configuration.
    """
    logging.info("This is the current config.yaml. Edit it before running the pipeline.")
    config_obj = get_config(config_path)
    logging.info(config_obj.model_dump_json(indent=2, exclude={"ncbi"}, exclude_defaults=True))  # doesn't work with NCBI object

@cli.command()
def help():
    """
    Show the help page for LoRe.
    """
    click.echo(cli.get_help(click.Context(cli)))

if __name__ == "__main__":
    # Setup basic logging if needed.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()
