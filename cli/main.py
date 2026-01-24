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
from pipeline.summary import save_cluster_details_csv
from pipeline.viz import save_cluster_svg
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
    logging.info("Fetched %s genome reports.",
                 format(len(genome_reports), ','))


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
@click.option("--config-path", default="config.yaml", help="Path to configuration file.")
def report(config_path):
    """
    Step 3: Generate clusters report from cached data (annotations, FASTA, clusters).
    Useful for stepwise execution or regenerating the report after manual changes.
    Fetches gene annotations and clusters proteins if missing.
    """
    config_obj = get_config(config_path)
    pipeline_obj = GenomePipeline(config_obj)

    # Get genome accessions from sampled or filtered reports
    genome_cache = pipeline_obj.cache_dir / "genome_reports_sampled.csv"
    if not genome_cache.exists():
        logging.info("No sampled genomes found at %s; using filtered list.", genome_cache)
        genome_cache = pipeline_obj.cache_dir / "genome_reports_filtered.csv"
    if not genome_cache.exists():
        logging.error("No genome reports found. Run 'lore genomes' first.")
        sys.exit("Couldn't find genome_reports. Exiting.")
    genome_reports = load_or_fetch_df(cache_path=genome_cache, allow_fetch=False)
    genome_accessions = genome_reports["accession"].tolist()

    # Fetch gene annotations if missing
    annotations_cache = pipeline_obj.cache_dir / "gene_annotations_filtered.csv"
    if not annotations_cache.exists():
        logging.info("Gene annotations not found; fetching from NCBI...")
        annotations_df = pipeline_obj.fetch_gene_annotations(genome_accessions)
    else:
        logging.info("Loading gene annotations from %s", annotations_cache)
        annotations_df = pd.read_csv(annotations_cache)

    # Check for protein FASTA
    proteins_cache = pipeline_obj.cache_dir / "proteins" / "unique_proteins.faa"
    if not proteins_cache.exists():
        logging.info("Protein FASTA not found; fetching from NCBI...")
        fasta_str = pipeline_obj.fetch_protein_fasta(genome_accessions)
    logging.info("Loading protein FASTA from %s", proteins_cache)
    fasta_str = proteins_cache.read_text(encoding="utf-8")

    # Trim proteins and cluster if needed
    trimmed_fasta, cluster_label = pipeline_obj.trim_proteins(fasta_str)

    # Check for cluster file; cluster if missing
    tail = "N" if config_obj.cluster_residues > 0 else "C"
    clustered_by = tail + str(abs(config_obj.cluster_residues))
    clusters_cache = pipeline_obj.cache_dir / "proteins" / f"{clustered_by}_clustered.csv"
    if not clusters_cache.exists():
        logging.info("Cluster file not found; running MMseqs2 clustering...")
        clusters_df = pipeline_obj.cluster_proteins(cluster_label)
    logging.info("Loading clusters from %s", clusters_cache)
    clusters_df = pd.read_csv(clusters_cache)

    # Generate report
    summary_df = pipeline_obj.make_report(annotations_df, fasta_str, clusters_df)
    logging.info("Clusters report saved to %s", pipeline_obj.cache_dir / "clusters_report.csv")
    logging.info("Report contains %s protein clusters.",
                 format(len(summary_df), ','))


@cli.command()
@click.argument("accession", type=str)
@click.option(
    "-r", "--report",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to the pipeline's clusters_report.csv file (defaults to cache).",
)
@click.option(
    "-o", "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Where to write the cluster FASTA (default to <accession>_cluster.faa in cache).",
)
@click.option("--config-path", default="config.yaml", help="Path to configuration file.")
def cluster2fasta(accession: str, report: Path, output: Path, config_path: Path):
    """
    Writes a FASTA file of co-clustered proteins.
    """
    config_obj = get_config(config_path)
    basepath = Path(config_obj.download_dir).expanduser() / sci_namer(config_obj.taxons[0])
    # Use the clusters report to find clusters
    # Note for the future: This could also just use mmseqs2 output
    if report is None:
        report = basepath / "clusters_report.csv"
    if not report.exists():
        logging.error("Clusters report not found at %s. Run the pipeline first.", report)
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
    logging.info("Wrote %s sequences to %s",
                 format(size, ','), output)


@cli.command()
@click.argument("accession", type=str)
@click.option(
    "-o", "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Where to write the cluster details CSV (default: <accession>_details.csv in cache).",
)
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config.yaml",
    show_default=True,
    help="Path to configuration file.",
)
@click.option(
    "-c", "--context",
    default=3,
    type=int,
    help="Number of flanking genes to include in the annotation context.",
)
def inspect(accession: str, output: Path | None, config_path: Path, context: int):
    """
    Generate a detailed report of all proteins in the cluster containing ACCESSION.
    Includes protein accession, assembly, nucleotide, locus tag, name, length, coordinates.
    """
    config_obj = get_config(config_path)
    basepath = Path(config_obj.download_dir).expanduser() / sci_namer(config_obj.taxons[0])
    cache_dir = basepath

    # Generate and write cluster details (loads annotations from per-genome files)
    if output is None:
        output = cache_dir / f"{accession}_details.csv"

    detail_df = save_cluster_details_csv(
        accession=accession,
        cache_dir=cache_dir,
        context=context,
        output_path=output.expanduser(),
    )
    logging.info("Cluster details report contains %s proteins", format(len(detail_df), ','))


@cli.command()
@click.argument("accession", type=str)
@click.option(
    "-o", "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Where to write the cluster SVG (default: <accession>_cluster.svg in cache).",
)
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config.yaml",
    show_default=True,
    help="Path to configuration file.",
)
@click.option(
    "-c", "--context",
    default=3,
    type=int,
    help="Number of flanking genes to include in the annotation context.",
)
@click.option(
    "-l", "--layout",
    type=click.Choice(["bp", "clamped", "order"], case_sensitive=False),
    default="bp",
    show_default=True,
    help="Layout style for genomic neighborhood visualization (bp: scaled by base pair, clamped: limits gaps and gene sizes, order: evenly spaced)."
)
@click.option(
    "--max-gap",
    type=float,
    default=100.0,
    show_default=True,
    help="Maximum gap size (in bp) to display between genes when using 'clamped' layout.",
)
@click.option(
    "--clamp-genes/--no-clamp-genes",
    default=False,
    show_default=True,
    help="Whether to clamp gene sizes when using 'clamped' layout.",
)
def neighborhood(accession: str, output: Path | None, config_path: Path, context: int,
                 layout: str, max_gap: float, clamp_genes: bool):
    """
    Make a diagram showing the genomic neighborhood for a protein.
    Saved as an SVG file.
    """
    config_obj = get_config(config_path)
    basepath = Path(config_obj.download_dir).expanduser() / sci_namer(config_obj.taxons[0])
    cache_dir = basepath
    # check layout options
    if layout != "clamped" and (max_gap != 100.0 or clamp_genes):
        logging.warning("--max-gap/--clamp-genes only apply when --layout=clamped")
    # Generate and write cluster details (loads annotations from per-genome files)
    if output is None:
        acc = accession.split('.')[0]
        output = cache_dir / f"{acc}_neighborhood_{layout}.svg"

    _ = save_cluster_svg(
        accession=accession,
        cache_dir=cache_dir,
        context=context,
        layout=layout,
        output_path=output.expanduser(),
        max_gap=max_gap,
        clamp_genes=clamp_genes,
    )
    logging.info("Wrote cluster genomic neighbourhood SVG to %s", output)


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
