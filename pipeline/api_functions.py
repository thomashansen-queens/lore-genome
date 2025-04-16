"""
This module contains wrapper functions for the NCBI Datasets API. It saves results as
DataFrames for easy processing and human readability.
"""
import io
import functools
import http.client
import logging
from time import sleep
import zipfile
from ncbi.datasets.openapi import GenomeApi
import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.utils import unwrap_single_value, parse_fasta

def retry(exceptions, tries=3, delay=1.75, logger=None):
    """
    A decorator that allows API calls to retry a set number of times before failing.
    :param exceptions: The exception(s) to catch and retry on.
    :param tries: The number of times to retry the function.
    :param delay: The delay between retries (exponentially increasing).
    :param logger: The logger to use for messages.
    :return: The result of the function call.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    sleeptime = delay ** attempt
                    msg = f'{e}, Retrying in {sleeptime} seconds...'
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    sleep(sleeptime)
            if logger:
                logger.error("Failed to execute %s after %s attempts.", func.__name__, tries)
        return wrapper
    return decorator


def get_genome_reports(api: GenomeApi, config: PipelineConfig) -> pd.DataFrame:
    """
    Fetch genome reports from NCBI Datasets API for a given set of taxons and search terms.
    This could be more accurately called an "assembly report" as it contains metadata about
    the genome assembly, not the genome itself.
    """
    taxons = config.taxons
    assembly_level = config.assembly_level
    search_terms = config.search_terms
    page_size = config.page_size
    refseq_only = config.refseq_only

    merged_df = pd.DataFrame()  # Initialize empty DataFrame to store results
    for term_set in search_terms:
        page_token = None
        page = 0
        while True:
            logging.info(
                "Fetching genome reports for taxons %s with search terms: %s",
                taxons, term_set
            )
            if page_token:
                logging.info("Page %s, page token %s", page, page_token)
            try:
                result = api.genome_dataset_reports_by_taxon(
                    taxons=taxons,
                    filters_assembly_level=assembly_level,
                    filters_search_text=term_set,
                    filters_assembly_source='refseq' if refseq_only else 'all',
                    page_size=page_size,
                    page_token=page_token,
                )
            except Exception as e:
                logging.error(
                    "Error fetching genome reports for search terms: %s: %s. Continuing...",
                    term_set, e,
                )
                break

            if result.reports is None:
                logging.warning("No results found for search terms: %s.", term_set)
                break

            reports = [report.to_dict() for report in result.reports]
            # NCBI uses _ for space, so __ is used here to separate nested keys
            df = pd.json_normalize(reports, sep='__')

            if not merged_df.empty and 'accession' in df.columns and 'accession' in merged_df.columns:
                df = df[~df['accession'].isin(merged_df['accession'])]
            logging.info("Found %s total results for: %s.", result.total_count, term_set)
            logging.info("New results: %s", len(df))

            merged_df = pd.concat([merged_df, df], ignore_index=True)
            if result.next_page_token:
                page += 1
                page_token = result.next_page_token
            else:
                break

    logging.info("Total unique genome reports fetched: %s.", len(merged_df))
    return merged_df


@retry(http.client.IncompleteRead, logger=logging.getLogger(__name__))
def get_genome_annotation_package(api: GenomeApi, accession: str) -> pd.DataFrame:
    """
    Fetch genome annotation package from NCBI Datasets API for a given accession.
    This contains one entry per gene in the genome.
    """
    logging.info("Downloading annotation package for %s...", accession)
    result = api.download_genome_annotation_package(
        accession=accession,
        filename=f'{accession}.zip',
    )
    df = unzip_dataframe(result, accession)
    genomic_regions = df['genomicRegions__geneRange__range'].apply(pd.Series)
    genomic_regions.columns = ['begin', 'end', 'orientation']
    df = df.drop(columns=['genomicRegions__geneRange__range', 'proteins'])
    df = df.join(genomic_regions)
    return df


def unzip_dataframe(result: bytes, accession: str) -> pd.DataFrame:
    """Process (in memory) a ZipFile object returned by the API."""
    with zipfile.ZipFile(io.BytesIO(result)) as zip_file:
        # should be found at 'ncbi_dataset/data/data_report.jsonl'
        jsonl_files = [f for f in zip_file.namelist() if f.endswith('.jsonl')]
        if not jsonl_files:
            logging.error("No .jsonl found in .zip for accession %s", accession)
            logging.error("Files found: %s", zip_file.namelist())
            return pd.DataFrame()
        elif len(jsonl_files) > 1:
            logging.error("Multiple .jsonl files found in .zip for accession %s", accession)
            return pd.DataFrame()
        with zip_file.open(jsonl_files[0]) as jsonl_file:
            df = pd.read_json(jsonl_file, lines=True)
    logging.info("Made DataFrame for %s, length %s", accession, format(len(df),','))

    # Unwrap single-value columns. This actually needs to be done twice for a clean table!
    for col in df.columns:
        df[col] = df[col].apply(unwrap_single_value)
    df = pd.json_normalize(df.to_dict(orient='records'), sep='__')
    for col in df.columns:
        df[col] = df[col].apply(unwrap_single_value)
    df = df.convert_dtypes()
    return df


@retry(http.client.IncompleteRead, logger=logging.getLogger(__name__))
def get_genome_proteins(api: GenomeApi, accessions: list, chunk_size: int = 50) -> str:
    """
    Fetch protein FASTA package from NCBI Datasets API for a given list of accessions.
    In cases where a protein exists in multiple accessions, it will not be duplicated.
    Splits accession list into chunks to avoid API limits.
    :param api: NCBI Datasets API object.
    :param accessions: List of genome accessions to fetch proteins for.
    :param chunk_size: Number of accessions to fetch at once.
    :return: Concatenated FASTA string of all proteins (>header\\nsequence)
    """
    accession_chunks = [
        accessions[i:i+chunk_size] for i in range(0, len(accessions), chunk_size)
    ]
    unique_proteins = {}
    for c, chunk in enumerate(accession_chunks):
        if len(accession_chunks) > 1:
            logging.info("Fetching protein FASTA package for %s accessions (chunk %s of %s)...",
                         len(chunk), c, len(accession_chunks))
        response = api.download_assembly_package(
            accessions=chunk,
            include_annotation_type=['PROT_FASTA'],
            hydrated='FULLY_HYDRATED',
        )
        with zipfile.ZipFile(io.BytesIO(response)) as zip_file:
            fasta_files = [f for f in zip_file.namelist() if f.endswith('.faa')]
            if not fasta_files:
                logging.error("No protein FASTA file found in the assembly package!")
                return ""
            logging.info("Found %s .faa files in assembly package. Processing...", len(fasta_files))
            if len(fasta_files) < len(chunk):
                logging.warning("Some accessions did not have a protein FASTA file.")
                logging.warning("Accessions with no protein FASTA: %s",
                    [a for a in accessions if f"{a}.faa" not in fasta_files])
            for fasta_file in fasta_files:
                n_start = len(unique_proteins)
                with zip_file.open(fasta_file) as fasta_handle:
                    fasta_content = fasta_handle.read().decode('utf-8')
                    chunk_proteins = parse_fasta(fasta_content)
                unique_proteins.update(chunk_proteins)
                n_end = len(unique_proteins)
                logging.info("Processed %s proteins, new unique protein total: %s (+%s)",
                    format(len(chunk_proteins),','), format(n_end,','), format(n_end-n_start,','))
    concatenated_fasta = [f">{header}\n{seq}" for header, seq in unique_proteins.items()]
    concatenated_fasta = "\n".join(concatenated_fasta)
    logging.info("Found %s proteins, (%s characters).",
        format(len(unique_proteins),','), format(len(fasta_content),','))
    return concatenated_fasta
