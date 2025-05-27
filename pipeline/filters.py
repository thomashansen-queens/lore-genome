"""Filtering various DataFrames fetched from NCBI Datasets API"""

import logging
import re
from typing import Sequence, Union
import pandas as pd

from pipeline.config import PipelineConfig

def filter_genome_reports(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Filters atypical and duplicated genomes from NCBI Datasets genome reports.
    Preferentially keeps GCF over GCA.

    :param df (DataFrame): DataFrame containing all API results.
    :return DataFrame: Filtered DataFrame.
    """
    # If post-API filtereing is enabled, filter by search terms
    if config.search_stage == 'post':
        logging.info("Filtering genome reports by search terms...")
        df = filter_by_search_terms(df, config.search_terms)
    # Typical genomes have no data in these columns
    if 'assembly_info__atypical__is_atypical' in df.columns:
        # This column contains either TRUE or null/NaN values, so fillna
        mask = df['assembly_info__atypical__is_atypical'].fillna(False).astype(bool)
        atypical = df[mask]
        logging.info(
            "Found %s atypical records. Filtering out:\n%s",
            len(atypical), atypical['accession'].to_list()
        )
        df = df[~mask]
    if 'assembly_info__genome_notes' in df.columns:
        mask = df['assembly_info__genome_notes'].notnull()
        notes = df[mask]
        logging.info(
            "Found %s records with notes. Filtering out:\n%s",
            {len(notes)}, notes['accession'].to_list()
        )
        logging.info("Notes:\n%s", notes['assembly_info__genome_notes'].to_list())
        df = df[~mask]
    # Preferentially keep GCF (RefSeq) genomes over GCA (GenBank)
    logging.info("Number of genome reports: %s", len(df))
    gca_genomes = df[df['accession'].str.startswith('GCA')]
    gcf_genomes = df[df['accession'].str.startswith('GCF')]
    logging.info(
        "RefSeq (GCF): %s, GenBank (GCA): %s",
        len(gcf_genomes), len(gca_genomes)
    )
    gcf_pairs = df[df['accession'].str.startswith("GCF")]['paired_accession']
    duplicated = df[df['accession'].isin(gcf_pairs)]
    logging.info("GCA genomes already in GCF (RefSeq): %s", len(duplicated))
    deduplicated_df = df[~df['accession'].isin(gcf_pairs)]
    if len(config.taxons[0].split(' ')) == 1:  # Ignore ANI match if only a genus is provided
        logging.info("Taxon assumed to be a genus. Ignoring best ANI match.")
    elif not deduplicated_df['average_nucleotide_identity__best_ani_match__organism_name'].isin(config.taxons).all():
        logging.warning(
            "Some assemblies' average nucleotide identity (ANIs) are best matched \
            to a different organism and will be removed. Matched organisms: %s",
            deduplicated_df['average_nucleotide_identity__best_ani_match__organism_name'].unique()
        )
        deduplicated_df = deduplicated_df[deduplicated_df['average_nucleotide_identity__best_ani_match__organism_name'].isin(config.taxons)]
    logging.info("Number of genome reports after filtering: %s", len(deduplicated_df))
    return deduplicated_df


def filter_by_search_terms(
        df: pd.DataFrame,
        search_terms: Sequence[Union[str, Sequence[str]]] | None = None,
        columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Filter a DataFrame down to rows where *any* of the search_terms appear
    (case-insensitive) in *any* of the given columns (word only, no substrings).

    :param df: DataFrame to filter.
    :param columns: List of column names (strings) to search in.
    :param search_terms: List of substrings to match (using OR logic).
    :returns: DataFrame containing search terms.
    """
    if not search_terms or search_terms == ['']:
        return df
    if not columns or columns == ['']:
        columns = df.columns.tolist()

    mask = pd.Series(False, index=df.index)
    for search_term_set in search_terms:
        if isinstance(search_term_set, str):
            search_term_set = [search_term_set]
        set_mask = pd.Series(True, index=df.index)
        for term in search_term_set:
            term_mask = pd.Series(False, index=df.index)
            pattern = rf"(?i)\b{re.escape(term)}\b"   # case-insensitive
            for col in columns:
                if col in df.columns:
                    term_mask |= df[col].astype(str).str.contains(pattern, na=False)
            set_mask &= term_mask  # AND logic (all terms must match)
        logging.info("%s entries match search term set %s", set_mask.sum(), search_term_set)
        prev_num_hits = mask.sum()
        mask |= set_mask  # OR logic (include matches from each search term set)
        logging.info("%s hits total (+%s)", mask.sum(), mask.sum() - prev_num_hits)
    return df[mask]


def filter_gene_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters gene annotations to keep only those that match the specified taxon.

    :param df (DataFrame): DataFrame containing all API results.
    :param config (PipelineConfig): Configuration object containing taxon IDs.
    :return DataFrame: Filtered DataFrame.
    """
    # Get summary data including duplicated entries, then deduplicate
    counts = (
        df
        .groupby('proteins__accessionVersion')
        .size()
        .rename('proteins__count')
        .reset_index()
    )
    accession_sources = (
        df
        .groupby('proteins__accessionVersion')['annotations__assemblyAccession']
        .agg(lambda x: list(set(x)))  # or ','.join(set(x)) if you want strings
        .reset_index()
        .rename(columns={'annotations__assemblyAccession': 'assemblies'})
    )
    filtered_df = df.drop_duplicates(
        subset=['proteins__accessionVersion'],
        keep='first'
    )
    filtered_df = filtered_df.merge(
        counts, on='proteins__accessionVersion', how='left',
    )
    filtered_df = filtered_df.merge(
        accession_sources, on='proteins__accessionVersion', how='left',
    )
    filtered_df = filtered_df.drop(columns=['locusTag', 'proteins__name',
        'annotations__assemblyAccession', 'genomicRegions__geneRange__accessionVersion',
        'begin', 'end', 'orientation',
    ])
    logging.info("Filtered gene annotations: %s", format(len(filtered_df),','))
    return filtered_df
