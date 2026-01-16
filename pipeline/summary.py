"""
A module for number crunching, statistics, and visualization of genome data.
"""
import ast
import logging
from pathlib import Path
import pandas as pd
from pipeline.utils import parse_fasta

def make_clusters_report(
    annotations_df: pd.DataFrame,
    fasta_str: str,
    clusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate a human-readable report for clustered proteins."""
    fasta_dict = parse_fasta(fasta_str)
    # 1. Assign clusters to annotations
    summary_df = annotations_df.merge(
        clusters_df,
        left_on='proteins__accessionVersion',
        right_on='member',
        how='left',
    )
    summary_df = summary_df[summary_df['cluster'].notna()]
    # 2. Sort to keep most abundant sequence per cluster when drop (tie goes to longest)
    summary_df = summary_df.sort_values(
        by=['cluster', 'proteins__count', 'proteins__length'],
        ascending=False,
    )
    # 3. Sum counts and lists of assemblies for each cluster
    # First, ensure assemblies column contains lists (handle string storage from CSV)
    summary_df['assemblies'] = summary_df['assemblies'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    summed = summary_df.groupby('cluster')[['assemblies', 'proteins__count']].agg('sum')
    clustered_accessions = (
        summary_df
        .groupby('cluster')['proteins__accessionVersion']
        .agg(lambda x: list(set(x)))
        .reset_index()
        .rename(columns={'proteins__accessionVersion': 'protein_accessions'})
    )
    clustered_accessions['protein_count'] = clustered_accessions['protein_accessions'].apply(len)
    summary_df = summary_df.merge(clustered_accessions, on='cluster', how='left')
    # 4. Get the max and min length of proteins in each cluster
    length_stats = (
        summary_df
        .groupby('cluster')['proteins__length']
        .agg(max_length='max', min_length='min')
        .reset_index()
    )
    # 5. Aggregate assemblies for each cluster
    summed['assembly_count'] = summed['assemblies'].apply(len)
    summed = summed.rename(columns={
        'proteins__accessionVersion': 'protein_accessions',
        'assemblies': 'assembly_accessions',
        'proteins__count': 'instances_count',
    })
    summary_df = summary_df.drop(columns=['member', 'assemblies', 'proteins__count', 'chromosomes'])
    summed = summed.drop(columns=['instances_count']) # Comment this line to keep instances_count (helps identify gene duplication)
    # 6. Drop duplicates to show only one representative entry per cluster
    summary_df = summary_df.drop_duplicates(
        subset=['cluster'],
        keep='first',
    )
    summary_df = summary_df.merge(length_stats, on='cluster', how='left')
    summary_df.insert(5, 'min_length', summary_df.pop('min_length'))
    summary_df.insert(6, 'max_length', summary_df.pop('max_length'))
    summary_df = summary_df.merge(summed, on='cluster', how='left')
    # 7. Add FASTA sequences
    summary_df['sequence'] = summary_df['proteins__accessionVersion'].map(
        lambda x: fasta_dict.get(x, None)
    )
    # Sort by protein length
    summary_df = summary_df.sort_values(by=['proteins__length'], ascending=False)
    # Rename columns for clarity
    summary_df = summary_df.rename(columns={
        'proteins__accessionVersion': 'top_accession',
        'proteins__length': 'top_length',
    })
    # De-Python-ify the lists for human readable CSV output
    summary_df['protein_accessions'] = summary_df['protein_accessions'].str.join(',')
    summary_df['assembly_accessions'] = summary_df['assembly_accessions'].str.join(',')
    return summary_df


def get_cluster_members(accession: str, report_df: pd.DataFrame) -> list:
    """
    Given a protein accession, find its cluster and return all co-clustered accessions.
    :param accession: Protein accession to look up
    :param report_df: DataFrame version of the clusters report
    :return: List of co-clustered accessions
    """
    if accession is None or accession.strip() == "":
        raise ValueError("No accession provided.")
    if len(accession.split('.')[0].lstrip("WP_")) < 9:  # NCBI WP_* accs are 9 digits
        logging.warning("Provided accession '%s' looks incomplete!", accession)
    cluster_row = report_df[report_df['protein_accessions'].astype(str).str.contains(accession, na=False)]
    if cluster_row.empty:
        raise ValueError(f"{accession} not found in the clusters report.")
    if len(cluster_row) > 1:
        raise ValueError(f"{accession} found in multiple clusters. Input accession is ambiguous!")
    # Parse comma-separated accessions
    accessions_str = cluster_row['protein_accessions'].values[0]
    accessions = [acc.strip() for acc in accessions_str.split(',')]
    return accessions


def build_cluster_details(
    accession: str,
    cache_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Tabulate all proteins in the cluster containing `accession`. Loads and
    returns raw annotation data from per-genome cache files for downstream use.

    :param accession: The protein accession (WP_*) to look up
    :param cache_dir: Path to cache directory
    :param context: Number of flanking proteins to include
    :return: DataFrame with columns: index, protein_accession, assembly_accession, 
             nucleotide_accession, locusTag, protein_name, protein_length, begin, end, orientation
             and a list of annotation DataFrames from each assembly (keys: anchor accessions)
    """
    # Load the clusters report to find cluster members
    clusters_report_path = cache_dir / "clusters_report.csv"
    if not clusters_report_path.exists():
        raise ValueError(f"Clusters report not found at {clusters_report_path}. Run 'lore report' first.")

    clusters_report = pd.read_csv(clusters_report_path)
    cluster_members = get_cluster_members(accession, clusters_report)
    logging.info("Found cluster with %s proteins including %s", len(cluster_members), accession)

    # The assembly_accessions column contains all assemblies for this cluster
    cluster_info = clusters_report[
        clusters_report['protein_accessions'].astype(str).str.contains(accession, na=False)
    ]
    assemblies = [acc.strip() for acc in cluster_info['assembly_accessions'].values[0].split(',')]
    logging.info("Cluster spans %s assemblies:\n%s", len(assemblies), ', '.join(assemblies))

    # Load annotation data from each assembly's cache file
    annotation_dir = cache_dir / "genome_annotation"
    all_protein_data = []
    annotation_dfs = {}

    for assembly in assemblies:
        assembly = assembly.split('.')[0]  # Ignore version suffix
        annotation_file = annotation_dir / f"{assembly}.csv"
        if not annotation_file.exists():
            logging.warning("Annotation file not found: %s", annotation_file)
            continue

        # Load the annotation data for this assembly
        ann_df = pd.read_csv(annotation_file)
        annotation_dfs[assembly] = ann_df

        # Filter to just the cluster members
        matching = ann_df[ann_df['proteins__accessionVersion'].isin(cluster_members)]

        if not matching.empty:
            all_protein_data.append(matching)

    if not all_protein_data:
        raise ValueError(f"No annotation data found for cluster members in {annotation_dir}")

    # Combine all protein data
    cluster_annotations = pd.concat(all_protein_data, ignore_index=True)
    logging.info("Retrieved annotations for %s proteins from %s assemblies",
                 len(cluster_annotations), len(all_protein_data))

    # Map NCBI column names to output column names (__ from flattened JSON)
    column_map = {
        'proteins__accessionVersion': 'protein_accession',
        'annotations__assemblyAccession': 'assembly_accession',
        'genomicRegions__geneRange__accessionVersion': 'nucleotide_accession',
        'proteins__name': 'protein_name',
        'proteins__length': 'protein_length',
        'begin': 'begin',
        'end': 'end',
        'orientation': 'orientation',
        'locusTag': 'locusTag',
    }
    # Select available columns and rename
    available_cols = [col for col in column_map if col in cluster_annotations.columns]
    detail_df = cluster_annotations[available_cols].copy()
    detail_df = detail_df.rename(columns={col: column_map[col] for col in available_cols})

    # Add index column
    detail_df.insert(0, 'index', range(1, len(detail_df) + 1))

    return detail_df, annotation_dfs


def make_genome_context_row(
    anchor_acc: str,
    ann_df: pd.DataFrame,
    context: int,
) -> pd.DataFrame:
    """
    Add context columns to the detail DataFrame, showing flanking proteins.

    :param anchor_acc: Anchor protein accession
    :param annotation_df: Cached annotation dataframe (keys: anchor accessions)
    :param context: Number of flanking proteins to include
    :return: Single-row DataFrame with context columns added
    """
    # Define direction by anchor gene's 5'->3' direction
    orientation = ann_df.loc[ann_df['proteins__accessionVersion'] == anchor_acc, 'orientation'].iloc[0]
    ann_df = ann_df.reset_index(drop=True)
    if orientation == "plus":
        three_sign, five_sign = -1, +1
    else:
        three_sign, five_sign = +1, -1
    # Find the row for the anchor protein
    anchor_idx = ann_df.index[ann_df['proteins__accessionVersion'] == anchor_acc]
    if len(anchor_idx) != 1:
        raise ValueError(f"{anchor_acc} not uniquely found after filtering")
    anchor_idx = int(anchor_idx.values[0])
    # Get context proteins as rows
    rows = []
    for i in range(1, context + 1):
        # .shift rotates DataFrame keeping the index
        three = ann_df.shift(three_sign * i).iloc[anchor_idx]
        five  = ann_df.shift(five_sign  * i).iloc[anchor_idx]
        rows.append({'anchor': anchor_acc, 'side': 'threeprime', 'i': i,
                     'acc': three.get('proteins__accessionVersion', pd.NA),
                     'name': three.get('name', pd.NA),
                     'begin': three.get('begin', pd.NA),
                     'end': three.get('end', pd.NA),
                     'orientation': three.get('orientation', pd.NA)})
        rows.append({'anchor': anchor_acc, 'side': 'fiveprime', 'i': i,
                     'acc': five.get('proteins__accessionVersion', pd.NA),
                     'name': five.get('name', pd.NA),
                     'begin': five.get('begin', pd.NA),
                     'end': five.get('end', pd.NA),
                     'orientation': five.get('orientation', pd.NA)})
    # pandas magic to pivot rows into columns
    context_df = pd.DataFrame(rows)
    context_df = context_df.set_index(['anchor', 'side', 'i']).unstack(level=['side', 'i'])
    flat_df = context_df.copy()
    flat_df.columns = [f'{side}_{i}_{field}' for field, side, i in flat_df.columns]
    flat_df = flat_df.reset_index(drop=False)
    flat_df['assembly_acc'] = ann_df.loc[anchor_idx, 'annotations__assemblyAccession']
    return flat_df


def ordered_columns(df: pd.DataFrame, context: int) -> list[str]:
    """Helper to order columns in the output DataFrame."""
    base_cols = ['index', 'protein_accession', 'assembly_accession', 'nucleotide_accession',
                 'protein_name', 'protein_length', 'begin', 'end', 'orientation', 'locusTag']
    fields = ['acc', 'name', 'begin', 'end', 'orientation']
    sides = ['fiveprime', 'threeprime']
    cols = [c for c in base_cols if c in df.columns]
    for side in sides:
        for i in range(1, context + 1):
            for field in fields:
                name = f"{side}_{i}_{field}"
                if name in df.columns:
                    cols.append(name)
    # catch-all for anything unexpected; shouldn't be needed
    extras = [c for c in df.columns if c not in cols]
    return cols + extras


def build_cluster_context(
    cluster_df: pd.DataFrame,
    annotation_dfs: dict[str, pd.DataFrame],
    context: int = 3,
) -> pd.DataFrame:
    """
    Add 5' and 3' genome context to cluster details table.
    
    :param accession: Protein accession to look up
    :param context: Number of flanking genes to include
    :return: Number of rows written
    """
    # Collect context rows for each protein in the cluster
    context_rows = []
    for anchor_acc, assembly_acc in cluster_df[['protein_accession', 'assembly_accession']].itertuples(index=False):
        assembly_acc = assembly_acc.split('.')[0]  # Ignore version suffix to match keys
        ann_df = annotation_dfs.get(assembly_acc)
        if ann_df is None:  # This should never happen unless files have moved
            logging.warning("No annotation data for assembly %s; skipping...", assembly_acc)
            continue
        context_rows.append(make_genome_context_row(anchor_acc, ann_df, context))
    # Assemble all context rows and merge to cluster details
    context_df = pd.concat(context_rows, ignore_index=True)
    cluster_df = cluster_df.merge(context_df,
                                  left_on=['protein_accession', 'assembly_accession'],
                                  right_on=['anchor', 'assembly_acc'],
                                  how='left')
    cluster_df = cluster_df.drop(columns=['anchor'])
    # Reorder columns
    ordered_cols = ordered_columns(cluster_df, context)
    cluster_df = cluster_df[ordered_cols]
    return cluster_df


def save_cluster_details_csv(
    accession: str,
    cache_dir: Path,
    context: int | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Write a detailed table of all proteins in the cluster containing `accession`,
    optionally adding genome context columns, to CSV.
    :param accession: The protein accession (WP_*) to look up
    :param cache_dir: Path to cache directory
    :param context: Number of flanking proteins to include
    :param output_path: Optional path to write the CSV file
    :return: DataFrame with cluster details
    """
    detail_df, annotation_dfs = build_cluster_details(accession, cache_dir)
    if context is not None and context > 0:
        detail_df = build_cluster_context(detail_df, annotation_dfs, context)
    if output_path is not None:
        detail_df.to_csv(output_path, index=False)
        logging.info("Wrote cluster details to %s with %s flanking genes",
                     output_path, context or 0)
    return detail_df
