"""
Given a protein accession, output a detailed report of all proteins in the same cluster.
Assembles data from gene_annotations by looking up each co-clustered protein.
"""
import logging
from pathlib import Path
import sys
import pandas as pd

def get_cluster_members(report_df: pd.DataFrame, accession: str) -> list:
    """
    Given a protein accession, find its cluster and return all co-clustered accessions.
    Parses comma-separated accessions (e.g., "WP_001, WP_002, WP_003").
    """
    cluster_row = report_df[report_df['protein_accessions'].astype(str).str.contains(accession, na=False)]
    if cluster_row.empty:
        raise ValueError(f"{accession} not found in the clusters report.")
    # Parse comma-separated accessions
    accessions_str = cluster_row['protein_accessions'].values[0]
    accessions = [acc.strip() for acc in accessions_str.split(',')]
    return accessions

def build_cluster_details(
    accession: str,
    cache_dir: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Build a detailed table for all proteins in the cluster containing `accession`.
    Loads and returns raw annotation data from per-genome cache files.

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
        sys.exit(f"Clusters report not found at {clusters_report_path}. Run 'lore report' first.")

    clusters_report = pd.read_csv(clusters_report_path)
    cluster_members = get_cluster_members(clusters_report, accession)
    logging.info("Found cluster with %d proteins including %s", len(cluster_members), accession)

    # The assembly_accessions column contains all assemblies for this cluster
    cluster_info = clusters_report[
        clusters_report['protein_accessions'].astype(str).str.contains(accession, na=False)
    ]

    if 'assembly_accessions' in cluster_info.columns:
        assemblies = [acc.strip() for acc in cluster_info['assembly_accessions'].values[0].split(',')]
        logging.info("Cluster spans %d assemblies", len(assemblies))
    else:
        # Fallback: try to find assemblies from genome_reports
        logging.warning("No assembly_accessions in cluster report; loading all genome annotation files")
        genome_annotation_dir = cache_dir / "genome_annotation"
        assemblies = [f.stem for f in genome_annotation_dir.glob("GCF_*.csv")]

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
        matching_proteins = ann_df[
            ann_df['proteins__accessionVersion'].isin(cluster_members)
        ]

        if not matching_proteins.empty:
            all_protein_data.append(matching_proteins)

    if not all_protein_data:
        sys.exit(f"No annotation data found for cluster members in {annotation_dir}")

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

    # Sort at the end?
    # detail_df = detail_df.sort_values(by='protein_accession', ignore_index=True)

    return detail_df, annotation_dfs

def make_genome_context_row(
    anchor_acc: str,
    ann_df: pd.DataFrame,
    context: int,
    # same_strand: bool = True,
) -> pd.DataFrame:
    """
    Add context columns to the detail DataFrame, showing flanking proteins.

    :param anchor_acc: Anchor protein accession
    :param annotation_df: Cached annotation dataframe (keys: anchor accessions)
    :param context: Number of flanking proteins to include
    # :param same_strand: Context genes only on same strand (False not yet implemented)
    :return: Single-row DataFrame with context columns added
    """
    # Define direction by anchor gene's 5'->3' direction
    orientation = ann_df.loc[ann_df['proteins__accessionVersion'] == anchor_acc, 'orientation'].iloc[0]
    # if same_strand:
    #     ann_df = ann_df[ann_df['orientation'] == orientation].copy()
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
    # catch-all for anything unexpected
    extras = [c for c in df.columns if c not in cols]
    return cols + extras

def write_cluster_details(
    accession: str,
    cache_dir: Path,
    context: int | None = None,
    # same_strand: bool = False,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Write cluster details to CSV file.
    
    :param accession: Protein accession to look up
    :param cache_dir: Cache directory path
    :param context: Number of flanking proteins to include
    :param output_path: Where to write the CSV (defaults to cache_dir/<accession>_details.csv)
    :return: Number of rows written
    """
    if output_path is None:
        output_path = cache_dir / f"{accession}_details.csv"
    # Build around anchor proteins first, keeping annotations for context
    detail_df, annotation_dfs = build_cluster_details(accession, cache_dir)
    if not context:
        detail_df.to_csv(output_path, index=False)
        return detail_df  # No context requested
    # Collect context rows for each protein in the cluster
    context_rows = []
    for anchor_acc, assembly_acc in detail_df[['protein_accession', 'assembly_accession']].itertuples(index=False):
        assembly_acc = assembly_acc.split('.')[0]  # Ignore version suffix to match keys
        ann_df = annotation_dfs.get(assembly_acc)
        if ann_df is None:  # This should never happen
            logging.warning("No annotation data for assembly %s; skipping...", assembly_acc)
            continue
        context_rows.append(make_genome_context_row(
            anchor_acc,
            ann_df,
            context=context,
            # same_strand=same_strand,
        ))
    # Assemble all context rows and merge to cluster details
    context_df = pd.concat(context_rows, ignore_index=True)
    detail_df = detail_df.merge(context_df, left_on='protein_accession', right_on='anchor', how='left')
    detail_df = detail_df.drop(columns=['anchor'])
    # Reorder columns
    ordered_cols = ordered_columns(detail_df, context)
    detail_df = detail_df[ordered_cols]
    # Write to CSV
    detail_df.to_csv(output_path, index=False)
    logging.info("Cluster details written to %s (%s proteins)", output_path, len(detail_df))

    return detail_df
