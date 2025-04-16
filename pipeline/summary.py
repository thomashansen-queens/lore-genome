"""
A module for number crunching, statistics, and visualization of genome data.
"""
import pandas as pd
from pipeline.utils import parse_fasta

def make_protein_report(
        annotations_df: pd.DataFrame,
        fasta_str: str,
        clusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate a human-readable report for clustered proteins."""
    fasta_dict = parse_fasta(fasta_str)
    # Assign clusters to annotations
    summary_df = annotations_df.merge(
        clusters_df,
        left_on='proteins__accessionVersion',
        right_on='member',
        how='left',
    )
    # Sort to keep most abundant sequence per cluster when drop (tie goes to longest)
    summary_df = summary_df.sort_values(
        by=['cluster', 'proteins__count', 'proteins__length'],
        ascending=False,
    )
    # Sum counts and lists of assemblies for each cluster
    summed = summary_df.groupby('cluster')[
        ['assemblies', 'proteins__count']
    ].agg('sum')
    # Aggregate accessions for each custer
    clustered_accessions = (
        summary_df
        .groupby('cluster')['proteins__accessionVersion']
        .agg(lambda x: list(set(x)))  # or ','.join(set(x)) if you want strings
        .reset_index()
        .rename(columns={'proteins__accessionVersion': 'cluster_accessions'})
    )
    summary_df = summary_df.merge(clustered_accessions, on='cluster', how='left')
    # Aggregate assemblies for each cluster
    summed['assembly_count'] = summed['assemblies'].apply(len)
    summed = summed.rename(columns={
        'proteins__accessionVersion': 'cluster_accessions',
        'assemblies': 'cluster_assemblies',
        'proteins__count': 'count',
    })
    summary_df = summary_df.drop(columns=['member', 'assemblies', 'proteins__count'])
    # Drop duplicates to show only one representative entry per cluster
    summary_df = summary_df.drop_duplicates(
        subset=['cluster'],
        keep='first',
    )
    summary_df = summary_df.merge(summed, on='cluster', how='left')
    # add FASTA sequences
    summary_df['sequence'] = summary_df['proteins__accessionVersion'].map(
        lambda x: fasta_dict.get(x, None)
    )
    # Sort by protein length
    summary_df = summary_df.sort_values(by=['proteins__length'], ascending=False)
    # Rename columns for clarity
    summary_df = summary_df.rename(columns={
        'proteins__accessionVersion': 'accession',
        'proteins__length': 'length',
    })
    return summary_df
