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
    summed = summary_df.groupby('cluster')[
        ['assemblies', 'proteins__count']
    ].agg('sum')
    clustered_accessions = (
        summary_df
        .groupby('cluster')['proteins__accessionVersion']
        .agg(lambda x: list(set(x)))  # or ','.join(set(x)) if you want strings
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
    # 8. Sort by protein length
    summary_df = summary_df.sort_values(by=['proteins__length'], ascending=False)
    # Rename columns for clarity
    summary_df = summary_df.rename(columns={
        'proteins__accessionVersion': 'top_accession',
        'proteins__length': 'top_length',
    })
    return summary_df
