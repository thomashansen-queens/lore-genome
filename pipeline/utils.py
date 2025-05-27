"""Various helper functions for the pipeline."""
import logging
from pathlib import Path
import re
import subprocess
import pandas as pd

def trim_fasta(fasta_lines: list, keep_residues: int) -> str:
    """
    Filters and trims sequences based on the cluster_tail parameter.

    :param fasta_lines (list): List FASTA lines, with alternating headers and sequences
    :param cluster_tail (int): Positive to keep first N residues, negative to keep last N residues.
    :return str: Processed FASTA lines.
    """
    processed_fasta = []
    limit = abs(keep_residues)
    if keep_residues > 0:
        logging.info("Trimming FASTA to first %s residues.", keep_residues)
    else:
        logging.info("Trimming FASTA to final %s residues.", abs(keep_residues))
    for i in range(0, len(fasta_lines), 2):
        header = fasta_lines[i].strip()
        sequence = fasta_lines[i + 1].strip()

        # Skip sequences that are too short
        if len(sequence) < limit:
            continue

        # Trim sequence based on which terminus you want to keep
        if keep_residues > 0:
            trimmed_seq = sequence[:keep_residues]  # Keep first n residues
        else:
            trimmed_seq = sequence[keep_residues:]  # Keep last n residues
        processed_fasta.append(header)
        processed_fasta.append(trimmed_seq)
    logging.info("%s sequences survived trimming.", len(processed_fasta) // 2)
    return '\n'.join(processed_fasta)


def sci_namer(full_name: str, style='snake') -> str:
    """
    Changes species name from e.g. "Escherichia coli BL21" to "e_coli".
    If style == 'scientific', will instead return E. coli.

    :param full_name (str): Full species name.
    :param style (str): 'snake' or 'scientific'.
    :return str: Formatted species name
    """
    split = re.split(r'\W+', full_name.strip())
    if len(split) >= 2:  # Genus species
        genus, species = split[0], split[1]
        if style == 'snake':
            return genus.lower()[0] + '_' + species.lower()
        elif style == 'scientific':
            return genus[0].upper() + '. ' + species.lower()
        else:
            raise ValueError("Invalid style. Choose 'snake' or 'scientific'.")
    elif len(split) == 1:  # Genus only
        if style == 'snake':
            return split[0].lower()
        elif style == 'scientific':
            return split[0]
    return ''


def unwrap_single_value(value):
    """Converts single-value lists to a single value."""
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def cluster_mmseqs(
        mmseqs: str,
        fasta_path: Path,
        cluster_prefix: Path,
        min_seq_id: float = 0.9,
) -> pd.DataFrame:
    """
    Clusters given proteins sequences using mmseqs2.

    :param mmseqs (str): Path to mmseqs2 executable.
    :param fasta_path (Path): Path to input FASTA file.
    :param cache_path (Path): Path to output TSV file.
    :return: DataFrame
    """
    command = [
            mmseqs,
            'easy-linclust',
            str(fasta_path),  # input FASTA
            str(cluster_prefix.with_suffix('')),  # output prefix
            str(cluster_prefix.parent / 'tmp'),  # temporary directory used by mmseqs2
            '--min-seq-id',
            str(min_seq_id),
            '-v',
            '2',  # verbosity 2: only show warnings and errors
        ]
    logging.info("Clustering DB using MMseqs2 easy-cluster...")
    logging.info("MMseqs2 command: %s", ' '.join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None  # silences typechecker... won't be None
    for line in process.stdout:
        print("MMseqs Output:", line.strip())
    process.wait()

    clust_df = pd.read_table(
        str(cluster_prefix) + '_cluster.tsv',
        header=None,
        names=['cluster', 'member'],
    )
    logging.info("Proteins clustered: %s", format(clust_df['member'].nunique(),','))
    logging.info("Clusters identified: %s", format(clust_df['cluster'].nunique(),','))
    return clust_df


def parse_fasta(fasta_str: str) -> dict:
    """
    Parse a FASTA formatted string into a dictionary mapping accession -> sequence.
    Assumes that each FASTA header starts with '>' and the accession is the first token.

    :param fasta_str (str): A string containing FASTA-formatted sequences.
    :returns dict: accession: sequence
    """
    fasta_dict = {}
    current_acc = None
    current_seq = ''
    for line in fasta_str.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            # New header
            if current_acc:
                fasta_dict[current_acc] = current_seq
            # Parse the header to get the accession (first token after '>')
            header = line[1:].strip()
            current_acc = header.split()[0]
            current_seq = ''
        else:
            current_seq = current_seq + line.strip()
    # Add the last sequence if any
    if current_acc:
        fasta_dict[current_acc] = current_seq
    return fasta_dict

def sample_genome_reports(
    reports_df: pd.DataFrame,
    genome_limit: int = 50,
    sampling_strategy: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample a specified number of genome reports from the DataFrame. Tries to select
    a diverse group of genomes based on their biosample location and collection date.

    :param reports_df (DataFrame): DataFrame containing genome reports.
    :param genome_limit (int): Maximum number of genomes to sample.
    :param samping_strategy (list): List of columns to use for sampling diversity.
    :param random_state (int): Random seed for reproducibility.
    :return list: List of sampled genome reports.
    """
    if genome_limit > len(reports_df):
        logging.warning("Genome limit exceeds available genome reports. Using all.")
        return reports_df
    if sampling_strategy is None or not sampling_strategy:
        logging.info("Ignoring diversity when sampling assemblies; using random sampling...")
        return reports_df.sample(n=genome_limit, random_state=random_state)
    elif sampling_strategy == ['default']:
        group_cols = [
            'assembly_info__biosample__geo_loc_name',  # geo_loc is inconsistent, but could be useful
            'assembly_info__biosample__collection_date',
        ]
    else:
        group_cols = sampling_strategy
    df = reports_df.copy()  # avoid modifying the original DataFrame
    # split geo_loc_name by : to isolate country/state if present
    if 'assembly_info__biosample__geo_loc_name' in df.columns:
        df['assembly_info__biosample__geo_loc_name'] = df['assembly_info__biosample__geo_loc_name'].apply(
            lambda x: x.split(':')[0] if isinstance(x, str) else x
        )
    # if present, include reference genome(s) then remove from pool for sampling
    reference_df = pd.DataFrame([])
    if 'assembly_info__refseq_category' in df:
        ref_mask = df['assembly_info__refseq_category']=='reference genome'
        reference_df = df[ref_mask].copy()
        logging.info("Auto-included %s reference genomes: %s",
                     len(reference_df), reference_df['accession'].to_list())
        df = df[~ref_mask]  # pop reference genomes from the pool
        genome_limit -= len(reference_df)
    # beyond the reference, use stratified sampling, prioritizing diverse biosamples
    df = df.sort_values(by=group_cols, ascending=True)  # sort for reproducibility
    groups = df.groupby(group_cols)
    group_reps = []
    for _, group_df in groups:
        # Pick one representative from each group.
        rep = group_df.sample(n=1, random_state=random_state)
        group_reps.append(rep)
    reps_df = pd.concat(group_reps)
    if len(reps_df) >= genome_limit:
        final_sample = reps_df.sample(n=genome_limit, random_state=random_state)
    else:
        remaining_needed = genome_limit - len(reps_df)
        df = df.drop(reps_df.index)
        additional_sample = df.sample(n=remaining_needed, random_state=random_state)
        final_sample = pd.concat([reps_df, additional_sample])
    final_sample = pd.concat([reference_df, final_sample])
    logging.info("Sampling %s genomes...", len(final_sample))
    return final_sample
