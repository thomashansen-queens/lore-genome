"""
This module contains the main pipeline for fetching and processing genome data.
It is run as a Class to allow for instantiation and configuration.

Refer to the README.md for usage instructions.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import pandas as pd

from ncbi.datasets.openapi import ApiClient, GenomeApi
from pipeline.config import PipelineConfig
from pipeline.api_functions import get_genome_reports, get_genome_annotation_package, get_genome_proteins
from pipeline.io_functions import load_or_fetch_df, load_or_fetch_text
from pipeline.utils import sci_namer, filter_genome_reports, filter_gene_annotations, trim_fasta, cluster_mmseqs, sample_genome_reports
from pipeline.summary import make_protein_report

class GenomePipeline:
    """A class to handle the genome pipeline for fetching and processing genome data."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.organism = sci_namer(config.taxons[0])
        self.cache_dir = Path(config.download_dir.expanduser()) / self.organism
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_client = ApiClient(configuration=config.ncbi)
        self.genome_api = GenomeApi(self.api_client)
        self.setup_logging()

    def setup_logging(self):
        """Set up logging for the pipeline instance."""
        logger = logging.getLogger()
        handler = RotatingFileHandler(self.cache_dir / 'pipeline.log', maxBytes=10*1024*1024, backupCount=5)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def fetch_and_filter_genomes(self, **kwargs) -> pd.DataFrame:
        """
        Gets "genome reports" from NCBI Datasets API. Deduplicates RefSeq and INSDC
        genomes, preferentially keeping RefSeq. Also removes atypical genomes.
        """
        cache_path = self.cache_dir / "genome_reports"
        reports_df = load_or_fetch_df(
            api=self.genome_api,
            cache_path=cache_path,
            fetch_func=get_genome_reports,
            config=self.config,
            **kwargs,
        )
        logging.info("Genome reports: %s", format(len(reports_df), ','))
        # Filtering
        cache_path_filtered = self.cache_dir / "genome_reports_filtered"
        filtered_df = load_or_fetch_df(
            cache_path=cache_path_filtered,
            fetch_func=filter_genome_reports,
            df=reports_df,
            config=self.config,
            **kwargs,
        )
        logging.info("Genome reports (filtered): %s", format(len(filtered_df), ','))
        return filtered_df

    def sample_genomes(self, reports_df: pd.DataFrame) -> list:
        """
        Samples genome reports based on the specified sampling strategy.
        """
        if self.config.genome_limit > 0:
            cache_path = self.cache_dir / "genome_reports_sampled"
            sampled_df = load_or_fetch_df(
                cache_path=cache_path,
                fetch_func=sample_genome_reports,
                reports_df=reports_df,
                genome_limit=self.config.genome_limit,
                sampling_strategy=self.config.sampling_strategy,
            )
            logging.info("Sampled genome reports: %s", format(len(sampled_df), ','))
            return sampled_df['accession'].tolist()
        logging.info("No sampling applied. Using all genome reports.")
        return reports_df['accession'].tolist()

    def fetch_gene_annotations(self, genome_accessions: list) -> pd.DataFrame:
        """
        Gets "gene annotation packages" from NCBI Datasets API, which contain
        metadata for the genes in the genome reports.
        """
        annotation_dir = self.cache_dir / "genome_annotation"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        dfs = []
        for assembly in genome_accessions:
            cache_path = annotation_dir / assembly
            df = load_or_fetch_df(
                api=self.genome_api,
                cache_path=cache_path,
                fetch_func=get_genome_annotation_package,
                accession=assembly,
            )
            df = pd.json_normalize(df.to_dict(orient='records'), sep='__')
            dfs.append(df)
        annotations_df = pd.concat(dfs, ignore_index=True)
        logging.info("Total gene annotations: %s from %s genomes",
                     format(len(annotations_df),','), len(dfs))
        # Filtering
        cache_path_filtered = self.cache_dir / "gene_annotations_filtered"
        filtered_df = load_or_fetch_df(
            cache_path=cache_path_filtered,
            fetch_func=filter_gene_annotations,
            df=annotations_df,
        )
        logging.info("Gene annotations (filtered): %s", format(len(filtered_df), ','))
        return filtered_df

    def fetch_protein_fasta(self, genome_accessions: list) -> str:
        """Get all unique protein FASTA sequences for given genomes."""
        proteins_dir = self.cache_dir / "proteins"
        proteins_dir.mkdir(parents=True, exist_ok=True)
        cache_path = proteins_dir / "unique_proteins.faa"
        fasta_str = load_or_fetch_text(
            api=self.genome_api,
            cache_path=cache_path,
            fetch_func=get_genome_proteins,
            accessions=genome_accessions,
        )
        return fasta_str

    def trim_proteins(self, fasta_str: str):
        """
        Trim the protein sequences for downstream clustering. Useful to cluster
        only by the first or last N residues.
        """
        # Assuming a function trim_fasta is defined in utils
        if self.config.cluster_residues != 0:
            fasta_str = fasta_str.split('\n')
            tail = "N" if self.config.cluster_residues > 0 else "C"
            cluster_label = f'proteins_{tail}{abs(self.config.cluster_residues)}'
            cache_path = self.cache_dir / "proteins" / f"{cluster_label}.faa"
            trimmed_fasta = load_or_fetch_text(
                cache_path=cache_path,
                fetch_func=trim_fasta,
                fasta_lines=fasta_str,
                keep_residues=self.config.cluster_residues,
            )
            return trimmed_fasta, cluster_label
        logging.info("No protein FASTA trimming required.")
        return fasta_str, None

    def cluster_proteins(self, cluster_label: str) -> pd.DataFrame:
        """Use mmseqs2 to cluster the protein sequences."""
        cache_path = self.cache_dir / "proteins" / "clustered.tsv"
        cluster_df = load_or_fetch_df(
            mmseqs=self.config.mmseqs,
            cache_path=cache_path,
            cluster_prefix=self.cache_dir / "proteins" / f"{cluster_label}",
            fasta_path=self.cache_dir / "proteins" / f"{cluster_label}.faa",
            fetch_func=cluster_mmseqs,
        )
        return cluster_df

    def make_report(self, annotations_df: pd.DataFrame, fasta_str: str, clusters_df: pd.DataFrame):
        """Generate a human-readable CSV for clustered proteins."""
        # fasta_dict = parse_fasta(fasta_str)
        summary_df = make_protein_report(annotations_df, fasta_str, clusters_df)
        # Save the report to CSV
        summary_df.to_csv(self.cache_dir / "protein_report.csv")
        # summary_df.to_csv(self.cache_dir / "protein_report.csv")
        logging.info("Protein report generated and saved!")
        return summary_df

    def run(self):
        """This function orchestrates the pipeline."""
        # Stage 1: Fetching and filtering genomes
        genomes = self.fetch_and_filter_genomes()
        # Limit the number of genomes to process
        # TO DO: Sample genomes more sensibly (i.e. unique bioprojects)
        genome_accessions = self.sample_genomes(genomes)
        annotations = self.fetch_gene_annotations(genome_accessions)
        protein_fasta = self.fetch_protein_fasta(genome_accessions)
        # Stage 2: Trimming and clustering proteins
        _, cluster_label = self.trim_proteins(protein_fasta)
        clusters = self.cluster_proteins(cluster_label)
        # Stage 3: Summarize
        self.make_report(annotations, protein_fasta, clusters)
        logging.info("Pipeline completed successfully.")
