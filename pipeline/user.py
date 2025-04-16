"""poo"""
import logging
from pipeline.pipeline import GenomePipeline
import pandas as pd

class DansPipeline(GenomePipeline):
    """poo"""
    def dan_run(self):
        """Custom run settings."""
        # Stage 1: Fetching and filtering genomes
        # genomes = self.fetch_and_filter_genomes()
        genome_list = [
            'GCF_002504185.1',
            'GCF_012274865.1',
            'GCF_012274985.1',
            'GCF_012275005.1',
            'GCF_001244315.1',
            'GCF_030238725.2',
            'GCF_038070605.1',
            'GCF_028228685.1',
            'GCF_001996365.2',
            'GCF_003691525.1',
            'GCF_004006515.1',
            'GCF_046118645.1',
            'GCF_046118655.1',
            'GCF_039658515.1',
            'GCF_045278325.1',
            'GCF_040254425.1',
            'GCF_045287955.1',
            'GCF_001304775.1',
            'GCF_001758605.1',
            'GCF_001887055.1',
            'GCF_026650865.1',
            'GCF_046532685.1',
            'GCF_030994145.1',
            'GCF_030994165.1',
            'GCF_030994185.1',
            'GCF_030994225.1',
            'GCA_030994245.1',
            'GCF_030994405.1',
            'GCF_030994565.1',
            'GCA_030994705.1',
            'GCA_030994885.1',
            'GCA_030995065.1',
            'GCA_030995145.1',
            'GCF_030998145.1',
            'GCF_033441435.1',
            'GCF_033441455.1',
            'GCF_034480095.1',
            'GCF_034480135.1',
            'GCF_034480485.1',
            'GCF_034480505.1',
            'GCF_034480605.1',
            'GCF_036670155.1',
            'GCF_038253855.1',
            'GCF_009649015.1',
            'GCF_006517795.1',
            'GCF_025917705.1',
            'GCF_030064615.2',
            'GCF_030064635.1',
            'GCF_033100075.1',
            'GCF_033573735.1',
            'GCF_018135645.1',
            'GCF_001433415.1',
            'GCF_001636035.1',
            'GCF_001682175.1',
            'GCF_001879585.1',
            'GCF_002209725.2',
        ]
        genomes = pd.DataFrame(genome_list)
        genomes.columns = ['accession']
        # Limit the number of genomes to process
        # TO DO: Sample genomes more sensibly (i.e. unique bioprojects)
        annotations = self.fetch_gene_annotations(genomes.head(self.config.genome_limit))
        fasta = self.fetch_protein_fasta(genomes.head(self.config.genome_limit))
        # Stage 2: Trimming and clustering proteins
        trimmed_fasta, cluster_label = self.trim_proteins(fasta)
        clusters = self.cluster_proteins(cluster_label)
        # Stage 3: Summarize
        self.make_report(annotations, fasta, clusters)
        logging.info("Pipeline completed successfully.")
