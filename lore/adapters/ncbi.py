"""
Adapter definition for NCBI data.
"""
from typing import Any

from lore.core.adapters import BaseAdapter, adapter_registry

class NcbiGenomeReportsAdapter(BaseAdapter):
    """
    Adapter for NCBI Genome Reports JSON data. Transforms the nested, complex JSON
    into a flat list of dictionaries with consistent keys for easier use in LoRe.
    """
    accepted_formats = {"json"}
    accepted_types = {"ncbi_genome_reports"}

    @property
    def schema(self):
        return {
            "genome_accession": "accession",
            "paired_accession": "paired_accession",
            "organism_taxid": "organism.tax_id",
            "organism_name": "organism.organism_name",
            "organism_strain": "organism.infraspecific_names.strain",
            # B. Assembly and bioproject info
            "bioproject_accession": "assembly_info.bioproject_accession",
            "bioproject_title": "assembly_info.bioproject_lineage[0].bioprojects[0].title",
            "release_date": ("assembly_info.release_date", self.safe_date),
            "sequencing_tech": "assembly_info.sequencing_tech",
            "assembly_method": "assembly_info.assembly_method",
            "contigs": ("assembly_stats.number_of_contigs", self.safe_int),
            "length_bp": ("assembly_stats.total_sequence_length", self.safe_int),
            # C. Biosample info
            "biosample_accession": "assembly_info.biosample.accession",
            "isolation_source": "assembly_info.biosample.isolation_source",
            "collection_year": (
                lambda x: x.get("assembly_info", {}).get("biosample", {}).get("collection_date", "")[:4]
                if x.get("assembly_info", {}).get("biosample", {}).get("collection_date") else None,
                self.safe_int,
            ),
            "collection_date": ("assembly_info.biosample.collection_date", self.safe_date),
            "collection_country":
                lambda x: x.get("assembly_info", {}).get("biosample", {}).get("geo_loc_name", "").split(':')[0]
                if x.get("assembly_info", {}).get("biosample", {}).get("geo_loc_name") else None,
            "collection_region":
                lambda x: x.get("assembly_info", {}).get("biosample", {}).get("geo_loc_name", "").split(':')[1]
                if ':' in x.get("assembly_info", {}).get("biosample", {}).get("geo_loc_name", "") else None,
            # D. Annotation info
            "annotation_pipeline": "annotation_info.pipeline",
            "genes_total": ("annotation_info.stats.gene_counts.total", self.safe_int),
            "genes_protein": ("annotation_info.stats.gene_counts.protein_coding", self.safe_int),
            "genes_nc": ("annotation_info.stats.gene_counts.non_coding", self.safe_int),
            "genes_pseudo": ("annotation_info.stats.gene_counts.pseudogene", self.safe_int),
            # Authors
            "submitter": "assembly_info.submitter",
            # "owner": "biosample.owner",
            # "owner_comments": "assembly_info.comments",
            # Quality metrics
            "assembly_level": "assembly_info.assembly_level",
            "n50": ("assembly_stats.contig_n50", self.safe_int),
            "completeness": ("checkm_info.completeness", self.safe_float),
            "contamination": ("checkm_info.contamination", self.safe_float),
            "best_ani_match": "average_nucleotide_identity.best_ani_match.organism_name",
            "best_ani_value": ("average_nucleotide_identity.best_ani_match.ani", self.safe_float),
            # "ani_comment": "average_nucleotide_identity.comment",
            "genome_notes": "assembly_info.genome_notes",
        }

    def parse(self, raw_data: Any) -> list[dict]:
        records = super().parse(raw_data)
        # Unwrap NCBI payload
        unwrapped = []
        for rec in records:
            if "reports" in rec and isinstance(rec["reports"], list):
                unwrapped.extend(rec["reports"])
            else:
                unwrapped.append(rec)
        return unwrapped


adapter_registry.register(NcbiGenomeReportsAdapter())


class NcbiGenomeAnnotationsAdapter(BaseAdapter):
    """
    Adapter for NCBI Genome Reports JSON data. Transforms the nested, complex JSON
    into a flat list of dictionaries with consistent keys for easier use in LoRe.
    """
    accepted_formats = {"jsonl"}
    accepted_types = {"ncbi_annotation_packages"}

    @property
    def schema(self):
        return {
            "genome_accession": "annotations[0].assemblyAccession",
            "protein_accession": "proteins[0].accessionVersion",
            "symbol": "symbol",
            "name": "name",
            "protein_length": "proteins[0].length",
            "chromosome": "chromosomes[0]",
            "begin": "genomicRegions[0].geneRange.range[0].begin",
            "end": "genomicRegions[0].geneRange.range[0].end",
            "orientation": "genomicRegions[0].geneRange.range[0].orientation",
            "gene_type": "geneType",
            "locus_tag": "locusTag",
        }


adapter_registry.register(NcbiGenomeAnnotationsAdapter())
