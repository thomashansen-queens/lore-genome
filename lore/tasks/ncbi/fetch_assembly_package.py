"""
Task to fetch genome 'assembly packages' from NCBI Datasets. Currently the main 
use is to fetch the protein FASTA files for a list of genome accessions.
"""

import io
import json
import logging
from pathlib import Path
import zipfile

import lore.dsl as lore
from lore.tasks.ncbi.client import GenomeApi, make_genome_api, retry
from ncbi.datasets.openapi.models.v2_annotation_for_assembly_type import V2AnnotationForAssemblyType
from ncbi.datasets.openapi.models.v2_assembly_dataset_request_resolution import V2AssemblyDatasetRequestResolution


# Enum value: (glob path in zip, output_key in outputs)
NCBI_TYPE_MAP = {
    "PROT_FASTA": ("protein.faa", "protein_fastas"),
    "GENOME_FASTA": ("genomic.fna", "genome_fastas"),
    "GENOME_GFF": ("genomic.gff", "gff_annotations"),
    "GENOME_GBFF": ("genomic.gbff", "gff_annotations"),
    "GENOME_GTF": ("genomic.gtf", "gff_annotations"),
    "CDS_FASTA": ("cds_from_genomic.fna", "genome_fastas"),
    "SEQUENCE_REPORT": ("sequence_report.jsonl", "sequence_reports"),
    "CATALOG": ("dataset_catalog.json", "dataset_catalogs"),
    "ASSEMBLY_REPORT": ("assembly_data_report.jsonl", "assembly_reports"),
}


class NcbiAssemblyPackageInputs:
    """Inputs for fetching assembly packages from NCBI"""
    genome_accessions = lore.ArtifactInput(
        description="List of genome accessions to fetch assembly packages for",
        select=lore.OPTIONAL_MULTIPLE,
        load_as=lore.ADAPTED,
        accepted_data="genome_accession",
        examples=["GCF_000005845.2, GCF_000006945.2"],
    )
    fetch_limit = lore.ValueInput(
        int | None,
        default=None,
        description="Maximum number of assembly packages to fetch. Stop fetching after this number. If None, fetch all available.",
        label="Fetch limit",
    )
    save_map = lore.ValueInput(
        bool,
        default=False,
        description="Whether to save a mapping of genome accessions to the protein "
        "accessions that were fetched for them. This is useful for downstream analysis "
        "to know which proteins came from which genomes. Not needed if you also have "
        "the annotation packages, which contain the same information (and more).",
        label="Save genome-to-protein map",
    )
    chromosomes = lore.ValueInput(
        list[str] | None,
        default=None,
        description="The default setting is all chromosome. Specify individual chromosome by string (1,2,MT or chr1,chr2.chrMT). Unplaced sequences are treated like their own chromosome ('Un'). The filter only applies to fasta sequence.",
        label="Chromosomes to include",
    )
    include_annotation_type = lore.ValueInput(
        list[V2AnnotationForAssemblyType] | None,
        default=None,
        description="The type of annotation to include in the assembly package.",
        label="Annotation type",
    )
    hydrated = lore.ValueInput(
        V2AssemblyDatasetRequestResolution,
        default=V2AssemblyDatasetRequestResolution.DATA_REPORT_ONLY,
        description="Whether to include hydrated annotation data in the assembly package.",
        label="Hydrated",
    )
    extract_catalog = lore.ValueInput(
        bool,
        default=False,
        description="(Automatically on when fetching 'data report only') Keep the dataset_catalog.json file to inspect what files are included by NCBI.",
        label="Extract dataset catalog",
    )


class NcbiAssemblyPackageOutputs:
    """Outputs from fetching assembly packages from NCBI"""
    protein_fastas = lore.TaskOutput(
        data_type="protein_fasta",
        label="Protein FASTAs",
        description="Protein FASTA files fetched from NCBI Datasets",
        yields=lore.OPTIONAL_MULTIPLE,
    )
    genome_fastas = lore.TaskOutput(
        data_type="genome_fasta",
        label="Genome FASTAs",
        description="Genome FASTA files fetched from NCBI Datasets",
        yields=lore.OPTIONAL_MULTIPLE,
    )
    gff_annotations = lore.TaskOutput(
        data_type="gff_annotation",
        label="GFF Annotations",
        description="GFF annotation files fetched from NCBI Datasets",
        yields=lore.OPTIONAL_MULTIPLE,
    )
    sequence_reports = lore.TaskOutput(
        data_type="ncbi_sequence_report",
        label="Sequence Reports",
        description="Sequence reports fetched from NCBI Datasets, containing metadata about the assembly.",
        yields=lore.OPTIONAL_MULTIPLE,
    )
    assembly_reports = lore.TaskOutput(
        data_type="ncbi_genome_reports",
        label="Assembly Reports",
        description="Sequencing and assembly metadata for each genome fetched.",
        yields=lore.OPTIONAL_MULTIPLE,
    )
    dataset_catalogs = lore.TaskOutput(
        data_type="ncbi_dataset_catalog",
        label="Dataset Catalogs",
        description="A catalogue describing what data was fetched (or would be fetched if fully hydrated) for each genome accession.",
        yields=lore.OPTIONAL_MULTIPLE,
    )
    failed_accessions = lore.TaskOutput(
        data_type="genome_accessions",
        label="Failed accessions",
        description="The list of genome accessions that failed to be fetched.",
        yields=lore.OPTIONAL,
    )


@retry(default_logger=logging.getLogger("lore.ncbi"))
def _fetch_assembly_package(api: GenomeApi, accessions: list[str], **kwargs) -> bytes:
    """
    Fetch assembly package from NCBI Datasets API for a list of genome accessions
    """
    from ncbi.datasets.openapi.models.v2_assembly_dataset_request_resolution import V2AssemblyDatasetRequestResolution
    from ncbi.datasets.openapi.models.v2_annotation_for_assembly_type import V2AnnotationForAssemblyType

    response = api.download_assembly_package_without_preload_content(accessions, **kwargs)
    return response.read()


@lore.memoize(prefix="ncbi_assembly_package")
def _fetch_single_assembly_package(
    ctx: lore.ExecutionContext,
    genome_accession: str,
    extension: str = "faa",
    **kwargs,
) -> bytes | None:
    """
    Fetch one assembly package from NCBI Datasets API and extract its contents
    """
    # 1. Instantiate uncacheable API client
    api_key = ctx.runtime.secrets.ncbi_api_key
    api = make_genome_api(api_key)

    clean_kwargs = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}

    # 2. Fetch the assembly package
    response = _fetch_assembly_package(api, [genome_accession], **clean_kwargs)
    if not response:
        return None
    return response


@lore.task(
    "ncbi.fetch_assembly_package",
    inputs=NcbiAssemblyPackageInputs,
    outputs=NcbiAssemblyPackageOutputs,
    name="Fetch NCBI Assembly Package",
    category="NCBI",
    icon="＞",
)
def fetch_assembly_package(
    ctx: lore.ExecutionContext,
    genome_accessions: list[str],
    fetch_limit: int | None = None,
    extract_catalog: bool = False,
    save_map: bool = False,  # TODO: Maybe implement this if we want to keep it
    **kwargs,
):
    """
    Fetch assembly packages from NCBI Datasets API.
    """
    # 1. Setup and configure
    if fetch_limit:
        genome_accessions = genome_accessions[:fetch_limit]
    ctx.logger.info("Fetching NCBI assembly packages for first %s accessions", len(genome_accessions))

    if not ctx.runtime.secrets.ncbi_api_key:
        ctx.logger.warning("No NCBI API key found in Secrets! Authentication may be rate-limited.")

    # 2. Determine which data is incoming
    requested_enums = kwargs.get("include_annotation_type") or []
    if not requested_enums:
        ctx.logger.warning("No annotation types selected! Task will only download metadata.")

    if kwargs.get("hydrated") == V2AssemblyDatasetRequestResolution.DATA_REPORT_ONLY:
        extract_catalog = True
        ctx.logger.info("Hydration level is set to 'data report only', so annotation files will not be included even if selected. To include annotation files, change hydration level to 'fully hydrated' or higher.")

    requested_types = [e.value if hasattr(e, "value") else str(e) for e in requested_enums]
    requested_types += ["CATALOG"] if extract_catalog else []

    # 3. Fetch and unzip data
    failed_accessions = []

    for genome_acc in genome_accessions:
        try:
            zip_bytes = _fetch_single_assembly_package(ctx, genome_acc, **kwargs)
            if not zip_bytes:
                failed_accessions.append(genome_acc)
                continue

            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                namelist = z.namelist()

                for req_type in requested_types:
                    if req_type not in NCBI_TYPE_MAP:
                        ctx.logger.warning("Requested annotation type %s is not yet implemented.", req_type)
                        continue

                    filename, output_key = NCBI_TYPE_MAP[req_type]
                    target_files = [f for f in namelist if filename in f]

                    for target_file in target_files:

                        with z.open(target_file) as f:
                            content = f.read()
                            safe_name = f"{genome_acc}_{filename}"

                            ctx.materialize_content(
                                content=content,
                                output_key=output_key,
                                name=safe_name,
                                extension=filename.split(".")[-1],
                            )

        except Exception as e:
            ctx.logger.error("Failed to process %s: %s", genome_acc, e, exc_info=True)
            failed_accessions.append(genome_acc)
            continue

    # 4. Optionally save the failed accessions for later re-fetch
    if failed_accessions:
        ctx.logger.warning("Failed accessions: %s", ", ".join(failed_accessions))

        ctx.materialize_content(
            content="\n".join(failed_accessions),
            output_key="failed_accessions",
            name="failed_accessions",
            extension="txt",
            data_type="genome_accession",
        )
