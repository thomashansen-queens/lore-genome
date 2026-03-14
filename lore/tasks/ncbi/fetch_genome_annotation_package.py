"""
Fetch genome annotation packages from NCBI Datasets API for a list of genome 
assembly accessions. Each package contains metadata describing the annotated 
locations of the genes in the assembly. The metadata is returned as a JSON Lines 
file, where each line describes one gene's annotation metadata.

https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/data-packages/gene-package/
"""

import io
import json
from time import sleep
import zipfile
import logging

import lore.dsl as lore
from lore.tasks.ncbi.client import GenomeApi, make_genome_api, retry

from ncbi.datasets.openapi.models.v2_genome_annotation_request_annotation_type import V2GenomeAnnotationRequestAnnotationType
from ncbi.datasets.openapi.models.v2_genome_annotation_request_genome_annotation_table_format import V2GenomeAnnotationRequestGenomeAnnotationTableFormat
from ncbi.datasets.openapi.models.v2_include_tabular_header import V2IncludeTabularHeader


class NcbiAnnotationPackageInputs:
    """
    Inputs for fetching genome annotation packages from NCBI.
    """
    genome_accessions = lore.ArtifactInput(
        description="List of genome assembly accessions (e.g. GCF_000005845.2) to fetch annotation packages for",
        select=lore.MULTIPLE,
        load_as=lore.ADAPTED,
        accepted_data="genome_accession",
        label="Genome accessions",
        examples=["GCF_000005845.2, GCF_000006945.2"],
    )
    annotation_ids = lore.ValueInput(
        list[str] | None,
        default=None,
        description="Limit the reports by internal, unstable annotation ids.",
        examples=["b7a1c8e4-8c9b-4d2a-9f1e-2c3d4e5f6a7b"],
    )
    fetch_limit = lore.ValueInput(
        int | None,
        default=None,
        description="Maximum number of genome reports to fetch. Stop fetching after this number. If None, fetch all available.",
    )
    locations = lore.ValueInput(
        list[str] | None,
        default=None,
        description="Locations with a chromosome or accession and optional start-stop range: chromosome|accession[:start-end]",
        examples=["NC_000913.3:1000-2000", "NC_000913.3"],
    )
    gene_types = lore.ValueInput(
        list[str] | None,
        default=None,
        description="Granular gene types to filter by when available",
        examples=["protein_coding", "tRNA"],
    )
    search_text = lore.ValueInput(
        list[str] | None,
        default=None,
        description="Search text filters (e.g. gene name, product name, locus tag)",
        examples=["DNA polymerase"],
    )
    include_annotation_type = lore.ValueInput(
        V2GenomeAnnotationRequestAnnotationType | None,
        default=None,
        description="Included annotation type to fetch for the assembly package.",
        widget="radio",
    )
    page_size = lore.ValueInput(
        int | None,
        default=20,
        description="The maximum number of features to return. If the number of results exceeds the page size, `page_token` can be used to retrieve the remaining results.",
        max=1000,
        examples=[20],
    )
    table_fields = lore.ValueInput(
        list[str] | None,
        default=None,
        description="Specify which fields to include in the tabular report",
        examples=["gene_symbol", "gene_type", "product_name"],
    )
    table_format = lore.ValueInput(
        V2GenomeAnnotationRequestGenomeAnnotationTableFormat | None,
        default=None,
        description="Optional pre-defined template for processing a tabular data request",
        widget="radio",
    )
    include_tabular_header = lore.ValueInput(
        V2IncludeTabularHeader,
        default=V2IncludeTabularHeader.INCLUDE_TABULAR_HEADER_FIRST_PAGE_ONLY,
        description="Whether this request for tabular data should include the header row",
    )


class NcbiAnnotationPackageOutputs:
    """The concatenated annotation records from the fetched annotation packages"""
    report = lore.TaskOutput(
        data_type="ncbi_annotation_packages",
        label="Annotation packages",
        description="The raw ZIP file content of the fetched annotations.",
        is_primary=True,
    )
    failed_accessions = lore.TaskOutput(
        data_type="genome_accessions",
        label="Failed accessions",
        description="The list of accessions that failed to be fetched.",
    )


@retry(default_logger=logging.getLogger("lore.ncbi"))
def _fetch_genome_annotation_package(api: GenomeApi, accession: str, **kwargs) -> bytes:
    """
    Fetch one genome annotation package from NCBI Datasets API. Result is a bytearray
    of the zip file content.
    """
    result = api.download_genome_annotation_package_without_preload_content(accession=accession, **kwargs)
    return result.read()


@lore.memoize(prefix="ncbi_annotation_package")
def _fetch_single_annotation_package(
    ctx: lore.ExecutionContext,
    accession: str,
    **kwargs,
) -> list[dict]:
    """
    Cachable! Fetches and the annotation package for a single accession.
    Returns a list of dictionaries.
    """
    api_key = ctx.runtime.secrets.ncbi_api_key
    api = make_genome_api(api_key)

    clean_kwargs = {
        k: v for k, v in kwargs.items()
        if v not in (None, "", [], {})
    }

    # 1. Fetch the zip bytes
    records = []
    zip_bytes = _fetch_genome_annotation_package(api, accession, **clean_kwargs)
    if not zip_bytes:
        ctx.logger.warning("No data returned for accession %s", accession)
        return records

    # 2. Extract the JSONL file from the zip
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        targets = [f for f in z.namelist() if f.endswith(".jsonl")]
        if not targets:
            ctx.logger.warning("No JSONL file found in zip for accession %s", accession)
            return records

        with z.open(targets[0]) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    ctx.logger.error("JSON decode error for accession %s: %s", accession, e)
                    continue

    sleep(0.2)
    return records


@lore.task(
    'ncbi.fetch_genome_annotation_package',
    inputs=NcbiAnnotationPackageInputs,
    outputs=NcbiAnnotationPackageOutputs,
    name="Fetch NCBI Genome Annotation Packages",
    category="NCBI",
    icon="📦︎",
)
def fetch_genome_annotation_package_handler(
    ctx: lore.ExecutionContext,
    genome_accessions: list[str],
    fetch_limit: int | None = None,
    **kwargs,
):
    """
    Task handler to fetch annotation packages from NCBI Datasets API for a list
    of accessions.
    """
    if fetch_limit:
        genome_accessions = genome_accessions[:fetch_limit]
    ctx.logger.info("Fetching NCBI genome annotations for first %s accessions", len(genome_accessions))

    if not ctx.runtime.secrets.ncbi_api_key:
        ctx.logger.warning("No NCBI API key found in Secrets! Authentication may be rate-limited.")

    # 2. Execution loop for each accession. API fetches one at a time.
    out_path = ctx.get_temp_path("annotation_packages.jsonl")
    record_count = 0
    failed_accessions = []

    with open(out_path, "w", encoding="utf-8") as tmp_out:
        for i, acc in enumerate(genome_accessions):
            try:
                records = _fetch_single_annotation_package(ctx, acc, **kwargs)

                for r in records:
                    tmp_out.write(json.dumps(r) + "\n")
                    record_count += 1

            except Exception as e:
                ctx.logger.error("Couldn't fetch annotation package for accession %s: %s", acc, e)
                failed_accessions.append(acc)

            if i % 10 == 0 and i > 0:
                ctx.logger.info("Processed %d annotation packages so far...", i)

    # 4. Finalize output
    ctx.logger.info("Fetch complete. Collected %s annotation records.", format(record_count, ","))
    if record_count == 0:
        raise ValueError("No annotation records were fetched. Please check the accessions and try again.")

    # 5. Materialize (Hand off to Session)
    ctx.materialize_file(
        output_key="report",
        source_path=out_path,
        metadata={
            "record_count": record_count,
            "accessions_failed": failed_accessions,
        },
    )

    # 6. Handle the failures as a separate Artifact
    if failed_accessions:
        failed_path = ctx.get_temp_path("failed_accessions.txt")
        with open(failed_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_accessions))

        ctx.materialize_file(
            output_key="failed_accessions",
            source_path=failed_path,
        )
