"""
Fetch genome annotation packages from NCBI Datasets API for a list of genome 
assembly accessions. Each package contains metadata describing the annotated 
locations of the genes in the assembly. The metadata is returned as a JSON Lines 
file, where each line describes one gene's annotation metadata.

https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/data-packages/gene-package/
"""

from enum import Enum
import io
import json
from time import sleep
import zipfile
import logging

import httpx

import lore.dsl as lore
from lore.builtins.tasks.ncbi.client import ncbi_client, retry

# --- NCBI Enums ---
class V2GenomeAnnotationRequestAnnotationType(str, Enum):
    GENOME_FASTA = 'GENOME_FASTA'
    RNA_FASTA = 'RNA_FASTA'
    PROT_FASTA = 'PROT_FASTA'

# --- In/out models ---

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
    table_fields = lore.ValueInput(
        list[str] | None,
        default=None,
        description="Specify which fields to include in the tabular report",
        examples=["gene_symbol", "gene_type", "product_name"],
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
def _fetch_genome_annotation_package(api: httpx.Client, accession: str, **kwargs) -> bytes:
    """
    Fetch one genome annotation package from NCBI Datasets API. Result is a bytearray
    of the zip file content.
    """
    headers = {"Accept": "application/zip"}
    result = api.get(
        f"/genome/accession/{accession}/annotation_report/download",
        params=kwargs,
        headers=headers,
    )
    return result.read()


@lore.memoize(prefix="ncbi_annotation_package", ignore="api")
def _fetch_single_annotation_package(
    ctx: lore.ExecutionContext,
    api: httpx.Client,
    accession: str,
    **kwargs,
) -> list[dict]:
    """
    Cachable! Fetches and the annotation package for a single accession.
    Returns a list of dictionaries.
    """
    # 1. Stringify Enums and clean kwargs for API call
    clean_kwargs = {}
    for k, v in kwargs.items():
        if v in (None, "", [], {}):
            continue
        if isinstance(v, list):
            clean_kwargs[k] = [item.value if hasattr(item, "value") else str(item) for item in v]
        elif hasattr(v, "value"):
            clean_kwargs[k] = v.value
        else:
            clean_kwargs[k] = v

    # 2. Fetch the zip bytes
    records = []

    zip_bytes = _fetch_genome_annotation_package(api, accession, **clean_kwargs)
    if not zip_bytes:
        ctx.logger.warning("No data returned for accession %s", accession)
        return records

    # 3. Extract the JSONL file from the zip
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
    "ncbi.fetch_genome_annotation_package",
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

    ncbi_config = ctx.get_config("ncbi")
    api_key = ncbi_config.api_key if ncbi_config else None
    if not api_key:
        ctx.logger.warning("No NCBI API key set in Settings! Authentication may be rate-limited.")

    # 1. Execution loop for each accession. API fetches one at a time.
    out_path = ctx.get_temp_path("annotation_packages.jsonl")
    record_count = 0
    failed_accessions = []

    with ncbi_client(api_key) as api:
        with open(out_path, "w", encoding="utf-8") as tmp_out:
            for i, acc in enumerate(genome_accessions):
                try:
                    records = _fetch_single_annotation_package(ctx, api, acc, **kwargs)

                    for r in records:
                        tmp_out.write(json.dumps(r) + "\n")
                        record_count += 1

                except Exception as e:
                    ctx.logger.error("Couldn't fetch annotation package for accession %s: %s", acc, e)
                    failed_accessions.append(acc)

                if i % 10 == 0 and i > 0:
                    ctx.logger.info("Processed %d annotation packages so far...", i)

    # 2. Finalize output
    ctx.logger.info("Fetch complete. Collected %s annotation records.", format(record_count, ","))
    if record_count == 0:
        raise ValueError("No annotation records were fetched. Please check the accessions and try again.")

    # 3. Materialize (Hand off to Session)
    ctx.materialize_file(
        output_key="report",
        source_path=out_path,
        metadata={
            "record_count": record_count,
            "accessions_failed": failed_accessions,
        },
    )

    # 4. Handle the failures as a separate Artifact
    if failed_accessions:
        failed_path = ctx.get_temp_path("failed_accessions.txt")
        with open(failed_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_accessions))

        ctx.materialize_file(
            output_key="failed_accessions",
            source_path=failed_path,
        )
