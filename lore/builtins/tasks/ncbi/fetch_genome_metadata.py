"""
Fetch assembly metadata via the NCBI Datasets API.
"""
import json
import logging
import lore
from time import sleep
from .datasets_client import datasets_client
from .config import retry


class AssemblyMetadataInputs:
    """Inputs for Assembly Metadata Task"""
    accessions = lore.ArtifactInput(
        accepted_data=["assembly", "assembly_accession", "genome_accession"],
        select="multiple",
        load_as="adapted",
        label="Genome accessions",
    )


class AssemblyMetadataOutputs:
    """Outputs for Assembly Metadata Task"""
    metadata = lore.TaskOutput(
        data_type="ncbi_genome_reports",
        label="Assembly metadata",
        is_primary=True,
    )


# TODO: Log pagination and set up page size params
@retry(default_logger=logging.getLogger("lore.ncbi")) # fallback logger for the module
def _fetch_assembly_metadata_chunk(api, chunk: list[str]) -> list[dict]:
    """
    Fetch one chunk of accessions from the NCBI Datasets v2alpha POST endpoint.
    """
    reports = []
    page_token = None

    while True:
        payload = {"accessions": chunk}
        if page_token:
            payload["page_token"] = page_token

        response = api.post(
            "/genome/dataset_report",
            json=payload,
            timeout=60.0,
        )

        data = response.json()

        page_reports = data.get("reports") or data.get("Reports") or []
        reports.extend(page_reports)

        page_token = data.get("next_page_token") or data.get("Next_page_token")
        if not page_token:
            break

        # Snooze for API rate limiting (especially for unauthenticated requests)
        sleep(0.34)

    return reports


@lore.task(
    "ncbi.datasets.assembly_metadata",
    inputs=AssemblyMetadataInputs,
    outputs=AssemblyMetadataOutputs,
    name="NCBI Datasets Assembly Metadata (by accession)",
    category="NCBI Datasets",
    preview_mode="full",
)
def assembly_metadata(
    ctx: lore.ExecutionContext,
    accessions: list[str | dict],
):
    """
    Fetches comprehensive assembly metadata using the NCBI Datasets API.
    """
    ncbi_config = ctx.get_config("ncbi")
    api_key = ncbi_config.api_key if ncbi_config else None

    if not api_key:
        ctx.logger.warning("No NCBI API key set in Settings! Authentication may be rate-limited.")

    # 1. Clean and extract accessions (handling both raw strings and adapted dicts)
    clean_accs = []
    for item in accessions:
        if isinstance(item, dict):
            # If the series adapter yielded a dictionary, grab the first value
            val = str(next(iter(item.values()), "")).strip()
        else:
            val = str(item).strip()

        if val:
            clean_accs.append(val)

    if not clean_accs:
        raise ValueError("No valid accessions provided.")

    # Deduplicate to avoid redundant API weight
    clean_accs = list(set(clean_accs))

    ctx.logger.info(f"Fetching metadata for {len(clean_accs)} unique assemblies...")

    # 2. Chunking logic (Datasets API POST limit is 1000 accessions)
    chunk_size = 1000
    acc_chunks = [clean_accs[i:i + chunk_size] for i in range(0, len(clean_accs), chunk_size)]

    all_reports = []

    with datasets_client(api_key=api_key) as api:
        for i, chunk in enumerate(acc_chunks):
            ctx.logger.info(f"Fetching metadata chunk {i+1}/{len(acc_chunks)} ({len(chunk)} accessions)...")

            chunk_reports = _fetch_assembly_metadata_chunk(api, chunk)
            all_reports.extend(chunk_reports)

            # Rate limiting
            if i < len(acc_chunks) - 1:
                sleep(0.1 if api_key else 0.34)

    if not all_reports:
        raise RuntimeError("NCBI Datasets returned no metadata for the provided accessions.")

    ctx.logger.info(f"Successfully retrieved metadata for {len(all_reports)} assemblies.")

    # 4. Materialize as a single, standard JSON array
    return ctx.materialize_content(
        content=json.dumps(all_reports, indent=2),
        output_key="metadata",
        name="assembly_reports",
        extension="json",
        metadata={
            "assembly_count": len(all_reports),
        },
    )
