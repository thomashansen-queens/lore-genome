"""
NCBI Datasets library
"""
from datetime import datetime
from enum import Enum
import functools
import http.client
import io
import json
import logging
from pathlib import Path
import tempfile
from time import sleep
from typing import List
import zipfile

try:
    from ncbi.datasets.openapi.api_client import ApiClient
    from ncbi.datasets.openapi.api.genome_api import GenomeApi
    from ncbi.datasets.openapi.configuration import Configuration as NcbiConfig
    HAS_SDK = True
except ImportError:
    HAS_SDK = False

from lore.core.tasks import Cardinality, Materialization, TaskOutput, task_registry, ArtifactInput, ValueInput
from lore.core.executor import ExecutionContext

# --- Enums for Dropdowns (taken from NCBI Open API) ---

class AssemblySource(str, Enum):
    """RefSeq: GCF_; GenBank: GCA_"""
    ALL = 'all'
    REFSEQ = 'refseq'
    GENBANK = 'genbank'

class AssemblyLevel(str, Enum):
    """Levels of genome assembly completeness."""
    COMPLETE_GENOME = 'complete_genome'
    CHROMOSOME = 'chromosome'
    SCAFFOLD = 'scaffold'
    CONTIG = 'contig'

class AssemblyVersion(str, Enum):
    """Excludes suppressed/replaced assemblies if CURRENT."""
    CURRENT = 'current'
    ALL_ASSEMBLIES = 'all_assemblies'

class MetagenomeDerived(str, Enum):
    """Filter based on metagenome derived status."""
    METAGENOME_DERIVED_UNSET = 'METAGENOME_DERIVED_UNSET'
    METAGENOME_DERIVED_ONLY = 'metagenome_derived_only'
    METAGENOME_DERIVED_EXCLUDE = 'metagenome_derived_exclude'

class TypeMaterial(str, Enum):
    """Physical specimens used to describe a taxon."""
    NONE = 'NONE'
    TYPE_MATERIAL = 'TYPE_MATERIAL'
    TYPE_MATERIAL_CLADE = 'TYPE_MATERIAL_CLADE'
    TYPE_MATERIAL_NEOTYPE = 'TYPE_MATERIAL_NEOTYPE'
    TYPE_MATERIAL_REFTYPE = 'TYPE_MATERIAL_REFTYPE'
    PATHOVAR_TYPE = 'PATHOVAR_TYPE'
    TYPE_MATERIAL_SYN = 'TYPE_MATERIAL_SYN'

# --- Genome assembly reports ---

class NcbiGenomeReportsInputs:
    """Inputs for fetching genome reports from NCBI."""
    taxons = ValueInput(
        list[str],
        description="NCBI Taxonomy ID or name (common or scientific) at any rank",
        label="Taxa",
        examples=["Vibrio cholerae", "28901"],
    )
    search_terms = ValueInput(
        list[str],
        default=None,
        description="Search term groups. Inner lists are AND, outer list is OR. Looks in all fields.",
        label="Search Terms",
        examples=["Canada AND hot springs", "nanopore"],
    )
    fetch_limit = ValueInput(
        int,
        default=None,
        description="Maximum number of genome reports to fetch. Stop fetching after this number. If None, fetch all available.",
        label="Fetch limit",
    )
    # Boolean Toggles (UI: Checkboxes)
    filters_reference_only = ValueInput(bool, label="Reference only", default=False, description="If true, only return reference and representative (GCF_ and GCA_) genome assemblies.")
    filters_has_annotation = ValueInput(bool, label="Has annotation", default=True, description="Return only annotated genome assemblies")
    filters_exclude_paired_reports = ValueInput(bool, label="Exclude paired reports", default=True, description="For paired (GCA/GCF) records, only return the primary record")
    filters_exclude_atypical = ValueInput(bool, label="Exclude atypical", default=False, description="If true, exclude atypical genomes (often have assembly issues)")
    filters_is_type_material = ValueInput(bool, label="Is type material", default=False, description="For any new species, reporting authors deposit 'type material' in a publicly available biorepository")
    filters_is_ictv_exemplar = ValueInput(bool, label="Is ICTV exemplar", default=False, description="If True, include only International Committee on Taxonomy of Viruses (ICTV) Exemplars")
    filters_exclude_multi_isolate = ValueInput(bool, label="Exclude multi-isolate", default=False, description="Exclude multi-isolate projects")
    tax_exact_match = ValueInput(bool, label="Exact taxon match", default=False, description="If True, only return assemblies with the given NCBI Taxonomy ID, or name. Otherwise, assemblies from taxonomy subtree are included, too.")

    # Enums (UI: Checkbox group)
    filters_assembly_level = ValueInput(
        list[AssemblyLevel],
        default=["complete_genome"],
        label="Assembly level",
        description="Only return genomes with a given assembly level (contig = most fragmented)",
    )
    # Enums (UI: Dropdown group)
    filters_assembly_source = ValueInput(
        AssemblySource,
        default=AssemblySource.ALL,
        label="Assembly source",
        description="Which database to query",
        widget="radio",
    )
    filters_assembly_version = ValueInput(
        AssemblyVersion,
        default=AssemblyVersion.CURRENT,
        label="Assembly version",
        description="Genomes can be replaced or suppressed over time. This filter controls whether to include these.",
        widget="radio",
    )
    filters_is_metagenome_derived = ValueInput(
        MetagenomeDerived,
        default=MetagenomeDerived.METAGENOME_DERIVED_UNSET,
        label="Is metagenome derived",
        description="Metagenome derived genomes are assembled from environmental samples"
    )
    filters_type_material_category = ValueInput(
        TypeMaterial,
        default=None,
        label="Type material category",
        description="Type material is a specimen that is used in the definition of a new species or subspecies.",
    )
    # UI: Date picker
    filters_first_release = ValueInput(
        datetime,
        default=None,
        label="Released after",
        description="Only return assemblies released after this date",
    )
    filters_last_release = ValueInput(
        datetime,
        default=None,
        label="Released before",
        description="Only return assemblies released before this date",
    )


class NcbiGenomeReportsOutputs:
    """The raw results from NCBI."""
    report = TaskOutput(
        data_type="ncbi_genome_reports",
        label="Genome reports",
        description="A list of genome assembly reports fetched from NCBI Datasets API. Each 'report' is a dict of metadata about the assembly.",
        is_primary=True,
    )

# --- Helpers ---

class MissingNcbiDatasetsError(RuntimeError):
    """Raised when NCBI Datasets SDK is not installed."""
    pass

def _ensure_ncbi_sdk():
    """Ensure that the NCBI Datasets SDK is installed."""
    if not HAS_SDK:
        raise MissingNcbiDatasetsError(
            "NCBI Datasets SDK is not installed. Please install the wheel in lore-genome/wheels"
        )

def make_genome_api(api_key: str | None = None) -> 'GenomeApi':
    """Create an NCBI Datasets GenomeApi client."""
    _ensure_ncbi_sdk()
    config = NcbiConfig()
    if api_key:
        config.api_key = {'cookieAuth': api_key}

    api_client = ApiClient(configuration=config)
    return GenomeApi(api_client=api_client)

def retry(exceptions, tries=4, delay=2, logger=None):
    """
    A decorator that allows API calls to retry a set number of times before failing.

    :param exceptions: The exception(s) to catch and retry on.
    :param tries: The number of times to retry the function.
    :param delay: The delay between retries (exponentially increasing).
    :param logger: The logger to use for messages.
    :return: The result of the function call.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exec = None
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exec = e
                    sleeptime = delay ** attempt
                    msg = f'{e}, Retrying in {sleeptime} seconds...'
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    sleep(sleeptime)
            if logger:
                logger.error("Failed to execute %s after %s attempts.", func.__name__, tries)
            raise last_exec if last_exec else Exception("API error. Also, bug in retry decorator.")
        return wrapper
    return decorator

# --- Fetch genome reports by taxon ---

@retry((http.client.IncompleteRead, TimeoutError), logger=logging.getLogger("lore.ncbi"))
def _fetch_genome_reports_page(api: GenomeApi, taxons: list[str], term_set: list[str], page_token: str | None, **kwargs):
    """
    Fetch one page of genome reports from NCBI Datasets API.
    This function exists separate from the iterator so it can be retried via decorator.
    """
    response = api.genome_dataset_reports_by_taxon_without_preload_content(
        taxons=taxons,
        filters_search_text=term_set,
        page_token=page_token,
        **kwargs,
    )
    raw_json = response.data.decode("utf-8")
    data = json.loads(raw_json)
    return data

def _iter_genome_reports(api: GenomeApi, taxons: list[str], term_set: list[str], fetch_limit: int | None = None, **kwargs):
    """
    Generator for paged genome reports results from NCBI Datasets API search.
    """
    # Cleaning: make sure LoRe kwargs don't get passed to the API call. These are mostly related to search term formatting which is handled separately.
    for k in ("search_terms", "filters_search_text"):
        kwargs.pop(k, None)

    page_token = None
    total_yielded = 0

    while True:
        result = _fetch_genome_reports_page(api, taxons, term_set, page_token, **kwargs)

        reports = result.get("reports", [])
        if reports:
            yield reports
            total_yielded += len(reports)

        if fetch_limit and total_yielded >= fetch_limit:
            break

        page_token = result.get("next_page_token")
        if not page_token:
            break


@task_registry.register(
    "ncbi.fetch_genome_reports",
    inputs=NcbiGenomeReportsInputs,
    outputs=NcbiGenomeReportsOutputs,
    name="Fetch NCBI Genome Reports",
    category="NCBI",
    icon="🗐",
)
def fetch_genome_reports_handler(
    ctx: ExecutionContext,
    taxons: list[str],
    search_terms: list[str] | None = None,
    fetch_limit: int | None = None,
    **kwargs,
) -> None:
    """
    Task handler to fetch genome reports from NCBI Datasets API based on the provided inputs.
    Outputs a JSON serializable list of genome report dicts.
    """
    # 1. Setup and configure
    ctx.logger.info("Fetching NCBI genome reports for taxons: %s", taxons)

    # Access secrets via Runtime
    api_key = ctx.runtime.secrets.ncbi_api_key
    if not api_key:
        ctx.logger.warning("No NCBI API key found in Secrets! Authentication may be rate-limited.")
    api = make_genome_api(api_key)

    # 2. Search logic (outer list is OR, inner list is AND)
    search_groups: List[List[str]] = []
    if search_terms:
        for raw_term in search_terms:
            parts = [t.strip() for t in raw_term.lower().split(' and ') if t.strip()]
            if parts:
                search_groups.append(parts)
    else:
        search_groups = [[]]

    # 3. Execution loop
    all_reports = []
    seen_accessions = set()
    count = 0

    for term_set in search_groups:
        ctx.logger.info("Fetching genome reports with terms: %s", ' AND '.join(term_set))
        iterator = _iter_genome_reports(
            api=api,
            taxons=taxons,
            term_set=term_set,
            fetch_limit=fetch_limit,
            **kwargs,
        )

        for page in iterator:
            for r in page:
                # d = r.to_dict()
                acc = r.get("accession")

                if acc and acc not in seen_accessions:
                    seen_accessions.add(acc)
                    all_reports.append(r)
                    count += 1

    ctx.logger.info("Completed fetching genome reports. Total unique reports: %d", count)

    if not all_reports:
        ctx.logger.warning("No genome reports found for the given criteria.")
        ctx.results.report = None
        return

    # 4. Materialize Artifact
    safe_taxon = taxons[0].replace(" ", "_").replace("/", "-")
    if len(taxons) > 1:
        safe_taxon += f"_and_{len(taxons)-1}_more"

    artifact = ctx.materialize_content(
        content=json.dumps(all_reports, indent=2),
        name=safe_taxon,
        extension="json",
        output_key="report",
        metadata={
            "genome_count": count,
            "taxons": taxons,
            "search_terms": search_terms,
        },
    )

# --- Fetch genome annotation package ---

class NcbiAnnotationPackageInputs:
    """
    Inputs for fetching genome annotation packages from NCBI.
    https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/data-packages/gene-package/
    The gene annotation report contains metadata describing the annotated locations
    of the genes in the data package and is only provided for WP_ accessions. The 
    file is in JSON Lines format, where each line is the metadata for one gene. Use 
    the dataformat tool for easy conversion to a tabular format of selected fields.
    """
    accessions = ArtifactInput(
        description="List of genome assembly accessions (e.g. GCF_000005845.2) to fetch annotation packages for",
        cardinality=Cardinality.ANY,
        load_as=Materialization.CONTENT,
        accepted_data=["genome_accessions"],
        label="Genome accessions",
        examples=["GCF_000005845.2, GCF_000006945.2"],
    )
    # Filters
    fetch_limit = ValueInput(
        int,
        default=None,
        description="Maximum number of genome reports to fetch. Stop fetching after this number. If None, fetch all available.",
    )
    symbols = ValueInput(
        list[str],
        default=None,
        description="Filter by gene symbols when available",
        examples=["rpoB", "16S"],
    )
    gene_types = ValueInput(
        list[str],
        default=None,
        description="Granular gene types to filter by when available",
        examples=["protein_coding", "tRNA"],
    )
    search_text = ValueInput(
        list[str],
        default=None,
        description="Search text filters (e.g. gene name, product name, locus tag)",
        examples=["DNA polymerase"],
    )


class NcbiAnnotationPackageOutputs:
    """The concatenated annotation records from the fetched annotation packages"""
    report = TaskOutput(
        data_type="ncbi_annotation_packages",
        label="Annotation packages",
        description="The combined gene annotation records from the fetched NCBI genome annotation packages. Each line is a JSON record describing one gene's annotation metadata.",
        is_primary=True,
    )
    failed_accessions = TaskOutput(
        data_type="genome_accessions",
        label="Failed accessions",
        description="The list of accessions that failed to be fetched.",
    )


@retry((http.client.IncompleteRead, TimeoutError), logger=logging.getLogger("lore.ncbi"))
def _fetch_genome_annotation_package(api: GenomeApi, accession: str, **kwargs) -> bytearray:
    """
    Fetch one genome annotation package from NCBI Datasets API. Result is a bytearray
    of the zip file content.
    """
    result = api.download_genome_annotation_package(accession=accession, **kwargs)
    return result


@task_registry.register(
    'ncbi.fetch_genome_annotation_package',
    inputs=NcbiAnnotationPackageInputs,
    outputs=NcbiAnnotationPackageOutputs,
    name="Fetch NCBI Genome Annotation Packages",
    category="NCBI",
    icon="📦︎",
)
def fetch_genome_annotation_handler(
    ctx: ExecutionContext,
    accessions: list[str],
    fetch_limit: int | None = None,
    filename: str | None = None,
    **kwargs,
):
    """
    Task handler to fetch annotation packages from NCBI Datasets API for a list
    of accessions.
    """
    # 1. Setup and configure
    ctx.logger.info("Fetching NCBI genome annotations for %s accessions", len(accessions))

    # Access secrets via Runtime
    api_key = getattr(ctx.runtime.secrets, "ncbi_api_key", None)
    if not api_key:
        ctx.logger.warning("No NCBI API key found in Secrets! Authentication may be rate-limited.")
    api = make_genome_api(api_key)

    # 2. Execution loop for each accession. API fetches one at a time.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_out:
        out_path = Path(tmp_out.name)
        record_count = 0
        failed_accessions = []

        for i, accession in enumerate(accessions):
            if fetch_limit and i >= fetch_limit:
                break
            try:
                ctx.logger.info("Fetching genome annotation package for accession: %s", accession)
                zip_bytes = _fetch_genome_annotation_package(api, accession, **kwargs)
                if not zip_bytes:
                    ctx.logger.warning(f"Empty API response for {accession}")
                    failed_accessions.append(accession)
                    continue

                # 3. Extract in memory
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                    targets = [f for f in z.namelist() if f.endswith(".jsonl")]
                    if targets:
                        with z.open(targets[0]) as f:
                            for line in f:
                                try:
                                    record = json.loads(line.decode("utf-8"))
                                    tmp_out.write(json.dumps(record) + "\n")
                                    record_count += 1
                                except json.JSONDecodeError as e:
                                    ctx.logger.error("Error decoding JSON line for %s: %s", accession, e)
                    else:
                        ctx.logger.warning(f"Could not find any '.jsonl' files in package for {accession}.")
                        failed_accessions.append(accession)

                # Be nice to the API (should I?)
                sleep(0.2)

            except Exception as e:
                ctx.logger.error("Couldn't fetch annotation package for accession %s: %s", accession, e)
                failed_accessions.append(accession)
                continue

            # Progress log
            if i % 20 == 0 and i > 0:
                ctx.logger.info("Fetched %d annotation packages so far...", i+1)

    # 4. Finalize output
    ctx.logger.info("Fetch complete. Collected %s annotation records.", format(record_count, ","))
    if failed_accessions:
        ctx.logger.warning(f"Failed to fetch {len(failed_accessions)} accessions: {failed_accessions}")
    if record_count == 0:
        ctx.logger.warning("No annotation records were successfully fetched. Task will return empty.")

    # 5. Materialize (Hand off to Session)
    safe_name = accessions[0].replace(" ", "_").replace("/", "-")
    if len(accessions) > 1:
        safe_name += f"_and_{len(accessions)-1}_more"
    filename = filename or f"annotations_{safe_name}.json"

    # Handle the failures as a separate Artifact
    if failed_accessions:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(failed_accessions))
            failed_path = Path(f.name)

        failed_artifact = ctx.materialize_file(
            output_key="failed_accessions",
            source_path=failed_path,
        )

    artifact = ctx.materialize_file(
        output_key="report",
        source_path=out_path,
        metadata={
            "record_count": record_count,
            "accessions_failed": failed_accessions,
        },
    )

# --- Fetch protein FASTA ---

class NcbiProteinFastaInputs:
    """Inputs for fetching protein FASTA files from NCBI"""
    accessions = ArtifactInput(
        description="List of genome accessions to fetch protein FASTA for",
        cardinality=Cardinality.ANY,
        load_as=Materialization.CONTENT,
        accepted_data=["genome_accession"],
        examples=["GCF_000005845.2, GCF_000006945.2"],
    )
    chunk_size = ValueInput(
        int,
        default=20,
        description="How many proteins to fetch in one API call. " \
        "Fetching in batches is more efficient and polite to the API than fetching one at a time, but setting this too high may cause timeouts or URL length issues. " \
        "In the future, I will figure out the ideal number here and hide this from the user.",
        label="Chunk size",
    )
    fetch_limit = ValueInput(
        int,
        default=None,
        description="Maximum number of assembly packages to fetch. Stop fetching after this number. If None, fetch all available.",
        label="Fetch limit",
    )
    save_map = ValueInput(
        bool,
        default=False,
        description="Whether to save a mapping of genome accessions to the protein " \
        "accessions that were fetched for them.This is useful for downstream analysis " \
        "to know which proteins came from which genomes. Not needed if you also have " \
        "the annotation packages, which contain the same information (and more).",
        label="Save genome-to-protein map",
    )

class NcbiProteinFastaOutputs:
    """Outputs from fetching protein FASTA files from NCBI"""
    fasta = TaskOutput(
        data_type="protein_fasta",
        label="Protein FASTA",
        description="The combined FASTA records for the fetched protein accessions.",
        is_primary=True,
    )
    failed_accessions = TaskOutput(
        data_type="genome_accessions",
        label="Failed accessions",
        description="The list of protein accessions that failed to be fetched.",
    )
    genome_to_protein_map = TaskOutput(
        data_type="genome_protein_map",
        label="Genome to protein map",
        description="A mapping of genome accessions to the protein accessions that were fetched for them.",
    )

from ncbi.datasets.openapi.models.v2_assembly_dataset_request_resolution import V2AssemblyDatasetRequestResolution
from ncbi.datasets.openapi.models.v2_annotation_for_assembly_type import V2AnnotationForAssemblyType

@retry((http.client.IncompleteRead, TimeoutError), logger=logging.getLogger("lore.ncbi"))
def _fetch_protein_fasta(api: GenomeApi, accessions: list[str], **kwargs) -> bytearray:
    """
    Fetch protein FASTA files from NCBI Datasets API for a list of genome accessions
    """
    result = api.download_assembly_package(
        accessions=accessions,
        include_annotation_type=[V2AnnotationForAssemblyType.PROT_FASTA],
        hydrated=V2AssemblyDatasetRequestResolution.FULLY_HYDRATED,
        **kwargs,
    )
    return result


@task_registry.register(
    "ncbi.fetch_protein_fasta",
    inputs=NcbiProteinFastaInputs,
    outputs=NcbiProteinFastaOutputs,
    name="Fetch NCBI Protein FASTA",
    category="NCBI",
    icon="＞",
)
def fetch_protein_fasta_handler(
    ctx: ExecutionContext,
    accessions: list[str],
    fetch_limit: int | None = None,
    chunk_size: int = 20,
    save_map: bool = False,
    **kwargs,
):
    """
    Task handler to fetch protein FASTA files from NCBI Datasets API for a list
    of genome accessions.
    """
    # 1. Setup and configure
    ctx.logger.info("Fetching NCBI protein FASTA for %s accessions", len(accessions))

    # Access secrets via Runtime
    api_key = getattr(ctx.runtime.secrets, 'ncbi_api_key', None)
    if not api_key:
        ctx.logger.warning("No NCBI API key found in Secrets! Authentication may be rate-limited.")
    api = make_genome_api(api_key)

    # Fetch in batches to be nice to the API and avoid URL length limits
    accession_chunks = [
        accessions[i:i+chunk_size] for i in range(0, len(accessions), chunk_size)
    ]
    missing_accessions = []
    seen_proteins = set()
    genome_to_protein_map = {}

    # Write to tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp_out:
        tmp_path = Path(tmp_out.name)

        for i, chunk in enumerate(accession_chunks):
            # apply fetch limit
            if fetch_limit and (i + 1) * chunk_size >= fetch_limit:
                chunk = chunk[:fetch_limit - i * chunk_size]
                if not chunk:
                    break

            response = _fetch_protein_fasta(api, chunk, **kwargs)

            with zipfile.ZipFile(io.BytesIO(response)) as z:
                fasta_files = [f for f in z.namelist() if f.endswith(".faa")]
                if not fasta_files:
                    ctx.logger.warning("No FASTA files found in package for chunk %s", i)

                chunk_missing = set(chunk)  # checked off as they're parsed

                for fasta_file in fasta_files:
                    # match nested file to an accession
                    matched_acc = next((acc for acc in chunk if acc in fasta_file), None)

                    if matched_acc and save_map and matched_acc not in genome_to_protein_map:
                        genome_to_protein_map[matched_acc] = []

                    try:
                        with z.open(fasta_file) as f:
                            # stream for line-by-line reading
                            text_stream = io.TextIOWrapper(f, encoding="utf-8")

                            write_current_seq = False
                            for line in text_stream:
                                if line.startswith(">"):
                                    # Extract protein accession ">WP_000123.1 protein name" -> "WP_000123.1"
                                    prot_acc = line.split(maxsplit=1)[0][1:]

                                    if save_map and matched_acc:
                                        genome_to_protein_map[matched_acc].append(prot_acc)

                                    # Deduplication logic
                                    if prot_acc not in seen_proteins:
                                        seen_proteins.add(prot_acc)
                                        write_current_seq = True
                                        tmp_out.write(line)
                                    else:
                                        write_current_seq = False

                                elif write_current_seq:
                                    # This is sequence data for a protein we decided to keep
                                    tmp_out.write(line)
                                    if not line.endswith("\n"):
                                        tmp_out.write("\n")

                        if matched_acc:
                            chunk_missing.discard(matched_acc)

                    except zipfile.BadZipFile:
                        ctx.logger.error("Error reading ZIP file for %s. Skipping.", fasta_file)

            missing_accessions.extend(chunk_missing)

    ctx.materialize_file(
        source_path=tmp_path,
        output_key="fasta",
        data_type="protein_fasta",
        metadata={
            "accessions_fetched": len(accessions) - len(missing_accessions),
        },
    )

    if missing_accessions:
        ctx.materialize_content(
            content="\n".join(missing_accessions),
            name="missing_accessions",
            extension="txt",
            output_key="failed_accessions",
            data_type="genome_accessions",
        )

    if save_map:
        ctx.materialize_content(
            content=json.dumps(genome_to_protein_map, indent=2),
            output_key="genome_to_protein_map",
            extension="json",
            data_type="genome_protein_map"
        )
