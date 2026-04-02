"""
Fetch genome reports from NCBI Datasets API based on taxonomic and search term criteria
"""
from datetime import datetime
from enum import Enum
import json
import logging

import httpx

import lore.dsl as lore
from .client import ncbi_client, retry


# --- Enums for Dropdowns (copied from NCBI Open API) ---

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
    taxons = lore.ValueInput(
        list[str],
        description="NCBI Taxonomy ID or name (common or scientific) at any rank",
        label="Taxa",
        examples=["Vibrio cholerae", "28901"],
    )
    search_terms = lore.ValueInput(
        list[str] | None,
        default=None,
        description="Search term groups. Inner lists are AND, outer list is OR. Looks in all fields.",
        label="Search Terms",
        examples=["Canada AND hot springs", "nanopore"],
    )
    fetch_limit = lore.ValueInput(
        int | None,
        default=None,
        description="Maximum number of genome reports to fetch. Stop fetching after this number. If None, fetch all available.",
        label="Fetch limit",
    )
    # Boolean Toggles (UI: Checkboxes)
    filters_reference_only = lore.ValueInput(bool, label="Reference only", default=False, description="If true, only return reference and representative (GCF_ and GCA_) genome assemblies.")
    filters_has_annotation = lore.ValueInput(bool, label="Has annotation", default=True, description="Return only annotated genome assemblies")
    filters_exclude_paired_reports = lore.ValueInput(bool, label="Exclude paired reports", default=True, description="For paired (GCA/GCF) records, only return the primary record")
    filters_exclude_atypical = lore.ValueInput(bool, label="Exclude atypical", default=False, description="If true, exclude atypical genomes (often have assembly issues)")
    filters_is_type_material = lore.ValueInput(bool, label="Is type material", default=False, description="For any new species, reporting authors deposit 'type material' in a publicly available biorepository")
    filters_is_ictv_exemplar = lore.ValueInput(bool, label="Is ICTV exemplar", default=False, description="If True, include only International Committee on Taxonomy of Viruses (ICTV) Exemplars")
    filters_exclude_multi_isolate = lore.ValueInput(bool, label="Exclude multi-isolate", default=False, description="Exclude multi-isolate projects")
    tax_exact_match = lore.ValueInput(bool, label="Exact taxon match", default=False, description="If True, only return assemblies with the given NCBI Taxonomy ID, or name. Otherwise, assemblies from taxonomy subtree are included, too.")

    # Enums (UI: Checkbox group)
    filters_assembly_level = lore.ValueInput(
        list[AssemblyLevel],
        default=["complete_genome"],
        label="Assembly level",
        description="Only return genomes with a given assembly level (contig = most fragmented)",
    )
    # Enums (UI: Dropdown group)
    filters_assembly_source = lore.ValueInput(
        AssemblySource,
        default=AssemblySource.ALL,
        label="Assembly source",
        description="Which database to query",
        widget="radio",
    )
    filters_assembly_version = lore.ValueInput(
        AssemblyVersion,
        default=AssemblyVersion.CURRENT,
        label="Assembly version",
        description="Genomes can be replaced or suppressed over time. This filter controls whether to include these.",
        widget="radio",
    )
    filters_is_metagenome_derived = lore.ValueInput(
        MetagenomeDerived,
        default=MetagenomeDerived.METAGENOME_DERIVED_UNSET,
        label="Is metagenome derived",
        description="Metagenome derived genomes are assembled from environmental samples"
    )
    filters_type_material_category = lore.ValueInput(
        TypeMaterial,
        default=None,
        label="Type material category",
        description="Type material is a specimen that is used in the definition of a new species or subspecies.",
    )
    # UI: Date picker
    filters_first_release_date = lore.ValueInput(
        datetime | None,
        default=None,
        label="Released after",
        description="Only return assemblies released after this date",
    )
    filters_last_release_date = lore.ValueInput(
        datetime | None,
        default=None,
        label="Released before",
        description="Only return assemblies released before this date",
    )


class NcbiGenomeReportsOutputs:
    """The raw results from NCBI."""
    report = lore.TaskOutput(
        data_type="ncbi_genome_reports",
        label="Genome reports",
        description="A list of genome assembly reports fetched from NCBI Datasets API. Each 'report' is a dict of metadata about the assembly.",
        is_primary=True,
    )

# --- Task Definition ---

@retry(default_logger=logging.getLogger("lore.ncbi"))
def _fetch_genome_reports_page(api: httpx.Client, taxons: list[str], term_set: list[str], page_token: str | None, **kwargs):
    """
    Fetch one page of genome reports from NCBI Datasets API.
    This function exists separate from the iterator so it can be retried via decorator.
    """
    # 1. Basic GET endpoint
    taxon_str = ",".join(taxons)
    url = f"/genome/taxon/{taxon_str}/dataset_report"

    # 2. Build the query parameters dictionary
    params = {}
    if term_set:
        params["filters.search_text"] = term_set
    if page_token:
        params["page_token"] = page_token

    # 3. Translate Python kwargs to NCBI API parameters
    for k, v in kwargs.items():
        if v in (None, "", [], {}):
            continue

        # Handle formatting (Enums and Datetimes)
        if isinstance(v, list):
            val = [item.value if hasattr(item, "value") else str(item) for item in v]
        elif hasattr(v, "value"):
            val = v.value
        elif isinstance(v, datetime):
            # NCBI uses ISO 8601 strings for dates
            val = v.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        else:
            val = v

        # Translate the key (replace the FIRST underscore with a dot if it's a filter)
        # e.g. "filters_assembly_level" -> "filters.assembly_level"
        api_key = k.replace("filters_", "filters.")

        params[api_key] = val

    response = api.get(url, params=params)
    return json.loads(response.read())


def _iter_genome_reports(api: httpx.Client, taxons: list[str], term_set: list[str], fetch_limit: int | None = None, **kwargs):
    """
    Generator for paged genome reports results from NCBI Datasets API search.
    """
    # Cleaning kwargs
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    page_token = None
    total_yielded = 0

    while True:
        result = _fetch_genome_reports_page(api, taxons, term_set, page_token, **clean_kwargs)

        reports = result.get("reports", [])
        if reports:
            yield reports
            total_yielded += len(reports)

        if fetch_limit and total_yielded >= fetch_limit:
            break

        page_token = result.get("next_page_token")
        if not page_token:
            break


@lore.memoize(prefix="ncbi_genome_reports")
def _fetch_all_reports_for_group(
    ctx: lore.ExecutionContext,
    taxons: list[str],
    term_set: list[str],
    fetch_limit: int | None = None,
    **kwargs,
) -> list[dict]:
    """
    Fetch all pages for a specific taxon/term_set combination. This function 
    exists for caching purposes: consumes the page-iterator and returns a list.
    """
    # 1. Instantiate un-cacheable API client
    ncbi_config = ctx.get_config("ncbi")
    api_key = ncbi_config.api_key if ncbi_config else None
    if not api_key:
        ctx.logger.warning("No NCBI API key set in Settings! Authentication may be rate-limited.")

    # 2. Generate with generator
    with ncbi_client(api_key) as api:
        iterator = _iter_genome_reports(
            api=api,
            taxons=taxons,
            term_set=term_set,
            fetch_limit=fetch_limit,
            **kwargs,
        )

        # 3. Consume generator into a list (cached)
        results = []
        for page in iterator:
            results.extend(page)

        return results


@lore.task(
    "ncbi.fetch_genome_reports",
    inputs=NcbiGenomeReportsInputs,
    outputs=NcbiGenomeReportsOutputs,
    name="Fetch NCBI Genome Reports",
    category="NCBI",
    icon="🗐",
)
def fetch_genome_reports_handler(
    ctx: lore.ExecutionContext,
    taxons: list[str],
    search_terms: list[str] | None = None,
    fetch_limit: int | None = None,
    **kwargs,
) -> None:
    """
    Task handler to fetch genome reports from NCBI Datasets API based on the provided inputs.
    Outputs a JSON serializable list of genome report dicts.
    """
    ctx.logger.info("Fetching NCBI genome reports for taxons: %s", taxons)

    ncbi_config = ctx.get_config("ncbi")
    if not ncbi_config or not ncbi_config.api_key:
        ctx.logger.warning("No NCBI API key set in Settings! Authentication may be rate-limited.")

    # 1. Search logic (outer list is OR, inner list is AND)
    search_groups: list[list[str]] = []
    if search_terms:
        for raw_term in search_terms:
            parts = [t.strip() for t in raw_term.lower().split(' and ') if t.strip()]
            if parts:
                search_groups.append(parts)
    else:
        search_groups = [[]]

    # 2. Execution loop
    all_reports = []
    seen_accessions = set()  # Avoid duplicates across search groups
    count = 0

    for term_set in search_groups:
        ctx.logger.info("Fetching genome reports with terms: %s", ' AND '.join(term_set))
        group_reports = _fetch_all_reports_for_group(
            ctx=ctx,
            taxons=taxons,
            term_set=term_set,
            fetch_limit=fetch_limit,
            **kwargs,
        )

        for r in group_reports:
            # d = r.to_dict()
            acc = r.get("accession")

            if acc and acc not in seen_accessions:
                seen_accessions.add(acc)
                all_reports.append(r)
                count += 1

    ctx.logger.info("Completed fetching genome reports. Total unique reports: %d", count)

    if not all_reports:
        ctx.logger.warning("No genome reports found for the given criteria.")
        raise ValueError("No genome reports found for the given criteria.")

    # 3. Materialize Artifact
    name_from_taxon = taxons[0].replace(" ", "_").replace("/", "-")
    if len(taxons) > 1:
        name_from_taxon += f"_and_{len(taxons)-1}_more"

    ctx.materialize_content(
        output_key="report",
        content=json.dumps(all_reports, indent=2),
        name=name_from_taxon,
        extension="json",
        metadata={
            "genome_count": count,
            "taxa": taxons,  # NCBI API uses `taxons`; we will not be subjugated
            "search_terms": search_terms,
        },
    )
