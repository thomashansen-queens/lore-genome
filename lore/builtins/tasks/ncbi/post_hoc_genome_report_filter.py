"""
Applies post-hoc filtering to the results of NCBI's Fetch Genome Reports.
These filters and options are available in the Task configuration, but by applying
them after the fact, a user can iteratively test different filters without needing
to re-query NCBI's API. Furthermore, it provides knowledge of how much data is
being filtered out at each step.
"""
from datetime import date, datetime
from typing import Any

import lore

from .fetch_genome_reports import (
    NcbiFilterOptions,
    AssemblySource,
    AssemblyVersion,
    MetagenomeDerived,
)


def _coerce_date(value: Any) -> date | None:
    """Best-effort parse of an NCBI release_date string (or a datetime) into a date."""
    if isinstance(value, datetime):
        return value.date()
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


class NcbiPostHocInputs(NcbiFilterOptions):
    """
    Input model for post-hoc filtering of NCBI genome reports inherited from the
    fetch task.
    """
    source = lore.ArtifactInput(
        label="NCBI Genome Reports",
        accepted_data="ncbi_genome_reports",
        select="single",
        load_as="raw",
    )


class NcbiPostHocOutputs:
    """
    Output model for post-hoc filtering of NCBI genome reports.
    """
    filtered_data = lore.TaskOutput(
        data_type=lore.Passthrough("source"),
        label="Filtered Genome Reports",
        description="A new Artifact containing only the records that match the query.",
        is_primary=True,
    )


@lore.task(
    "ncbi.filter_post_hoc",
    inputs=NcbiPostHocInputs,
    outputs=NcbiPostHocOutputs,
    name="Post-hoc filter NCBI Genome Reports",
    category="NCBI",
    icon="🗐>",
    preview_mode="live",
)
def ncbi_post_hoc_handler(
    ctx: lore.ExecutionContext,
    source: Any,
    **kwargs,
):
    """
    Locally filters an existing set of NCBI genome reports without re-querying the
    API. Applies the same filters as the fetch task, but client-side, and logs how
    many records each filter removes so a user can see what each one costs.
    """
    adapter = ctx.get_input_adapter("source")
    if adapter is None:
        raise ValueError("No adapter found for the input Artifact(s).")

    source_artifacts = ctx.input_artifacts.get("source", [])
    ext = source_artifacts[0].extension if source_artifacts else "json"

    # 1. Parse into lossless, nested records
    records = adapter.parse(source)
    if not records:
        raise ValueError("Source artifact is empty.")
    original_count = len(records)

    def assembly_info(r: dict) -> dict:
        return r.get("assembly_info") or {}

    def apply_filter(predicate, label: str) -> None:
        """Keep only records satisfying `predicate`, logging how many were removed."""
        nonlocal records
        before = len(records)
        records = [r for r in records if predicate(r)]
        removed = before - len(records)
        if removed:
            ctx.logger.info(
                "Filter '%s' removed %d records (%d remain).",
                label, removed, len(records)
            )

    # 2. Apply filters (mirrors the server-side filters of fetch_genome_reports).

    if kwargs.get("filters_reference_only"):
        # RefSeq assemblies are NCBI's curated "reference" set (GCF_ accessions).
        apply_filter(lambda r: str(r.get("accession", "")).startswith("GCF_"), "reference_only")

    if kwargs.get("filters_has_annotation"):
        apply_filter(lambda r: bool(r.get("annotation_info")), "has_annotation")

    if kwargs.get("filters_exclude_paired_reports"):
        # A primary (GCF) record names its GenBank twin via paired_accession; drop the twin.
        paired = {r.get("paired_accession") for r in records if r.get("paired_accession")}
        apply_filter(lambda r: r.get("accession") not in paired, "exclude_paired_reports")

    if kwargs.get("filters_exclude_atypical"):
        # 'atypical' is only present (a dict of warnings) on atypical assemblies.
        apply_filter(lambda r: not assembly_info(r).get("atypical"), "exclude_atypical")

    if kwargs.get("filters_exclude_multi_isolate"):
        apply_filter(
            lambda r: "multi-isolate" not in str(assembly_info(r).get("genome_notes", "")).lower(),
            "exclude_multi_isolate",
        )

    levels = kwargs.get("filters_assembly_level")
    if levels:
        # Normalize enum / "Complete Genome" -> "complete_genome".
        wanted = {str(getattr(lvl, "value", lvl)).lower() for lvl in levels}
        apply_filter(
            lambda r: str(assembly_info(r).get("assembly_level", "")).lower().replace(" ", "_") in wanted,
            "assembly_level",
        )

    source_db = getattr(kwargs.get("filters_assembly_source"), "value", kwargs.get("filters_assembly_source"))
    if source_db == AssemblySource.REFSEQ.value:
        apply_filter(lambda r: str(r.get("accession", "")).startswith("GCF_"), "assembly_source=refseq")
    elif source_db == AssemblySource.GENBANK.value:
        apply_filter(lambda r: str(r.get("accession", "")).startswith("GCA_"), "assembly_source=genbank")

    version = getattr(kwargs.get("filters_assembly_version"), "value", kwargs.get("filters_assembly_version"))
    if version == AssemblyVersion.CURRENT.value:
        # 'current' excludes suppressed/replaced assemblies.
        apply_filter(
            lambda r: str(assembly_info(r).get("assembly_status", "current")).lower() != "suppressed",
            "assembly_version=current",
        )

    after = _coerce_date(kwargs.get("filters_first_release_date"))
    if after:
        def released_after(r):
            d = _coerce_date(assembly_info(r).get("release_date"))
            return d is not None and d >= after
        apply_filter(released_after, "released_after")

    before = _coerce_date(kwargs.get("filters_last_release_date"))
    if before:
        def released_before(r):
            d = _coerce_date(assembly_info(r).get("release_date"))
            return d is not None and d <= before
        apply_filter(released_before, "released_before")

    # 3. TODO: Filters without a clear post-hoc signal yet: skip them,
    #   but warn user until fixed (rather than silently doing nothing)
    meta_derived = getattr(
        kwargs.get("filters_is_metagenome_derived"), "value", kwargs.get("filters_is_metagenome_derived")
    )
    unsupported = [
        label
        for key, label in (
            ("filters_is_type_material", "Is type material"),
            ("filters_is_ictv_exemplar", "Is ICTV exemplar"),
            ("filters_type_material_category", "Type material category"),
            ("tax_exact_match", "Exact taxon match (needs the original query)"),
        )
        if kwargs.get(key)
    ]
    if meta_derived and meta_derived != MetagenomeDerived.METAGENOME_DERIVED_UNSET.value:
        unsupported.append("Is metagenome derived")
    if unsupported:
        ctx.logger.warning("Filters not yet supported post-hoc were skipped: %s", ", ".join(unsupported))

    # 4. Emit the surviving records as a new Artifact of the same type.
    ctx.logger.info("Post-hoc filtering complete. %d of %d records remain.", len(records), original_count)
    ctx.materialize_content(
        output_key="filtered_data",
        content=adapter.serialize(records, ext=ext),
        extension=ext,
        metadata={
            "original_record_count": original_count,
            "filtered_record_count": len(records),
        },
    )
