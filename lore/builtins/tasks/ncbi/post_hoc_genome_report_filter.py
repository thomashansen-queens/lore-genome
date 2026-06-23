"""
Applies post-hoc filtering to the results of NCBI's Fetch Genome Reports.
These filters and options are available in the Task configuration, but by applying
them after the fact, a user can iteratively test different filters without needing
to re-query NCBI's API. Furthermore, it provides knowledge of how much data is
being filtered out at each step.
"""
from typing import Any
import pandas as pd

import lore

from .fetch_genome_reports import NcbiFilterOptions

class NcbiPostHocInputs(NcbiFilterOptions):
    """
    Input model for post-hoc filtering of NCBI genome reports inherited from the
    fetch task.
    """
    source = lore.ArtifactInput(
        label="NCBI Genome Reports",
        accepted_data="ncbi_genome_reports",
        select=lore.SINGLE,
        load_as=lore.RAW,
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


# @lore.task(
#     "ncbi.filter_post_hoc",
#     inputs=NcbiPostHocInputs,
#     outputs=NcbiPostHocOutputs,
#     name="Post-hoc filter NCBI Genome Reports",
#     category="NCBI",
#     icon="🗐>",
#     preview_mode="live",
# )
def ncbi_post_hoc_handler(
    ctx: lore.ExecutionContext,
    source: Any,
    **kwargs,
):
    """
    Locally filters an existing set of NCBI genome reports without re-queryging the API.
    """
    adapter = ctx.get_input_adapter("source")
    if adapter is None:
        raise ValueError("No adapter found for the input Artifact(s).")

    source_artifacts = ctx.input_artifacts.get("source", [])
    ext = source_artifacts[0].extension if source_artifacts else "json"

    # 1. Parse into DataFrame for easier filtering
    parsed_records = adapter.parse(source)
    if not parsed_records:
        raise ValueError("Source artifact is empty.")

    df_parsed = pd.DataFrame(parsed_records)
    df_adapted = pd.DataFrame(adapter.adapt(parsed_records))

    # 2. Apply filters
    if kwargs.get("filters_reference_only"):
        # accession starts with GCF

    if kwargs.get("filters_exclude_paired_reports"):
        # for each GCF accession, get its paired_accession
        # exclude all records with an accession in the paired_accessions list

    if kwargs.get("filter_has_annotation"):
        # annotation_info is present and non-empty
        if "annotation_info" in df_parsed.columns:
            df = 

    if kwargs.get("filters_exclude_atypical"):
        # assembly_info.atypical is present

    if kwargs.get("filters_is_type_material"):
        # not yet sure how to identify this one post-hoc
        # will do a query for this = True on E. coli to figure it out

    if kwargs.get("filters_is_ictv_exemplar"):
        # not yet sure how to identify this one post-hoc
        # will do a query for this = True on some virus to figure it out

    if kwargs.get("filters_exclude_multi_isolate"):
        # assembly_info.genome_notes.contains the substring "multi-isolate"

    if kwargs.get("tax_exact_match"):
        # taxon_id matches exactly
        # Tricky to do since organism.tax_id is an arbitrary string-number
        # Could instead use organism.organism_name is an exact match

    if kwargs.get("filters_assembly_level"):
        # assembly_info.assembly_level is in the list of assembly levels provided

    if kwargs.get("filters_assembly_source"):
        # This is another way of saying "refseq only" but allows GCA only
        # Should just delete refseq only and use this
        # if AssemblySource.REFSEQ: GCF only
        # if AssemblySource.GENBANK: GCA only

    if kwargs.get("filters_assembly_version"):
        # if AssemblyLevel.CURRENT:
        # assembly_info.assembly_status != suppressed

    if kwargs.get("filters_is_metagenome_derived"):
        # not sure how to find this

    if kwargs.get("filters_type_material_category"):
        class TypeMaterial(str, Enum):
            """Physical specimens used to describe a taxon."""
            NONE = 'NONE'
            TYPE_MATERIAL = 'TYPE_MATERIAL'
            TYPE_MATERIAL_CLADE = 'TYPE_MATERIAL_CLADE'
            TYPE_MATERIAL_NEOTYPE = 'TYPE_MATERIAL_NEOTYPE'
            TYPE_MATERIAL_REFTYPE = 'TYPE_MATERIAL_REFTYPE'
            PATHOVAR_TYPE = 'PATHOVAR_TYPE'
            TYPE_MATERIAL_SYN = 'TYPE_MATERIAL_SYN'

    if kwargs.get("filters_first_release_date"):
        # If an assembly gets updated, I'm unsure if it a "first release" is tracked
        # Could just check:
        # assembly_info.release_date is after the provided date

    if kwargs.get("filters_last_release_date"):
        # assembly_info.release_date is before the provided date

    # 3. Map back to lossless parsed format for output
    surviving_indices = df_parsed.index.to_list()
    final_records = [parsed_records[i] for i in surviving_indices]

    ctx.logger.info("Pos-hoc filtering complete. %d of %d records remain after filtering.",
                    len(final_records), len(parsed_records))

    # 4. Serialize output and save as new Artifact
    content = adapter.serialize(final_records, ext=ext)
    ctx.materialize_content(
        output_key="filtered_data",
        content=content,
        extension=ext,
        metadata={
            "original_record_count": len(parsed_records),
            "filtered_record_count": len(final_records),
        }
    )
