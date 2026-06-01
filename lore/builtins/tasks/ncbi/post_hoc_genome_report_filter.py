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


@lore.task(
    "ncbi.filter_post_hoc",
    inputs=NcbiPostHocInputs,
    outputs=NcbiPostHocOutputs,
    name="Post-hoc filter NCBI Genome Reports",
    category="NCBI",
    icon="🗐>",
    live_preview=True,
)
def ncbi_post_hoc_handler(
    ctx: lore.ExecutionContext,
    source: Any,
    **kwargs,
):
    """
    Locally filters an existing set of NCBI genome reports without re-queryging the API.
    """
    adapter = ctx.get_input_adapter("source")
    source_artifacts = ctx.input_artifacts.get("source", [])
    ext = source_artifacts[0].extension if source_artifacts else "json"

    # 1. Parse into DataFrame for easier filtering
    parsed_records = adapter.parse(source)
    if not parsed_records:
        raise ValueError("Source artifact is empty.")
    