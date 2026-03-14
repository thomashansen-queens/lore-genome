"""
Task for filtering a list of records based on a pandas query string.
"""
import pandas as pd

from lore.core.adapters import TableAdapter
import lore.dsl as lore


class QueryInputs:
    """Input model for the filter by query task."""
    source = lore.ArtifactInput(
        label="Artifact(s) to filter",
        accepted_data="table",
        select=lore.MULTIPLE,
        load_as=lore.RAW,
    )
    query_string = lore.ValueInput(
        str,
        label="Query string",
        description="Pandas query string."
            "Copy and paste for NCBI Strict PostHoc Filter: ➜\n"
            "~(genome_accession.str.startswith('GCA_', na=False) & paired_accession.str.startswith('GCF_', na=False)) "
            "and assembly_level in ['Complete Genome', 'Chromosome'] "
            "and genome_notes.isnull() "
            "and best_ani_match.str.contains('YOUR ORGANISM HERE', na=False)",
        examples=["assembly_level == 'Complete Genome' and year > 2020"],
    )


class QueryOutputs:
    """Output model for the filter by query task."""
    filtered_data = lore.TaskOutput(
        data_type="table (override in handler)",
        label="Filtered Data",
        description="A new Artifact containing only the records that match the query.",
        is_primary=True,
    )


@lore.task(
    'filter.query',
    inputs=QueryInputs,
    outputs=QueryOutputs,
    name="Filter by query",
    category="Data processing",
    icon="🔍︎",
    live_preview=True,
)
def filter_query_handler(ctx: lore.ExecutionContext, source: list[dict], query_string: str):
    """
    Non-desctructively filter a list of records using a pandas query string on
    the adapted DataFrame.
    """
    # 1. Get adapter and input artifact metadata
    adapter = ctx.get_input_adapter("source")
    if adapter is None:
        raise ValueError("No adapter found for the input Artifact(s).")
    if not isinstance(adapter, TableAdapter):
        raise ValueError(f"The adapter for the input Artifact(s) must be a TableAdapter, but got {type(adapter)}.")

    source_artifacts = ctx.input_artifacts.get("source", [])
    inherited_type = source_artifacts[0].data_type if source_artifacts else "unknown"
    ext = source_artifacts[0].extension if source_artifacts else "json"

    # 2. Adapt to DataFrame
    adapted_records = adapter.adapt(source)
    df = pd.DataFrame(adapted_records)

    if df.empty:
        raise ValueError("The adapted DataFrame is empty. Check the input data and adapter schema.")

    # 3. Apply filter via pandas query
    try:
        surviving_indices = df.query(query_string).index.tolist()
    except Exception as e:
        raise ValueError(f"Invalid query string '{query_string}': {e}") from e

    ctx.logger.info("Query matched %s out of %s records.", len(surviving_indices), len(df))

    # 4. Map back to RAW data for preservation of provenance
    final_records = [source[i] for i in surviving_indices]

    # 5. Materialize
    ctx.materialize_content(
        output_key="filtered_data",
        content=adapter.serialize(final_records, extension=ext),
        extension=ext,
        data_type=inherited_type,
        metadata={"query": query_string, "count": len(final_records)},
    )
