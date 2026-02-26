"""
Task for filtering a list of records based on a pandas query string.
"""
import pandas as pd

from lore.core.tasks import task_registry, Cardinality, Materialization, ArtifactInput, ValueInput, TaskOutput
from lore.core.executor import ExecutionContext


class QueryInputs:
    """Input model for the filter by query task."""
    source = ArtifactInput(
        label="Artifact(s) to filter",
        accepted_data=["*"],
        cardinality=Cardinality.ONE_OR_MORE,
        load_as=Materialization.ID,
    )
    query_string = ValueInput(
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
    artifact = TaskOutput(
        data_type="*",
        label="Filtered Data",
        description="A new Artifact containing only the records that match the query.",
        is_primary=True,
    )


@task_registry.register(
    'filter.query',
    inputs=QueryInputs,
    outputs=QueryOutputs,
    name="Filter by query",
    category="Data processing",
    icon="🔍︎",
)
def filter_query_handler(ctx: ExecutionContext, source: list[str], query_string: str):
    """
    Filter a list of records using a pandas query string.
    Returns a new Artifact with the filtered results.
    """
    # 1. Validation
    source_artifacts = ctx.input_artifacts.get("source", [])
    if not source_artifacts:
        raise ValueError("At least one source Artifact must be provided.")
    if len(set(art.data_type for art in source_artifacts)) > 1:
        raise ValueError("Multiple source Artifacts with different data types found. Dynamic merging not supported.")
    inherited_type = source_artifacts[0].data_type
    extension = source_artifacts[0].extension or "json"

    # 2. Set up Adapter
    from lore.core.adapters import adapter_registry, BaseAdapter
    matches = adapter_registry.get_adapters(source_artifacts[0])
    adapter = matches[0] if matches else BaseAdapter()
    ctx.logger.info("Using Adapter: %s for parsing", adapter.name)

    # 3. Memory-safe dynamic merging
    raw_records = []
    with ctx.runtime.get_session(ctx.session_id) as s:
        for art in source_artifacts:
            raw_text = s.load_artifact_data(art.id)
            records = adapter.parse(raw_text)
            raw_records.extend(records)

    ctx.logger.info("Merged %s records from %s source Artifacts", len(raw_records), len(source_artifacts))
    sampling_pool = adapter.adapt(raw_records)

    # 4. Apply the query
    df = pd.DataFrame(sampling_pool)
    if df.empty:
        raise ValueError("The source Artifact(s) contain no parseable data.")

    try:
        filtered_df = df.query(query_string)
    except Exception as e:
        raise ValueError(f"Invalid query string: {query_string}\n{e}") from e

    # 5. Map back to RAW
    indices = filtered_df.index.tolist()
    final_records = [raw_records[i] for i in indices]

    # 6. Materialize
    artifact = ctx.materialize_content(
        output_key="artifact",
        content=adapter.serialize(final_records, extension=extension),
        extension=extension,
        data_type=inherited_type,
        metadata={"query": query_string, "count": len(final_records)},
    )
