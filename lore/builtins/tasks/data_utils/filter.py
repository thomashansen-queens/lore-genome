"""
Task for filtering a list of records based on a pandas query string.
"""
import re

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
    regex = lore.ValueInput(
        bool,
        label="Regex mode",
        description="Treat the query string as a raw Regular Expression across all columns (bypasses Pandas query).",
        default=False,
    )
    query_string = lore.ValueInput(
        str | None,
        label="Query string",
        description="Pandas query string."
            "Copy and paste for NCBI Strict PostHoc Filter: ➜ "
            "~(genome_accession.str.startswith('GCA_', na=False) & paired_accession.str.startswith('GCF_', na=False)) "
            "and assembly_level in ['Complete Genome', 'Chromosome'] "
            "and genome_notes.isnull() "
            "and best_ani_match.str.contains('YOUR ORGANISM HERE', na=False)",
        default=None,
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


@lore.memoize(prefix="filter_query_adapter", ignore=["source", "adapter"])
def _load_dataframe(
    ctx: lore.ExecutionContext,
    source: list[dict],
    adapter: TableAdapter,
    cache_key: str,
) -> pd.DataFrame:
    """Helper function to allow use of memoization for loading."""
    adapted_records = adapter.adapt(source)
    df = pd.DataFrame(adapted_records).reset_index(drop=True)
    return df


def _make_query_pattern(query_string: str) -> str:
    """Convert a simple query string into a case-insensitive substring search pattern."""
    new_query = query_string.strip()
    new_query = new_query.replace('"', '').replace("'", "")

    # Turn list into | separated for regex search
    if "," in new_query:
        parts = [re.escape(p.strip()) for p in new_query.split(",") if p.strip()]
        new_query = "|".join(parts)
    return re.escape(new_query)


@lore.task(
    'filter.query',
    inputs=QueryInputs,
    outputs=QueryOutputs,
    name="Filter by query",
    category="Data processing",
    icon="🔍︎",
    live_preview=True,
)
def filter_query_handler(
    ctx: lore.ExecutionContext,
    source: list[dict],
    regex: bool = False,
    query_string: str | None = None,
):
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
    artifact_ids = "_".join(sorted(a.id for a in source_artifacts))
    cache_key = f"{adapter.name}_{artifact_ids}"

    df = _load_dataframe(ctx, source, adapter, cache_key)

    if df.empty:
        raise ValueError("The adapted DataFrame is empty. Check the input data and adapter schema.")

    # 3. Apply filter via pandas query
    if not query_string or not str(query_string).strip():
        surviving_indices = df.index.tolist()
        query_metadata = "True (no filtering applied)"

    elif regex:
        # A. Explicit Regex mode (skip escaping, skip pandas query)
        try:
            mask = df.astype(str).apply(
                lambda col: col.str.contains(query_string, case=False, na=False, regex=True)
            ).any(axis=1)
            surviving_indices = df[mask].index.tolist()
            query_metadata = f"Regex search: {query_string}"
        except Exception as e:
            raise ValueError(f"Invalid regex pattern '{query_string}': {e}") from e

    else:
        try:
            # B. Pandas Query string
            surviving_indices = df.query(query_string).index.tolist()
            query_metadata = query_string

        except Exception as query_err:
            if any(op in query_string for op in ["==", "!=", ">", "<", "&", "|", "~"]):
                raise ValueError(
                    f"Invalid Pandas Query Syntax: {query_err}\n"
                    f"Query: {query_string}"
                ) from query_err
            else:
                ctx.logger.warning(
                    "Pandas query failed (%s). Falling back to substring search for: %s", 
                    query_err, query_string
                )
            try:
                # C. Fallback to case-insensitive substring search
                search_pattern = _make_query_pattern(query_string)
                mask = df.astype(str).apply(
                    lambda col: col.str.contains(search_pattern, case=False, na=False, regex=True)
                ).any(axis=1)
                surviving_indices = df[mask].index.tolist()
                query_metadata = f"Fallback substring search: {query_string}"
            except Exception as e:
                raise ValueError(f"Invalid query string '{query_string}': {e}") from e

    ctx.logger.info("Query matched %s out of %s records.", len(surviving_indices), len(df))


    # Temporary fix for preview adapter, which expects a list of strings rather than dicts.
    if len(source) > 0 and type(source[0]) is dict:
        # 4. Map back to RAW data for preservation of provenance
        final_records = [source[i] for i in surviving_indices]
        content = adapter.serialize(final_records, extension=ext)
    else:
        content = df.iloc[surviving_indices].to_csv(index=False)
    

    # 5. Materialize
    ctx.materialize_content(
        output_key="filtered_data",
        content=content,
        extension=ext,
        data_type=inherited_type,
        metadata={
            "query": query_string,
            "count": len(surviving_indices),
            "query_metadata": query_metadata,
        },
    )
