"""
Given a column name and whitelist/blacklist strings, will filter out entries which do not/do contain the string
"""
from typing import Any
import pandas as pd

import lore
import re

class Inputs:
    """Input model for the filter by query task."""
    source = lore.ArtifactInput(
        label="Artifact(s) to filter",
        accepted_data="tabular",  # LoRe 'trait' system will interpret this
        select="multiple",
        load_as="raw",
    )
    column = lore.ValueInput(
        str,
        label="Column",
        description="The name of the column to filter by",
        default="",
    )
    whitelist = lore.ValueInput(
        str,
        label="Whitelist",
        description="Comma separated list of text to include entries by.",
        default="",
    )
    blacklist = lore.ValueInput(
        str,
        label="Blacklist",
        description="Comma separated list of text to filter out entries by.",
        default="",
    )


class Outputs:
    """Output model for the filter by query task."""
    filtered_data = lore.TaskOutput(
        data_type=lore.Passthrough("source"),
        label="Filtered Data",
        description="A new Artifact containing only the records that match the query.",
        is_primary=True,
    )

@lore.memoize(prefix="filter_query_adapter", ignore=["source", "adapter"])
def _load_dataframe(
    ctx: lore.ExecutionContext,
    parsed_records: list[dict],
    adapter: lore.TabularAdapter,
    cache_key: str,
    config: dict,
) -> pd.DataFrame:
    """Helper function to allow use of memoization for loading."""
    # 1. Load parsed records into a DataFrame
    adapted_records = adapter.adapt(parsed_records, config=config)
    df = pd.DataFrame(adapted_records).reset_index(drop=True)

    # 2. None-ify empty strings
    df = df.replace("", None)

    # 3. Attempt to coerce numeric-only columns to numeric types (skip empty)
    for col in df.columns:
        if df[col].notna().sum() == 0:
            continue

        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() == df[col].notna().sum():
            df[col] = coerced

    df = df.convert_dtypes()

    # 4. Convert column names to underscore format 
    df.columns = [
        re.sub(r'[^a-zA-Z0-9]+', '_', col).strip('_').lower()
        for col in df.columns
    ]

    return df

@lore.task(
    "filter.simple",
    inputs=Inputs,
    outputs=Outputs,
    name="Simple Filter",
    category="Data processing",
    icon="🔍︎",
    preview_mode="live",
)
def simple_filter_handler(
    ctx: lore.ExecutionContext,
    source: Any,
    column: str,
    whitelist: str = "",
    blacklist: str = ""
):
    """
    Filters out entries in a table by a provided column and blacklist/whitelist text query.
    """
    # 1. Get adapter and input artifact metadata
    adapter = ctx.get_input_adapter("source")
    if adapter is None:
        raise ValueError("No adapter found for the input Artifact(s).")
    if not isinstance(adapter, lore.TabularAdapter):
        raise ValueError(f"The adapter for the input Artifact(s) must be a TabularAdapter, but got {type(adapter)}.")

    # Because this task loads as RAW, we manually package config from metadata
    source_artifacts = ctx.input_artifacts.get("source", [])
    ext = source_artifacts[0].extension if source_artifacts else "json"
    config = {**(source_artifacts[0].metadata or {}), "ext": ext}

    # 2. Parsed the raw data into records
    parsed_records = adapter.parse(source, config=config)

    # 3. Adapt to DataFrame
    artifact_ids = "_".join(sorted(a.id for a in source_artifacts))
    cache_key = f"{adapter.name}_{artifact_ids}_{len(parsed_records)}"

    df = _load_dataframe(ctx, parsed_records, adapter, cache_key, config)

    if df.empty:
        raise ValueError("The adapted DataFrame is empty. Check the input data and adapter schema.")

    column = re.sub(r'[^a-zA-Z0-9]+', '_', column).strip('_').lower()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in table columns. Available columns: {", ".join(df.columns)}")

    surviving_indices = df.index
    # Create and apply blacklist filter
    if blacklist:
        blacklist_pattern = "|".join([x.strip() for x in blacklist.split(",")])
        blacklist_indices = df.index[
            df[column].str.contains(
                blacklist_pattern,
                case=False,
                na=False
            )
        ]
        surviving_indices = surviving_indices.difference(blacklist_indices)
    # Create and apply whitelist filter
    if whitelist:
        whitelist_pattern = "|".join([x.strip() for x in whitelist.split(",")])
        whitelist_indices = df.index[
            df[column].str.contains(
                whitelist_pattern,
                case=False,
                na=False
            )
        ]
        surviving_indices = surviving_indices.intersection(whitelist_indices)

    ctx.logger.info("Query matched %s out of %s records.", len(surviving_indices), len(df))

    # 5. Map back to parsed (but unadapted) data for preservation of provenance
    final_records = [parsed_records[i] for i in surviving_indices]
    content = adapter.serialize(final_records, config=config)

    # 6. Materialize
    ctx.materialize_content(
        output_key="filtered_data",
        content=content,
        extension=ext,
        metadata={
            "count": len(surviving_indices),
        },
    )
