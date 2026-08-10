"""
Grouping utilities for tabular data.
"""
from enum import StrEnum
import pandas as pd
from typing import Any

import lore


class AggregationMethod(StrEnum):
    """Supported Pandas aggregation methods."""
    FIRST = "first"
    LAST = "last"
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    COUNT = "count"


class GroupInputs:
    """Input model for the group by task."""
    source = lore.ArtifactInput(
        label="Artifact(s) to group",
        accepted_data="tabular",  # LoRe 'trait' system will interpret this
        select="multiple",
        load_as="raw",
    )
    group_cols = lore.ValueInput(
        list[str],
        label="Group by columns",
        description="List of column names to group by.",
        default=[],
    )
    agg_method = lore.ValueInput(
        AggregationMethod,
        label="Aggregation functions",
        description="Function to apply to each group.",
        default=AggregationMethod.FIRST,
    )


class GroupOutputs:
    """Output model for the group by task."""
    grouped_data = lore.TaskOutput(
        data_type=lore.Passthrough("source"),
        label="Grouped Data",
        description="A new Artifact containing the grouped data.",
        is_primary=True,
    )


@lore.task(
    "data_utils.group",
    inputs=GroupInputs,
    outputs=GroupOutputs,
    category="Data processing",
    description="Group data by specified columns and apply aggregation functions.",
    icon="⧉",
    preview_mode="live",
)
def group_data(
    ctx: lore.ExecutionContext,
    source: Any,
    group_cols: list[str],
    agg_method: AggregationMethod,
):
    """
    Groups the source data by the specified columns and applies the given aggregation functions.
    Multiple selected source artifacts are combined into a single dataset before grouping.
    """
    if not source:
        return

    # 1. Get the appropriate adapter for the source data
    adapter = ctx.get_input_adapter("source")
    if not adapter:
        raise ValueError("No suitable adapter found for the provided source.")

    # 2. Get the source artifacts
    source_artifacts = ctx.input_artifacts.get("source", [])
    if not source_artifacts:
        raise ValueError("No source Artifacts provided.")

    # Because this task loads as RAW, we manually package config from metadata
    ext = source_artifacts[0].extension
    config = {**(source_artifacts[0].metadata or {}), "ext": ext}

    ctx.logger.info("Starting Group operation on %s artifact(s).", len(source_artifacts))
    ctx.logger.info("Group columns: %s", group_cols)
    ctx.logger.info("Aggregation method: %s", agg_method.value)

    # 3. Parse the whole source into one combined set of records
    parsed = adapter.parse(source, config=config)
    parsed_records = parsed if isinstance(parsed, list) else [parsed]
    if not parsed_records:
        raise ValueError("No records found in the input data.")

    df = pd.DataFrame(parsed_records)

    # 4. Group the data by the specified columns
    if not group_cols:
        ctx.logger.warning("No group columns specified. Returning original data.")
        grouped_df = df
    else:
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Group columns not found in data: {missing}")

        # 5. Group and aggregate
        try:
            grouped_df = (
                df.groupby(group_cols)
                .agg(agg_method.value)
                .reset_index()
            )
        except Exception as e:
            raise ValueError(f"Error during grouping/aggregation: {e}") from e

    # 6. Convert the grouped DataFrame back to records
    ctx.logger.info("Grouped data has %d records.", len(grouped_df))

    content = adapter.serialize(grouped_df.to_dict(orient="records"), config=config)
    ctx.materialize_content(
        output_key="grouped_data",
        content=content,
        extension=ext,
    )
