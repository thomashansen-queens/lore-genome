"""Filter tabular records by comparing a column with a numeric value."""
import re
from typing import Any

import pandas as pd

import lore


_NUMERIC_EXPRESSION = re.compile(
    r"^\s*(==|!=|>=|<=|>|<)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)
class NumericFilterInputs:
    """Input model for the numeric comparison filter."""

    source = lore.ArtifactInput(
        label="Artifact(s) to filter",
        accepted_data="tabular",
        select="multiple",
        load_as="raw",
    )
    column = lore.ValueInput(
        str,
        label="Column",
        description="The name of the numeric column to filter by.",
        default="",
    )
    expression = lore.ValueInput(
        str,
        label="Comparison",
        description='A numeric comparison, such as ">= 3" or "< 10.5".',
        default="",
        examples=[">= 3"],
    )


class NumericFilterOutputs:
    """Output model for the numeric comparison filter."""

    filtered_data = lore.TaskOutput(
        data_type=lore.Passthrough("source"),
        label="Filtered Data",
        description="A new Artifact containing only the records that match the comparison.",
        is_primary=True,
    )


def _numeric_filter_mask(values: pd.Series, expression: str) -> pd.Series:
    """Return a boolean mask for a supported numeric comparison expression."""
    match = _NUMERIC_EXPRESSION.fullmatch(expression)
    if match is None:
        raise ValueError(
            f"Invalid numeric comparison '{expression}'. "
            "Use one of ==, !=, >, >=, <, <= followed by a number."
        )

    comparison = match.group(1)
    threshold = float(match.group(2))
    numeric_values = pd.to_numeric(values, errors="coerce")
    if comparison == "==":
        mask = numeric_values == threshold
    elif comparison == "!=":
        mask = numeric_values != threshold
    elif comparison == ">":
        mask = numeric_values > threshold
    elif comparison == ">=":
        mask = numeric_values >= threshold
    elif comparison == "<":
        mask = numeric_values < threshold
    else:
        mask = numeric_values <= threshold
    return mask.fillna(False)


@lore.task(
    "filter.numeric",
    inputs=NumericFilterInputs,
    outputs=NumericFilterOutputs,
    name="Filter by numeric comparison",
    category="Data processing",
    icon="🔍︎",
    preview_mode="live",
)
def numeric_filter_handler(
    ctx: lore.ExecutionContext,
    source: Any,
    column: str,
    expression: str = "",
):
    """Filter tabular records by a numeric comparison on one column."""
    adapter = ctx.get_input_adapter("source")
    if adapter is None:
        raise ValueError("No adapter found for the input Artifact(s).")
    if not isinstance(adapter, lore.TabularAdapter):
        raise ValueError(
            "The adapter for the input Artifact(s) must be a TabularAdapter, "
            f"but got {type(adapter)}."
        )

    source_artifacts = ctx.input_artifacts.get("source", [])
    ext = source_artifacts[0].extension if source_artifacts else "json"
    config = {**(source_artifacts[0].metadata or {}), "ext": ext}
    parsed_records = adapter.parse(source, config=config)

    adapted_records = adapter.adapt(parsed_records, config=config)
    df = pd.DataFrame(adapted_records).reset_index(drop=True)
    df = df.replace("", None).convert_dtypes()
    df.columns = [
        re.sub(r"[^a-zA-Z0-9]+", "_", col).strip("_").lower()
        for col in df.columns
    ]

    if df.empty:
        raise ValueError("The adapted DataFrame is empty. Check the input data and adapter schema.")

    normalized_column = re.sub(r"[^a-zA-Z0-9]+", "_", column).strip("_").lower()
    if normalized_column not in df.columns:
        raise ValueError(
            f"Column '{normalized_column}' not found in table columns. "
            f"Available columns: {', '.join(df.columns)}"
        )

    mask = _numeric_filter_mask(df[normalized_column], expression)
    surviving_indices = df.index[mask].tolist()
    ctx.logger.info(
        "Numeric comparison matched %s out of %s records.",
        len(surviving_indices),
        len(df),
    )

    final_records = [parsed_records[i] for i in surviving_indices]
    content = adapter.serialize(final_records, config=config)
    ctx.materialize_content(
        output_key="filtered_data",
        content=content,
        extension=ext,
        metadata={
            "column": normalized_column,
            "expression": expression,
            "count": len(surviving_indices),
        },
    )
