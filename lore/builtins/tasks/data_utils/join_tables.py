"""
Utility task to join two tabular artifacts
"""
from collections.abc import Iterator
from enum import StrEnum
import lore
import pandas as pd


class JoinHow(StrEnum):
    """Enumeration of join types for JoinTables Task"""
    INNER = "inner"
    OUTER = "outer"
    LEFT = "left"
    RIGHT = "right"


class JoinTablesInputs:
    """Inputs for JoinTables Task"""
    left_table = lore.ArtifactInput(
        accepted_data=["tabular"],  # trait
        select="single",
        load_as="adapted_stream",
        label="Left Table",
        description="The first table to join.",
    )
    right_table = lore.ArtifactInput(
        accepted_data=["tabular"],
        select="single",
        load_as="adapted_stream",
        label="Right Table",
        description="The second table to join.",
    )
    left_on = lore.ValueInput(
        str,
        label="Left Column Name",
        description="The name of the column in the Left Table to join on.",
    )
    right_on = lore.ValueInput(
        str | None,
        default=None,
        label="Right Column Name",
        description="The name of the column in the Right Table to join on. Defaults to the Left Column Name if left blank.",
    )
    how = lore.ValueInput(
        JoinHow,
        default=JoinHow.INNER,
        label="Join type",
        description="Type of merge to be performed.",
    )


class JoinTablesOutputs:
    """Outputs for JoinTables Task"""
    joined_table = lore.TaskOutput(
        data_type=lore.Passthrough("left_table"),  # inherits the data type of the left table
        label="Joined Table",
        is_primary=True,
    )


@lore.task(
    "data.join_tables",
    inputs=JoinTablesInputs,
    outputs=JoinTablesOutputs,
    name="Join Tables",
    category="Data Utilities",
    preview_mode="full",
    icon="⏵⏴",
)
def join_tables(
    ctx: lore.ExecutionContext,
    left_table: Iterator[dict],
    right_table: Iterator[dict],
    left_on: str,
    right_on: str | None,
    how: JoinHow = JoinHow.INNER,
):
    """
    Joins two tabular artifacts based on a common key/column.
    """
    # 1. Consume the input tables as streams for maximum RAM efficiency
    # NOTE: There are more memory-efficient ways to join large tables, but they have computational cost
    try:
        ctx.logger.debug("Consuming left table stream into DataFrame...")
        df_left = pd.DataFrame(left_table)
        ctx.logger.debug("Consuming right table stream into DataFrame...")
        df_right = pd.DataFrame(right_table)
    except MemoryError as e:
        ctx.logger.error("Out of memory: The input tables are too large to join in memory.")
        raise RuntimeError("Out of memory: The input tables are too large to join in memory.") from e

    if df_left.empty:
        raise ValueError("Left table is empty after adaptation.")
    if df_right.empty:
        raise ValueError("Right table is empty after adaptation.")

    # 2. Validate columns and configuration
    if left_on not in df_left.columns:
        raise ValueError(
            f"Column '{left_on}' not found in Left Table. Available: {list(df_left.columns)}"
        )

    if right_on is None:
        right_on = left_on
    if right_on not in df_right.columns:
        raise ValueError(
            f"Column '{right_on}' not found in Right Table. Available: {list(df_right.columns)}"
        )

    # 3. Perform the join (simple pandas)
    df_left[left_on] = df_left[left_on].astype(str)
    df_right[right_on] = df_right[right_on].astype(str)

    ctx.logger.debug(f"Performing '{how}' join on columns '{left_on}' and '{right_on}'...")
    try:
        df_joined = pd.merge(
            df_left,
            df_right,
            left_on=left_on,
            right_on=right_on,
            how=how.value,
        )
    except Exception as e:
        ctx.logger.error(f"Error during join operation: {e}")
        raise RuntimeError(f"Error during join operation: {e}") from e

    # 4. Materialize the output table as a new artifact
    out_path = ctx.get_temp_path("joined_table.tsv")
    df_joined.to_csv(out_path, sep="\t", index=False)

    ctx.materialize_file(
        source=out_path,
        output_key="joined_table",
    )
