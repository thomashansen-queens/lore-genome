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


class JoinConflict(StrEnum):
    """How to handle column name collisions"""
    SUFFIX = "suffix"
    DROP_LEFT = "drop_left"
    DROP_RIGHT = "drop_right"


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
    key_regex_clean = lore.ValueInput(
        str | None,
        default=None,
        label="Column cleaning regex",
        description="Optional regex pattern to apply to column values before joining. For example, to remove version suffixes from NCBI accessions, use: \\.\\d+",
    )
    conflict_strategy = lore.ValueInput(
        JoinConflict,
        default=JoinConflict.SUFFIX,
        label="Column name conflict strategy",
        description="How to handle identitcal column names when joining tables.",
    )


class JoinTablesOutputs:
    """Outputs for JoinTables Task"""
    joined_table = lore.TaskOutput(
        data_type=lore.Passthrough("left_table"),  # inherits the data type of the left table
        label="Joined Table",
        is_primary=True,
    )


def check_column(df: pd.DataFrame, target_col: str, table_name: str):
    """Check if a column exists in a DataFrame (permissive)."""
    # 1. Direct match
    if target_col in df.columns:
        return target_col

    # 2. Case-insensitive match
    target_lower = target_col.lower()
    for actual_col in df.columns:
        if actual_col.lower() == target_lower:
            return actual_col

    # 3. Fail
    raise ValueError(
        f"Column '{target_col}' not found in {table_name}. Available columns: {list(df.columns)}"
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
    key_regex_clean: str | None = None,
    conflict_strategy: JoinConflict = JoinConflict.SUFFIX,
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
    left_on = check_column(df_left, left_on, "Left Table")

    if right_on is None:
        right_on = left_on
    else:
        right_on = check_column(df_right, right_on, "Right Table")

    # 3. Key cleaning (optional)
    if key_regex_clean:
        ctx.logger.info(f"Applying regex '{key_regex_clean}' to join keys...")
        df_left["_join_key"] = df_left[left_on].astype(str).str.replace(key_regex_clean, "", regex=True)
        df_right["_join_key"] = df_right[right_on].astype(str).str.replace(key_regex_clean, "", regex=True)

    # 4. Merge conflict resolution
    col_collisions = set(df_left.columns).intersection(set(df_right.columns)) - {"_join_key", left_on, right_on}
    if col_collisions:
        ctx.logger.warning(f"Resolving overlapping columns: {col_collisions}. Strategy: {conflict_strategy.value}")
        if conflict_strategy == JoinConflict.DROP_LEFT:
            df_left = df_left.drop(columns=list(col_collisions))
        elif conflict_strategy == JoinConflict.DROP_RIGHT:
            df_right = df_right.drop(columns=list(col_collisions))

    # 5. Perform the join (simple pandas)
    df_left[left_on] = df_left[left_on].astype(str)
    df_right[right_on] = df_right[right_on].astype(str)

    ctx.logger.debug(f"Performing '{how}' join on columns '{left_on}' and '{right_on}'...")
    try:
        df_joined = pd.merge(
            df_left,
            df_right,
            left_on=left_on if not key_regex_clean else "_join_key",
            right_on=right_on if not key_regex_clean else "_join_key",
            how=how.value,
            suffixes=("_left", "_right"),  # if strategy is DROP, this will never be used
        )
    except Exception as e:
        ctx.logger.error(f"Error during join operation: {e}")
        raise RuntimeError(f"Error during join operation: {e}") from e

    if "_join_key" in df_joined.columns:
        df_joined = df_joined.drop(columns=["_join_key"])

    # 4. Materialize the output table as a new artifact
    out_path = ctx.get_temp_path("joined_table.tsv")
    df_joined.to_csv(out_path, sep="\t", index=False)

    ctx.materialize_file(
        source=out_path,
        output_key="joined_table",
    )
