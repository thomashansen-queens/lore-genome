"""
Extract a specific column/series from a table.
"""
from collections.abc import Iterator

import lore


class ExtractColumnInputs:
    """Inputs for ExtractColumn Task"""
    table = lore.ArtifactInput(
        select="single",
        load_as="adapted_stream",
        accepted_data="tabular",
    )
    column_name = lore.ValueInput(
        str,
        label="Column Name",
        description="The name of the column to extract from the table.",
    )
    deduplicate = lore.ValueInput(
        bool,
        label="Deduplicate",
        description="Whether to remove duplicate values from the extracted column.",
        default=False,
    )


class ExtractColumnOutputs:
    """Outputs for ExtractColumn Task"""
    column = lore.TaskOutput(
        data_type="series",
        label="Extracted Column",
        is_primary=True,
    )


@lore.task(
    "extract_column",
    inputs=ExtractColumnInputs,
    outputs=ExtractColumnOutputs,
    name="Extract Column",
    category="Data processing",
    icon="✂",
    preview_mode="full",
)
def extract_column(
    ctx: lore.ExecutionContext,
    table: Iterator[dict],
    column_name: str,
    deduplicate: bool = False,
):
    """
    Extract a specific column/series from a table.
    """
    out_path = ctx.get_temp_path(column_name + ".txt")

    # 1. Check if the column exists in the first row (header)
    try:
        first_row = next(table)
    except StopIteration:
        raise ValueError("The input table is empty.")

    if column_name not in first_row:
        raise ValueError(
            f"Column '{column_name}' not found in the table. Available columns: {first_row}"
        )

    unique_values = set()

    # 2. Stream column to disk
    with open(out_path, "w", encoding="utf-8") as out_file:
        # If CSV is malformed/truncated
        for row in table:
            if len(row) != len(first_row):
                ctx.logger.warning(
                    "Row length %s does not match header length %s. Skipping row: %s",
                    len(row), len(first_row), row,
                )
                continue

            # Get value and skip if empty or duplicate and deduplicate is True
            val = str(row.get(column_name, "")).strip()
            if not val:
                continue
            if deduplicate:
                if val in unique_values:
                    continue
                unique_values.add(val)

            out_file.write(val + "\n")

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise ValueError(f"No values found for column '{column_name}'.")

    # 3. Materialize the output
    ctx.materialize_file(
        source=out_path,
        output_key="column",
        data_type=column_name,
        extension="txt",
    )
