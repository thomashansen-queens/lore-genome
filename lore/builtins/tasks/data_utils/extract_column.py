"""
Extract a specific column/series from a table.
"""
import itertools
from collections.abc import Iterator

import lore


class ExtractColumnInputs:
    """Inputs for ExtractColumn Task"""
    table = lore.ArtifactInput(
        select="single",
        load_as="adapted_stream",
        accepted_data="tabular",
        label="Input table",
    )
    column_name = lore.ValueInput(
        str,
        label="Column to extract",
        description="The name of the column to extract from the table.",
    )
    new_name = lore.ValueInput(
        str | None,
        label="New column name (optional)",
        description="Optional new name for the extracted column. If not provided, the original column name will be used.",
        default=None,
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
    new_name: str | None = None,
    deduplicate: bool = False,
):
    """
    Extract a specific column/series from a table.
    """
    out_path = ctx.get_temp_path(column_name + ".tsv")

    # 1. Check if the column exists in the first row (header)
    try:
        first_row = next(table)
    except StopIteration:
        raise ValueError("The input table is empty.")

    # Case-insensitive column lookup
    actual_key = next((k for k in first_row.keys() if k.lower() == column_name.lower()), None)

    if not actual_key:
        raise ValueError(
            f"Column '{column_name}' not found in the table. Available columns: {first_row}"
        )

    unique_values = set()

    # 2. Stream column to disk. Chain first_row back in: it was pulled off the
    #    stream for the header check above and must still be written, or every
    #    extraction silently drops its first record (and single-row tables yield nothing).
    with open(out_path, "w", encoding="utf-8") as out_file:
        # If CSV is malformed/truncated
        for row in itertools.chain([first_row], table):
            if len(row) != len(first_row):
                ctx.logger.warning(
                    "Row length %s does not match header length %s. Skipping row: %s",
                    len(row), len(first_row), row,
                )
                continue

            # Get value and skip if empty or duplicate and deduplicate is True
            val = str(row.get(actual_key, "")).strip()
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
    out_name = new_name if new_name else column_name
    ctx.materialize_file(
        source=out_path,
        output_key="column",
        data_type=out_name.lower().strip(),
        extension="tsv",
        metadata={
            "columns": [out_name],
            "deduplicate": deduplicate,
            "header": False,
        }
    )
