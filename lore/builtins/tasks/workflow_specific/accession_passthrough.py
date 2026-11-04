"""
Passthrough task for converting a list of accessions into a single-column table.
"""
import csv

import lore


class Inputs:
    accessions = lore.ValueInput(
        list[str],
        label="Accessions",
        description="Accessions to place in the output table.",
    )

    data_type = lore.ValueInput(
        str,
        label="Data Type",
        description="The lore data type of the outputted file (not to be confused with file extension).",
        default="csv"
    )

    column_name = lore.ValueInput(
        str,
        label="Column name",
        description="Name of the column containing the accessions.",
        default="accessions"
    )


class Outputs:
    accession_table = lore.TaskOutput(
        data_type="csv",
        label="Accession Table",
        is_primary=True,
    )


@lore.task(
    "workflow_specific.accession_passthrough",
    inputs=Inputs,
    outputs=Outputs,
    name="Accession Passthrough",
    category="Workflow-Specific",
    preview_mode="live",
)
def accession_passthrough(
    ctx: lore.ExecutionContext,
    accessions: list[str],
    data_type="csv",
    column_name="accessions"
):
    """Write the provided accessions to a single-column CSV table."""
    output_path = ctx.get_temp_path("accession_table.csv")

    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([column_name])
        writer.writerows([accession] for accession in accessions)

    ctx.materialize_file(
        source=output_path,
        output_key="accession_table",
        data_type=data_type
    )
