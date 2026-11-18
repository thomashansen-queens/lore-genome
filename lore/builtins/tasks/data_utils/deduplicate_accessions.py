"""
Deduplicates a tabular artifact using an identifier and assembly or nucleotide
accessions.
"""

import csv
import io
import math
import re
from typing import Any

import lore


class DeduplicateAccessionsInputs:
    """Inputs for the accession-based deduplication task."""

    source = lore.ArtifactInput(
        label="Table to deduplicate",
        accepted_data="tabular",
        select="single",
        load_as="raw",
    )
    identifier_column = lore.ValueInput(
        str,
        label="Identifier column",
        description="Rows are deduplicated separately for each value in this column.",
    )
    use_nucleotide_accession = lore.ValueInput(
        bool,
        label="Use nucleotide accession fallback",
        description=(
            "When a row has no assembly accession, deduplicate it using its nucleotide "
            "accession instead."
        ),
        default=False,
    )


class DeduplicateAccessionsOutputs:
    """Outputs for the accession-based deduplication task."""

    deduplicated_data = lore.TaskOutput(
        data_type=lore.Passthrough("source"),
        label="Deduplicated data",
        description="The deduplicated table in its original format and data type.",
        is_primary=True,
    )
    removed_data = lore.TaskOutput(
        data_type="csv",
        label="Removed data",
        description="Rows removed as duplicates or because required accessions were missing.",
    )


def _normalize_column_name(column: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(column)).strip("_").lower()


def _find_column(records: list[dict], requested: str, purpose: str) -> str:
    target = _normalize_column_name(requested)
    matches = [column for column in records[0] if _normalize_column_name(column) == target]
    if not matches:
        raise ValueError(
            f"{purpose} column '{requested}' not found. Available columns: {list(records[0])}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"{purpose} column '{requested}' is ambiguous after column-name normalization: "
            f"{matches}"
        )
    return matches[0]


def _clean_accession(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    accession = str(value).strip()
    if not accession or accession.casefold() in {"nan", "none", "null", "<na>"}:
        return None
    return accession


def _assembly_key(accession: str) -> tuple[str, int]:
    match = re.fullmatch(r"(GCA|GCF)_(\d+)(?:\.(\d+))?", accession, re.IGNORECASE)
    if not match:
        return accession.casefold(), 0

    prefix, number, version = match.groups()
    return f"{number}.{version or ''}", int(prefix.upper() == "GCF")


def _removed_csv(records: list[dict]) -> str:
    fieldnames = list(dict.fromkeys(column for record in records for column in record))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


@lore.task(
    "data.deduplicate_accessions",
    inputs=DeduplicateAccessionsInputs,
    outputs=DeduplicateAccessionsOutputs,
    name="Deduplicate by accession",
    category="Data Utilities",
    description=(
        "Deduplicate rows per identifier by assembly accession, preferring GCF over "
        "its matching GCA accession, with an optional nucleotide-accession fallback."
    ),
    preview_mode="full",
)
def deduplicate_accessions(
    ctx: lore.ExecutionContext,
    source: Any,
    identifier_column: str,
    use_nucleotide_accession: bool = False,
):
    """Deduplicate by identifier and accession, preserving the source row format."""
    source_artifacts = ctx.input_artifacts.get("source", [])
    if not source_artifacts:
        raise ValueError("No source artifact provided.")

    source_artifact = source_artifacts[0]
    extension = source_artifact.extension
    data_type = source_artifact.data_type
    config = {**(source_artifact.metadata or {}), "ext": extension}

    adapter = ctx.get_input_adapter("source")
    if adapter is None:
        raise ValueError("No adapter found for the source artifact.")
    if not isinstance(adapter, lore.TabularAdapter):
        raise ValueError(
            f"The adapter for the source artifact must be a TabularAdapter, "
            f"but got {type(adapter)}."
        )

    records = adapter.parse(source, config=config)
    if not records:
        raise ValueError("The source table is empty.")

    adapted_records = adapter.adapt(records, config=config)
    if len(adapted_records) != len(records):
        raise ValueError("The tabular adapter changed the number of source rows.")

    identifier_key = _find_column(adapted_records, identifier_column, "Identifier")
    assembly_key = _find_column(adapted_records, "assembly_accession", "Assembly accession")
    nucleotide_key = _find_column(adapted_records, "nucleotide_accession", "Nucleotide accession")

    winners: dict[tuple[Any, ...], tuple[int, int]] = {}
    keep = [False] * len(records)

    for index, row in enumerate(adapted_records):
        identifier = _clean_accession(row.get(identifier_key))
        assembly = _clean_accession(row.get(assembly_key))

        if assembly is not None:
            accession_group, priority = _assembly_key(assembly)
            accession_kind = "assembly"
        elif use_nucleotide_accession:
            nucleotide = _clean_accession(row.get(nucleotide_key))
            if nucleotide is None:
                continue
            accession_group = nucleotide.casefold()
            accession_kind = "nucleotide"
            priority = 0
        else:
            continue

        group_key = (
            ("missing_identifier", index) if identifier is None else ("identifier", identifier)
        )
        key = (group_key, accession_kind, accession_group)
        previous = winners.get(key)
        if previous is None:
            winners[key] = (index, priority)
            keep[index] = True
        elif priority > previous[1]:
            keep[previous[0]] = False
            winners[key] = (index, priority)
            keep[index] = True

    kept_records = [record for record, keep_row in zip(records, keep, strict=True) if keep_row]
    removed_records = [
        record for record, keep_row in zip(records, keep, strict=True) if not keep_row
    ]
    ctx.logger.info(
        "Deduplicated %s rows to %s; %s rows written to removed_data.",
        len(records),
        len(kept_records),
        len(removed_records),
    )

    result = ctx.materialize_content(
        output_key="deduplicated_data",
        content=adapter.serialize(kept_records, config=config),
        extension=extension,
        data_type=data_type,
    )
    ctx.materialize_content(
        output_key="removed_data",
        content=_removed_csv(removed_records),
        name="removed_data",
        extension="csv",
        data_type="csv",
    )
    return result
