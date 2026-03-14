"""
Task for merging a list of records into a single file.
"""
import pandas as pd
from typing import Any

import lore.dsl as lore


class MergeInputs:
    """Input model for the merge task."""
    source = lore.ArtifactInput(
        label="Artifact(s) to merge",
        accepted_data="*",
        select=lore.MULTIPLE,
        load_as=lore.RAW,
    )
    deduplicate = lore.ValueInput(
        bool,
        label="Deduplicate records",
        description="Remove identical records.",
        default=False,
    )
    group_by = lore.ValueInput(
        list[str] | None,
        label="Group by key(s)",
        description="If deduplicating tabular data, only look at these specific columns to determine uniqueness (e.g., 'protein_accession')",
        default=None,
        examples=[["protein_accession"]],
    )


class MergeOutputs:
    """Output model for the merge task."""
    merged_data = lore.TaskOutput(
        data_type="dynamic (override in handler)",
        label="Merged Data",
        description="A new Artifact containing the merged records.",
        is_primary=True,
    )


@lore.task(
    'merge',
    inputs=MergeInputs,
    outputs=MergeOutputs,
    name="Merge Artifacts",
    category="Data processing",
    icon="⊕",
    live_preview=True,
)
def merge_handler(
    ctx: lore.ExecutionContext,
    source: list[Any],
    deduplicate: bool = False,
    group_by: list[str] | None = None,
):
    """
    Merge a list of records into a single file
    """
    # 1. Get adapter and input artifact metadata
    adapter = ctx.get_input_adapter("source")
    if adapter is None:
        raise ValueError("No adapter found for the input Artifact(s).")

    source_artifacts = ctx.input_artifacts.get("source", [])
    if not source_artifacts:
        raise ValueError("No source Artifacts provided.")

    inherited_type = source_artifacts[0].data_type
    ext = source_artifacts[0].extension

    # TODO: --- Ultrafast shortcut for live previews of huge data ---
    # if getattr(ctx, "is_preview", False) or ctx.__class__.__name__ == "PreviewContext":
    #     ctx.logger.info("Generating stratified live preview...")

    #     preview_records = []
    #     # Calculate how many records to pull from each file (aiming for ~32 total)
    #     records_per_file = max(1, 32 // len(source))

    #     for file_content in source:
    #         # Stream a few records from each file
    #         stream = adapter.adapt_stream(iter(file_content.splitlines()))
    #         for i, record in enumerate(stream):
    #             if i >= records_per_file:
    #                 break
    #             preview_records.append(record)

    #     ctx.logger.info("Sampled %s records across %s files.", len(preview_records), len(source))

    #     # Serialize just the tiny sample
    #     ctx.materialize_content(
    #         output_key="merged_data",
    #         content=adapter.serialize(preview_records),
    #         extension=ext,
    #         data_type=inherited_type,
    #         metadata={"preview": True, "note": "Stratified sample across all files"},
    #     )
    #     return

    # 2. Deduplicate
    final_records = source

    if group_by and not deduplicate:
        ctx.logger.warning(
            "Group by keys provided but no deduplication not requested!"
        )
        group_by = None

    if not deduplicate:
        # Fast path: skip deduplication and just concatenate if requested.
        ctx.logger.info("Deduplication disabled. Performing high-speed raw concatenation.")

        # Join files, ensuring they are separated by a newline
        merged_content = "\n".join(s.strip() for s in source if s.strip()) + "\n"
        record_count = "unknown (raw merge)"

    else:
        # Slow path: Parse and deduplicate records in-memory.
        ctx.logger.info("Deduplication enabled. Parsing data into memory...")

        # 1. Parse into records/table through the Adapter layer
        all_records = []
        for file_content in source:
            all_records.extend(adapter.adapt(file_content))

        df = pd.DataFrame(all_records)
        if df.empty:
            raise ValueError("No records found in the input data.")

        # 2. Drop duplicates
        if group_by:
            missing = [k for k in group_by if k not in df.columns]
            if missing:
                group_by = [k for k in group_by if k in df.columns]

                if not group_by:
                    ctx.logger.warning("Group by key %s not found. No valid keys remain.", missing)
                    df = df.drop_duplicates()
                else:
                    ctx.logger.warning("Group by keys %s not found. Proceeding with valid keys.", missing)
                    df = df.drop_duplicates(subset=group_by)
            else:
                df = df.drop_duplicates(subset=group_by)

        # 3. Serialize back to original format
        final_records = df.to_dict(orient="records")
        record_count = len(final_records)

        ctx.logger.info("Deduplicated %s -> %s records.",
                        format(len(all_records), ","), format(record_count, ","))

        merged_content = adapter.serialize(final_records, extension=ext)

    # 3. Materialize the merged data and log the count of surviving records
    ctx.materialize_content(
        output_key="merged_data",
        content=merged_content,
        extension=ext,
        data_type=inherited_type,
        metadata={
            "group_by": group_by,
            "original_count": len(source),
            "count": record_count,
        },
    )
