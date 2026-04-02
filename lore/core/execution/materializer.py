"""
Inputs materialization logic for Task execution. Turns Artifact references into
real data based on TaskDefinition instructions and available Adapters.
"""

import io
import itertools
import types
from typing import TYPE_CHECKING, Any, get_origin
from pydantic.fields import FieldInfo

from lore.core.adapters import TableAdapter
from lore.core.io import get_reader_for
from lore.core.tasks import Materialization, Cardinality, TaskDefinition


if TYPE_CHECKING:
    from lore.core.sessions import Session
    from lore.core.artifacts import Artifact


def materialize_task_inputs(
    s: "Session",
    task_def: "TaskDefinition",
    raw_inputs: dict,
) -> tuple[dict, dict]:
    """
    Resolves raw input references (i.e. Artifacts) to actual data based on the
    TaskDefinition's DSL instructions (Materialization, series extraction) and
    available Adapters.

    Note: Intentionally permits Duck Typing. If an Artifact ID is not found, the
    raw string will be passed through to the handler. This allows users to paste
    raw data (like comma-separated accessions) directly into Artifact fields.
    Alternatively, a user could pass comma-separated Artifact IDs and they will
    be resolved to Artifacts, then adapted/concatenated as instructed.
    """
    resolved = {}
    input_artifacts_snapshot = {}  # key: input_name, value: list[Artifact] or Artifact

    for key, value in raw_inputs.items():
        # 1. Get input field metadata from TaskDefinition
        field_info, extra = task_def.field_meta(key)

        if not extra.get("is_artifact"):
            # primitive (e.g. dict, str, int), pass through
            resolved[key] = value
            continue

        # 2. Extract Metadata from DSL
        materialization = extra.get("load_as")
        cardinality = extra.get("select")
        if not materialization or not cardinality:
            raise ValueError(
                f"Task input '{key}' missing load_as (`materialization`) or select (`cardinality`)"
            )
        accepted_data = extra.get("accepted_data", [])

        # 3. Resolve IDs to Artifacts (bulk fetch then snapshot)
        # Duck typing: If input is an artifact ID, treat it as one! Otherwise,
        # allow users to manually enter input for fields that accept artifacts
        resolved_list = []
        artifacts = []
        task_inputs = value if isinstance(value, list) else [value]

        for val in task_inputs:
            if not val:
                continue  # val is falsy (e.g. None, empty)

            artifact = s.get_artifact(val)

            if artifact is not None:
                artifacts.append(artifact)
            else:
                resolved_list.append(_materialize_manual_input(val, materialization, field_info))

        input_artifacts_snapshot[key] = artifacts

        # 4. Process each Artifact according to the instructions
        processed_artifacts = []  # auto-concatenate if multiple items
        for a in artifacts:
            item_data = _materialize_single_artifact(s, a, materialization, accepted_data)
            processed_artifacts.append(item_data)

        # 5. Handle packaging & concatenation
        # Auto-concatenate if it's a series type and multiple items are allowed
        if Cardinality(cardinality).allows_multiple:
            if processed_artifacts and isinstance(processed_artifacts[0], types.GeneratorType):
                resolved[key] = itertools.chain(*processed_artifacts)
            else:
                flattened = resolved_list.copy()  # start with non-input artifacts
                for item in processed_artifacts:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)
                resolved[key] = flattened
        else:
            total_inputs = len(resolved_list) + len(processed_artifacts)
            if total_inputs > 1:
                raise ValueError(
                    f"Input '{key}' does not allow multiple items, but got "
                    f"{total_inputs} (including {len(resolved_list)} manual inputs)"
                )

            if resolved_list:
                resolved[key] = resolved_list[0]  # manual input takes precedence
            else:
                resolved[key] = processed_artifacts[0] if processed_artifacts else None

    return resolved, input_artifacts_snapshot


def _materialize_single_artifact(
    s: "Session",
    artifact: "Artifact",
    materialization: str,
    accepted_data: list[str],
) -> Any:
    """
    Helper to Materialize an Artifact into real data per DSL instructions
    If loading as CONTENT, will prioritize the narrowest type of accepted data
    i.e. Series > Adapted > Raw
    """
    m = Materialization(materialization)

    # 1. Pure Manifest lookup
    if m == Materialization.ARTIFACT:
        return artifact

    # 2. The handler will access the filepath directly
    path = s.get_artifact_path(artifact.id)

    if m == Materialization.PATH:
        return str(path)

    # 3. Read data for the handler
    reader = get_reader_for(path)

    if m == Materialization.RAW:
        return reader.read_full()

    if m == Materialization.RAW_STREAM:
        return reader.stream()

    # 4. Adapt the data for the handler
    adapters = artifact.get_adapters()
    adapter = adapters[0] if adapters else None

    if m == Materialization.ADAPTED_STREAM:
        raw_generator = reader.stream()
        return adapter.adapt_stream(raw_generator) if adapter else raw_generator

    if m == Materialization.PREVIEW:
        raw_data, metadata = reader.preview(limit=100)
        if adapter:
            return adapter.adapt(raw_data)
        return raw_data

    if m == Materialization.ADAPTED:
        raw_data = reader.read_full()

        # A. Try to provide a series (only for TableAdapters)
        for adapter in adapters:
            adapted_data = adapter.adapt(raw_data)
            if not adapted_data:
                continue

            if (
                isinstance(adapter, TableAdapter)
                and isinstance(adapted_data, list)
                and isinstance(adapted_data[0], dict)
            ):
                actual_columns = adapted_data[0].keys()  # dynamically generated

                for accepted in accepted_data:
                    if accepted in actual_columns:
                        return [row[accepted] for row in adapted_data if accepted in row]

            # B. If no series, try adapting the entire payload
            return adapted_data

        # C. Fallback to raw content if no adapters worked
        return raw_data

    return artifact.id  # fallback to ID if no instructions


def _materialize_manual_input(
    value: Any,
    materialization: str,
    field_info: FieldInfo,
) -> Any:
    """
    Attempts to coerce a manual input value into the expected type for a handler.
    """
    m = Materialization(materialization)

    if m == Materialization.ARTIFACT:
        raise ValueError(
            f"Not a valid Artifact ID: {value}."
            f"Input for '{field_info.title}' strictly requires a LoRē Artifact record."
        )

    if m == Materialization.PATH:
        from pathlib import Path
        try:
            Path(value.strip()).resolve(strict=True)
            return str(value.strip())
        except FileNotFoundError:
            # log a warning and reject input or raise an error?
            raise ValueError(
                f"Manual input for '{field_info.title}' was evaluated as a File Path, "
                f"but no file exists at: {value}"
            )

    if m == Materialization.RAW:
        return value

    if m == Materialization.RAW_STREAM:
        # Does this need to be/should it be io.StringIO(str(value))?
        return io.StringIO(str(value))

    def pseudo_adapt(val: Any) -> Any:
        """
        For adapted content, attempt to coerce. This is a best-effort measure that allows users to 
        input comma-separated values.
        """
        annotation = field_info.annotation
        origin = get_origin(annotation) or annotation

        # 1. Simulated series slicing
        if origin in (list, set, tuple):
            return [v.strip() for v in str(val).split(",")]

        # 2. Simualte single primitive adaptation
        if origin is str:
            return str(val).strip()

        # 3. Hard stop (cannot provide expected type e.g. list[FastaRecord])
        raise ValueError(
            f"Failed to coerce manual input '{val}' for field '{field_info.title}'. "
            f"Expected Artifact ID list or string, got {origin}."
        )

    if m in (Materialization.ADAPTED, Materialization.PREVIEW):
        return pseudo_adapt(value)

    if m == Materialization.ADAPTED_STREAM:
        return iter(pseudo_adapt(value))

    raise ValueError(f"Unsupported materialization type for manual input: {materialization}")
