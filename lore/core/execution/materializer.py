"""
Inputs materialization logic for Task execution. Turns Artifact references into
real data based on TaskDefinition instructions and available Adapters.
"""

import io
import itertools
import types
from typing import TYPE_CHECKING, Any, get_origin
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from lore.core.adapters import TableAdapter
from lore.core.bindings import Binding, LiteralBinding, ReferenceBinding, UserInputBinding
from lore.core.io import get_reader_for
from lore.core.tasks import Materialization, Cardinality, TaskDefinition
from lore.core.utils.pydantic import is_collection_type


if TYPE_CHECKING:
    from lore.core.sessions import Session
    from lore.core.artifacts import Artifact


def materialize_task_inputs(
    s: "Session",
    task_def: "TaskDefinition",
    bindings: dict[str, list[Binding]],
) -> tuple[dict, dict]:
    """
    Resolves raw input references (i.e. Artifacts) to actual data based on the
    TaskDefinition's DSL instructions (Materialization, series extraction) and
    available Adapters.
    """
    resolved = {}
    input_artifacts_snapshot = {}  # key: input_name, value: list[Artifact] or Artifact

    for key, binding_list in bindings.items():
        # 1. Get input field metadata from TaskDefinition
        field_info, extra = task_def.field_meta(key)

        # 2. Primitives / Non-artifact fields
        if not extra.get("is_artifact"):
            vals = [
                b.value for b in binding_list 
                if isinstance(b, (LiteralBinding, UserInputBinding))
            ]
            try:
                if is_collection_type(field_info.annotation):
                    resolved[key] = TypeAdapter(field_info.annotation).validate_python(vals)
                else:
                    if len(vals) >  1:
                        raise ValueError(
                            f"Input '{key}' does not allow multiple values, "
                            f"but got {len(vals)}: {vals}. Check pipeline connections."
                        )
                    raw_val = vals[0] if vals else None
                    if raw_val is not None:
                        resolved[key] = TypeAdapter(field_info.annotation).validate_python(raw_val)
                    else:
                        resolved[key] = None
            except Exception as e:
                raise ValueError(
                    f"Failed to cast input '{key}' to type {field_info.annotation}: {e}"
                )
            continue

        # 3. Extract Metadata from DSL
        materialization = extra.get("load_as")
        cardinality = extra.get("select")
        if not materialization or not cardinality:
            raise ValueError(
                f"Task input '{key}' missing load_as (`materialization`) or select (`cardinality`)"
            )
        accepted_data = extra.get("accepted_data", [])
        allows_multiple = Cardinality(cardinality).allows_multiple

        # 4. Resolve IDs to Artifacts (bulk fetch then snapshot)
        artifacts = []
        manual_inputs = []

        for b in binding_list:
            if not b:
                continue  # val is falsy (e.g. None, empty)

            if isinstance(b, ReferenceBinding):
                if b.artifact_id:
                    # Pinned edge: Grab the specific Artifact
                    artifact = s.get_artifact(b.artifact_id)
                    if artifact:
                        artifacts.append(artifact)
                else:
                    # Unpinned edge: Get artifact(s) from the output
                    task = s.get_task(b.source_id)
                    if not task:
                        raise ValueError(f"Upstream task ID '{b.source_id}' not found")
                    
                    output_artifacts = task.outputs.get(b.output_key, [])
                    if not output_artifacts:
                        # No artifacts produced for this output key
                        continue

                    # For unpinned edges, do not raise on cardinality mismatch
                    if not allows_multiple:
                        artifact = s.get_artifact(output_artifacts[-1])
                        if artifact:
                            artifacts.append(artifact)
                    else:
                        for output_artifact_id in output_artifacts:
                            artifact = s.get_artifact(output_artifact_id)
                            if artifact:
                                artifacts.append(artifact)

            elif isinstance(b, (LiteralBinding, UserInputBinding)):
                if b.value is not None and b.value != "":
                    artifact = s.get_artifact(str(b.value))
                    if artifact:
                        # Literal reference to an existing Artifact ID
                        artifacts.append(artifact)
                    else:
                        # Manually inputted value that is not an Artifact ID
                        manual_inputs.append(
                            _materialize_manual_input(b.value, materialization, field_info)
                        )

        input_artifacts_snapshot[key] = artifacts

        # 5. Process each Artifact according to the instructions.
        # Prepare for auto-concatenate if multiple items are provided.
        processed_artifacts = []
        for a in artifacts:
            item_data = _materialize_single_artifact(s, a, materialization, accepted_data)
            processed_artifacts.append(item_data)

        # 6. Handle packaging & concatenation
        # Auto-concatenate if it's a series type and multiple items are allowed
        if allows_multiple:
            if processed_artifacts and isinstance(processed_artifacts[0], types.GeneratorType):
                resolved[key] = itertools.chain(*processed_artifacts)
            else:
                flattened = []
                for item in manual_inputs + processed_artifacts:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)
                resolved[key] = flattened
        else:
            total_inputs = len(manual_inputs) + len(processed_artifacts)
            if total_inputs > 1:
                raise ValueError(
                    f"Input '{key}' does not allow multiple items, but got "
                    f"{total_inputs} (including {len(manual_inputs)} manual inputs)"
                )

            if manual_inputs:
                resolved[key] = manual_inputs[0]  # manual input takes precedence
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
