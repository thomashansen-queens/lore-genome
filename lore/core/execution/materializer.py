"""
The LoRē magic materializer: Transforms raw input references (artifact IDs)
into real, usable data for Task handlers based on instruction from the Task
Definition and available Adapters. This is where the "magic" happens: no
data transformation, ETL, or heavy lifting should be done in Task handlers.
Slicing and dicing of data are done here according to DSL instructions and
semantic types.
"""
from dataclasses import dataclass, field
import io
import itertools
import types
from typing import TYPE_CHECKING, Any, get_origin
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from lore.core.bindings import Binding, LiteralBinding, ReferenceBinding, UserInputBinding
from lore.core.io import get_reader_for
from lore.core.tasks import AdapterStrategy, Materialization, Cardinality, TaskDefinition
from lore.core.utils.pydantic import is_collection_type


if TYPE_CHECKING:
    from lore.core.sessions import Session
    from lore.core.artifacts import Artifact


@dataclass
class MaterializedInputs:
    """Data transfer object for materialized dependencies and associated data"""
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, list["Artifact"]] = field(default_factory=dict)
    io_metadata: dict[str, list[dict]] = field(default_factory=dict)


def materialize_task_inputs(
    s: "Session",
    task_def: "TaskDefinition",
    bindings: dict[str, list[Binding]],
    strategy: AdapterStrategy = AdapterStrategy.AUTO,
) -> MaterializedInputs:
    """
    Resolves raw input references (i.e. Artifacts) to actual data based on the
    TaskDefinition's DSL instructions (Materialization, series extraction) and
    available Adapters.

    Returns: MaterializedInputs object containing the materialized inputs and
    associated metadata.
    """
    dto = MaterializedInputs()

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
                    dto.data[key] = TypeAdapter(field_info.annotation).validate_python(vals)
                else:
                    if len(vals) >  1:
                        raise ValueError(
                            f"Input '{key}' does not allow multiple values, "
                            f"but got {len(vals)}: {vals}. Check pipeline connections."
                        )
                    raw_val = vals[0] if vals else None
                    if raw_val is not None:
                        dto.data[key] = TypeAdapter(field_info.annotation).validate_python(raw_val)
                    else:
                        dto.data[key] = None
            except Exception as e:
                raise ValueError(
                    f"Failed to cast input '{key}' to type {field_info.annotation}: {e}"
                )
            continue

        # 3. Extract Metadata from DSL
        materialization = extra.get("materialization")
        cardinality = extra.get("cardinality")
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

        dto.artifacts[key] = artifacts

        # 5. Process each Artifact according to the instructions.
        # Prepare for auto-concatenate if multiple items are provided.
        processed_artifacts = []
        slot_io_metadata = []

        for a in artifacts:
            item_data, io_meta = _materialize_single_artifact(
                session=s,
                artifact=a,
                materialization=materialization,
                accepted_data=accepted_data,
                strategy=strategy,
            )
            processed_artifacts.append(item_data)
            slot_io_metadata.append(io_meta)

        # 6. Handle packaging & concatenation
        # Auto-concatenate if it's a series type and multiple items are allowed
        if allows_multiple:
            if processed_artifacts and isinstance(processed_artifacts[0], types.GeneratorType):
                dto.data[key] = itertools.chain(*processed_artifacts)
            else:
                flattened = []
                for item in manual_inputs + processed_artifacts:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)
                dto.data[key] = flattened
        else:
            total_inputs = len(manual_inputs) + len(processed_artifacts)
            if total_inputs > 1:
                raise ValueError(
                    f"Input '{key}' does not allow multiple items, but got "
                    f"{total_inputs} (including {len(manual_inputs)} manual inputs)"
                )

            if manual_inputs:
                dto.data[key] = manual_inputs[0]  # manual input takes precedence
            else:
                dto.data[key] = processed_artifacts[0] if processed_artifacts else None

        # Store IO ledger for this slot
        dto.io_metadata[key] = slot_io_metadata

    return dto


def _materialize_single_artifact(
    session: "Session",
    artifact: "Artifact",
    materialization: str,
    accepted_data: list[str],
    strategy: AdapterStrategy = AdapterStrategy.AUTO,
) -> tuple[Any, dict]:
    """
    Helper to Materialize an Artifact into real data per DSL instructions
    If loading as CONTENT, will prioritize the narrowest type of accepted data
    i.e. Series > Adapted > Raw

    returns: Tuple of (materialized data, io_metadata)
    io_metadata is a dict that can include info like total_rows, truncated, etc.
    that is currently only populated for PEEK previews. Could be useful in the
    future to include io metadata for other materialization strategies.
    """
    m = Materialization(materialization)
    path = session.get_artifact_path(artifact.id)
    peek_limit = session.runtime.settings.preview_peek_limit
    io_meta = {}

    # --- Adapter can override Task contract ---

    if strategy == AdapterStrategy.PEEK:
        if m in (Materialization.ADAPTED, Materialization.ADAPTED_STREAM):
            m = Materialization.PREVIEW
        elif m in (Materialization.RAW, Materialization.RAW_STREAM):
            reader = get_reader_for(path)
            raw_data, io_meta = reader.preview(peek_limit=peek_limit)
            return raw_data, io_meta

    elif strategy == AdapterStrategy.LAZY:
        if m == Materialization.ADAPTED:
            m = Materialization.ADAPTED_STREAM
        elif m == Materialization.RAW:
            m = Materialization.RAW_STREAM

    elif strategy == AdapterStrategy.EAGER:
        if m == Materialization.ADAPTED_STREAM:
            m = Materialization.ADAPTED
        elif m == Materialization.RAW_STREAM:
            m = Materialization.RAW

    # --- Post-override: Follow DSL instructions ---

    # 1. Pure Manifest lookup
    if m == Materialization.ARTIFACT:
        return artifact, io_meta

    # 2. The handler will access the filepath directly
    if m == Materialization.PATH:
        return str(path), io_meta

    # 3. Read data for the handler
    reader = get_reader_for(path)

    if m == Materialization.RAW:
        return reader.read_full(), {"io_strategy": "Full load"}

    if m == Materialization.RAW_STREAM:
        return reader.stream(), {"io_strategy": "Streamed"}

    # 4. Adapt the data for the handler - metadata is passed as config
    adapters = artifact.get_adapters()
    adapter = adapters[0] if adapters else None
    config = {**(artifact.metadata or {}), "ext": artifact.extension}

    if m == Materialization.ADAPTED_STREAM:
        raw_generator = reader.stream()
        data = adapter.adapt_stream(raw_generator, config=config) if adapter else raw_generator
        return data, {"io_strategy": "Streamed (Adapted)"}

    if m == Materialization.PREVIEW:
        raw_data, io_meta = reader.preview(peek_limit=peek_limit)
        if adapter:
            return adapter.adapt(raw_data, config=config), io_meta
        return raw_data, io_meta

    if m == Materialization.ADAPTED:
        raw_data = reader.read_full()
        io_meta = {"io_strategy": "Full load (Adapted)"}

        # A. Try to provide a series (only for TabularAdapters)
        for adapter in adapters:
            for accepted in accepted_data:
                series = adapter.get_series(raw_data, accepted, config=config)
                if series is not None:
                    return series, io_meta

            # B. If no series, try adapting the entire payload
            adapted_data = adapter.adapt(raw_data, config=config)
            return adapted_data, io_meta

        # C. Fallback to raw content if no adapters worked
        return raw_data, io_meta

    return artifact.id, io_meta  # fallback to ID if no instructions


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
