"""
The LoRē magic materializer: Transforms raw input references (artifact IDs)
into real, usable data for Task handlers based on instruction from the Task
Definition and available Adapters. This is where the "magic" happens: no
data transformation, ETL, or heavy lifting should be done in Task handlers.
Slicing and dicing of data are done here according to DSL instructions and
semantic types.
"""
from collections.abc import Iterator
from dataclasses import dataclass, field
import io
import itertools
import logging
import re
from typing import TYPE_CHECKING, Any, get_origin
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from lore.core.bindings import Binding, LiteralBinding, ReferenceBinding, UserInputBinding
from lore.core.readers import get_reader_for
from lore.core.tasks import AdapterStrategy, Materialization, Cardinality, TaskDefinition
from lore.core.utils.pydantic import is_collection_type


if TYPE_CHECKING:
    from lore.core.sessions import Session
    from lore.core.artifacts import Artifact
    from lore.core.readers import BaseReader

logger = logging.getLogger(__name__)


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
            # A lazy stream (generator, itertools.chain, ...) is chained lazily;
            # materialized values (lists, scalars) fall through to flatten. Note a
            # list is Iterable but not an Iterator, so it correctly takes the else.
            if processed_artifacts and isinstance(processed_artifacts[0], Iterator):
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


# Only warn on full-read fallbacks if the file is large enough to matter
_FALLBACK_WARN_BYTES = 10 * 1024 * 1024  # 10 MB


def _stream_with_fallback(
    reader: "BaseReader",
    config: dict | None = None,
    *,
    adapted: bool = False,
) -> tuple[Iterator[Any], dict]:
    """
    Resolve a reader to (record_iterator, io_metadata), transparently falling
    back to a whole-file read_full() for non-streamable formats (those whose
    stream() raises NotImplementedError, e.g. a monolithic JSON document).
    Checking first iteration raises here rather than inside handler code.
    """
    suffix = " (Adapted)" if adapted else ""
    gen = reader.stream(config)
    try:
        first = next(gen)
    except NotImplementedError:
        # Warn only when the full in-memory load is large enough to matter
        size = reader.path.stat().st_size if reader.path.exists() else 0
        if size >= _FALLBACK_WARN_BYTES:
            logger.warning(
                "%s cannot stream '%s' (%.0f MB); fell back to a full in-memory read, "
                "forfeiting streaming's memory efficiency. Consider a streaming reader "
                "for this format.",
                type(reader).__name__, reader.path.name, size / 1024 / 1024,
            )
        # Then read the whole file but wrap it in an iterator for streaming handlers
        data = reader.read_full(config)
        return iter(data), {"io_strategy": f"Full load{suffix}"}
    except StopIteration:
        return iter(()), {"io_strategy": f"Streamed{suffix}"}  # empty stream
    return itertools.chain([first], gen), {"io_strategy": f"Streamed{suffix}"}


def _materialize_single_artifact(
    session: "Session",
    artifact: "Artifact",
    materialization: str,
    accepted_data: list[str],
    strategy: AdapterStrategy = AdapterStrategy.AUTO,
) -> tuple[Any, dict]:
    """
    Materialize one Artifact into the data a handler asked for, per the DSL.

    Two things to consider when materializing an Artifact:
      - Shape (from TaskDefinition's 'load_as'): A pointer (ARTIFACT / PATH), the
        reader's raw records (RAW*), or adapted records (ADAPTED*). For ADAPTED,
        if 'accepted_data' names a column the adapter can slice, the handler gets
        that column as a flat list.
      - Delivery (from 'strategy'): How the handler wants the data. 'stream' (lazy,
        O(1) memory) vs materialized, and how much is read (a PEEK for fast UI
        previews).
    returns: (data, io_metadata)
    """
    m = Materialization(materialization)

    # 1. Pointer to a file or artifact: No reading or adapting, just give to handler
    if m == Materialization.ARTIFACT:
        return artifact, {}
    if m == Materialization.PATH:
        bundle = session.get_artifact_path_bundle(artifact.id)
        return bundle, {}

    path = session.get_artifact_path(artifact.id)
    reader = get_reader_for(path)
    config = {**(artifact.metadata or {}), "ext": artifact.extension}
    peek_limit = session.runtime.settings.preview_peek_limit

    # 2. Resolve delivery: 'load_as' sets the defaults. LAZY forces streaming, EAGER forces a
    #    full load, PEEK bounds the read.
    wants_adapted = m in (Materialization.ADAPTED, Materialization.ADAPTED_STREAM)
    stream = m in (Materialization.RAW_STREAM, Materialization.ADAPTED_STREAM)
    if strategy == AdapterStrategy.LAZY:
        stream = True
    elif strategy == AdapterStrategy.EAGER:
        stream = False
    peek = strategy == AdapterStrategy.PEEK

    # 3. Acquire the reader's records. Streaming is a lazy full read. Peek
    #    bounds a materialized read. io_metadata reports reader completeness.
    if stream and not peek:
        records, io_meta = _stream_with_fallback(reader, config, adapted=wants_adapted)
    elif stream:  # peeked stream: partial load, but still handed back as an iterator
        peeked, io_meta = reader.preview(peek_limit=peek_limit)
        records = iter(peeked)
    elif peek:
        records, io_meta = reader.preview(peek_limit=peek_limit)
    else:
        label = "Full load (Adapted)" if wants_adapted else "Full load"
        records, io_meta = reader.read_full(), {"io_strategy": label}

    # 4. RAW: hand the reader's records straight to the handler
    if not wants_adapted:
        return records, io_meta

    # 5. ADAPTED: run records through the best-matching adapter. No adapter
    #    available? Best-effort raw passthrough
    adapters = artifact.get_adapters()
    if not adapters:
        return records, io_meta

    adapter = adapters[0]
    adapted = adapter.adapt_stream(records, config=config) if stream else adapter.adapt(records, config=config)

    # 5a. Peek at first record to see if accepted_data is present as a column
    first_record = None
    if stream:
        adapted = iter(adapted)
        try:
            first_record = next(adapted)
            adapted = itertools.chain([first_record], adapted)
        except StopIteration:
            pass  # Empty stream
    elif isinstance(adapted, list):
        first_record = adapted[0] if adapted else None

    target_col = next(
        (col for col in accepted_data if isinstance(first_record, dict) and col in first_record),
        None,
    )
    if target_col is None:
        return adapted, io_meta

    # 5b. Apply slicing. Generator for streaming, list for materialized.
    if stream:
        sliced = (str(r[target_col]) for r in adapted if r.get(target_col) is not None)
        return sliced, io_meta
    else:
        sliced = [str(r[target_col]) for r in adapted if r.get(target_col) is not None]
        return sliced, io_meta


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
        For adapted content, attempt to coerce. This is a best-effort measure that
        lets users type comma-, tab-, and/or newline-separated values.
        """
        annotation = field_info.annotation
        origin = get_origin(annotation) or annotation

        # 1. Simulated series slicing: split on any run of commas/tabs/newlines
        if is_collection_type(annotation):
            return [v.strip() for v in re.split(r"[,\t\n]+", str(val)) if v.strip()]

        # 2. Simualte single primitive adaptation
        if origin is str:
            return str(val).strip()

        # 3. Hard stop (cannot provide expected type e.g. list[FastaRecord])
        raise ValueError(
            f"Failed to coerce manual input '{val}' for field '{field_info.title}'. "
            f"Expected Artifact ID list or string, got {origin}."
        )

    if m == Materialization.ADAPTED:
        return pseudo_adapt(value)

    if m == Materialization.ADAPTED_STREAM:
        return iter(pseudo_adapt(value))

    raise ValueError(f"Unsupported materialization type for manual input: {materialization}")
