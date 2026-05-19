"""
Theoretical matching logic to connect Task outputs and Inputs into a
directed acyclic graph (DAG). Matching uses semantic types. Edges are
built from Task-Artifact relationships.
"""
from typing import TYPE_CHECKING, Any, Literal

from lore.core.adapters import BaseAdapter
from lore.core.bindings import Binding, LiteralBinding, ReferenceBinding, UserInputBinding
from lore.core.topology.traits import DataTrait

if TYPE_CHECKING:
    from lore.core.sessions.session import Session
    from lore.core.tasks.models import TaskDefinition


def is_primitive_compatible(source_type: Any, target_extra: dict) -> bool:
    """
    Checks if a primitive Python type (e.g. int, str, CustomModel) can satisfy 
    the data requirements of an input field (a ValueInput).
    """
    target_type = target_extra.get("annotated_type")
    if not target_type or not source_type:
        return False

    # True wildcard: Annotated as Any
    if target_type == "Any" or source_type == "Any":
        return True

    # Type checking: Can the source safely act as the target?
    try:
        # e.g. int -> float is compatible, but not float -> int
        return issubclass(source_type, target_type)
    except TypeError:
        # fallback for complex types e.g. list[int] or union types
        return source_type == target_type


def is_output_compatible(
    source_extra: dict,
    target_extra: dict,
    adapters: list["BaseAdapter"] | None = None,
) -> bool:
    """
    Evaluates whether a Task's output can satisfy the data requirements of an input field.
    """
    provided = str(source_extra.get("data_type", "unknown")).lower()
    accepted = target_extra.get("accepted_data", provided)
    adapters = adapters or []

    # 1. Normalize to a list for uniform processing
    if not isinstance(accepted, list):
        accepted = [accepted]

    # 2. True wildcard (input accepts anything)
    for requirement in accepted:
        # A. Trait match - Broad semantic category (e.g. "tabular")
        if isinstance(requirement, DataTrait):
            if requirement.is_satisfied_by(provided, adapters):
                return True

        # B. Direct match - Exact type match (e.g. "parquet")
        elif isinstance(requirement, str):
            req_lower = requirement.lower()
            if provided == req_lower:
                return True

            if any(adapter.provides(req_lower) for adapter in adapters):
                return True

    return False


def infer_bindings_from_raw(
    raw_inputs: dict[str, Any],
    session: "Session",
    ui_modes: dict[str, Literal["literal", "reference", "user_input"]] | None = None,
) -> dict[str, list[Binding]]:
    """
    From raw user inputs (strings, file uploads, artifact IDs), infer what
    bindings to create. If an Artifact ID is detected, find where that 
    Artifact was produced. If it came from a Task in the Session, create an
    edge (ReferenceBinding).
    If the input is an unpinned edge (ReferenceBinding with value using the
    UI syntax 'ref:task_id::output_key'), resolve it.
    """
    inferred_bindings = {}
    ui_modes = ui_modes or {}

    for key, val in raw_inputs.items():
        mode = ui_modes.get(key, "literal")

        # 1. Handle UserInputBinding
        if mode == "user_input":
            default_val = val if val not in (None, "", []) else None
            inferred_bindings[key] = [UserInputBinding(
                input_key=key,
                value=default_val,
            )]
            continue

        # 2. All Task inputs are lists of Bindings, even single values
        if isinstance(val, list):
            val_list = val
        else:
            val_list = [val] if val is not None and val != "" else []

        bindings = []

        # 3. Scan for Artifact references
        for item in val_list:
            if isinstance(item, str) and not item.strip():
                continue

            str_item = str(item).strip()

            # 4. Unpinned edge (will resolve at execution time)
            if str_item.startswith("ref:"):
                parts = str_item[4:].split("::", 1)
                if len(parts) == 2:
                    bindings.append(ReferenceBinding(
                        source_id=parts[0],
                        output_key=parts[1],
                        artifact_id=None,  # Unpinned edge
                    ))
                    continue

            # 5. Not an Artifact reference
            artifact = session.get_artifact(str_item)
            if not artifact:
                bindings.append(LiteralBinding(value=item))
                continue

            # 6. Artifact found! Pinned edge (concrete Artifact ID)
            if artifact.created_by_task_id and artifact.created_by_output_key:
                bindings.append(ReferenceBinding(
                    source_id=artifact.created_by_task_id,
                    output_key=artifact.created_by_output_key,
                    artifact_id=artifact.id  # ReferenceBinding is now pinned
                ))
            else:
                # User typed a raw string or reference to an uploaded Artifact
                bindings.append(LiteralBinding(value=item))

        # 7. Package inferred bindings for this input key
        inferred_bindings[key] = bindings

    return inferred_bindings


def extract_lineage(bindings: dict[str, list[Binding]], task_def: "TaskDefinition") -> list[str]:
    """Uses Bindings to build a list of parent Artifact IDs for the DAG."""
    parent_ids = []
    for field_name, bindings_list in bindings.items():
        _, extra = task_def.field_meta(field_name)
        if extra.get("is_artifact"):
            for b in bindings_list:
                # Concrete/Pinned edge
                if isinstance(b, ReferenceBinding) and b.artifact_id:
                    parent_ids.append(b.artifact_id)
                # Uploaded or external Artifact ID provided directly
                elif isinstance(b, LiteralBinding) and isinstance(b.value, str):
                    if b.value.strip():
                        parent_ids.append(b.value)
    return list(set(parent_ids))
