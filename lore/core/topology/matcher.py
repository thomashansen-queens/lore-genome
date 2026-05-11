"""
Theoretical matching logic to connect Task outputs and Inputs into a
directed acyclic graph (DAG). Matching uses semantic types. Edges are
built from Task-Artifact relationships.
"""
from typing import Any, Literal

from lore.core.adapters import BaseAdapter
from lore.core.bindings import Binding, LiteralBinding, ReferenceBinding, UserInputBinding
from lore.core.sessions.session import Session


def is_output_compatible(
    source_extra: dict,
    target_extra: dict,
    adapters: list["BaseAdapter"] = [],
) -> bool:
    """
    Evaluates whether a Task's output can satisfy the data requirements of an input field.
    """
    default_req = target_extra.get("data_type", "unknown")
    accepted_data = set(target_extra.get("accepted_data", [default_req]))

    # 1. True wildcard (input accepts anything)
    if "*" in accepted_data:
        return True

    # 2. Direct semantic check (Does the base Artifact type match?)
    produced_type = source_extra.get("data_type", "unknown")
    if produced_type in accepted_data:
        # TODO: Won't this be true for unknown accepted_data and unknown produced_type?
        return True

    # 3. Adapter conversion check
    adapters = adapters or []
    for requirement in accepted_data:
        for adapter in adapters:
            if adapter.provides(requirement):
                return True

    return False


def infer_bindings_from_raw(
    raw_inputs: dict[str, Any],
    session: Session,
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
