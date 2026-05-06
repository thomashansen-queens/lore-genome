"""
Theoretical matching logic to connect Task outputs and Inputs into a
directed acyclic graph (DAG). Matching uses semantic types. Edges are
built from Task-Artifact relationships.
"""

from lore.core.adapters import BaseAdapter


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


# def infer_bindings_from_raw(raw_inputs: dict[str, Any], session: Session) -> dict[str, list[Binding]]:
#     inferred_bindings = {}

#     for key, val in raw_inputs.items():
#         val_list = val if isinstance(val, list) else ([val] if val is not None and val != "" else [])
#         bindings = []
        
#         for item in val_list:
#             if isinstance(item, str) and not item.strip():
#                 continue
                
#             str_item = str(item).strip()
#             artifact = session.get_artifact(str_item)
            
#             # If the artifact exists and has a producer...
#             if artifact and getattr(artifact, "created_by_task_id", None):
#                 # 💡 Create a PINNED reference!
#                 bindings.append(ReferenceBinding(
#                     source_id=artifact.created_by_task_id,
#                     output_key=artifact.output_key or "default_output",
#                     artifact_id=artifact.id  # Lock it in!
#                 ))
#             else:
#                 # User typed a raw string or uploaded an external file
#                 bindings.append(LiteralBinding(value=item))

#         inferred_bindings[key] = bindings

#     return inferred_bindings
