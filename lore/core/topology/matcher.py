"""
Theoretical matching logic to determine if a Task's output can be
piped into another Task's input.
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
