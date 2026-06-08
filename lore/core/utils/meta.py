"""
Meta utilities for the LoRē framework.
"""
from typing import Any, Iterator


def iter_dsl_attrs(cls: type) -> Iterator[tuple[str, Any]]:
    """
    Yields public (name, value) pairs of attributes across a class's method
    resolution order (MRO), allowing DSL classes to inherit fields from
    base classes.
    Traverses base classes first (reversed MRO) so parent fields come first.
    1. Parent classes will appear at the top of the list in the UI
    2. Child classes will override parent fields if names are shadowed
    """
    for base_class in reversed(cls.__mro__):
        if base_class is object:
            continue

        # vars() preserves order of definition in classes
        for name, value in vars(base_class).items():
            # skip private and dunder attribs
            if not name.startswith("_"):
                yield name, value


def has_field_type(cls: type, field: type) -> bool:
    """
    Checks if a class or its parents contain fields of a specific type.
    Useful for duck-typing checks to determine if a class is a LoRe DSL
    definition.
    """
    return any(isinstance(v, field) for _, v in iter_dsl_attrs(cls))
