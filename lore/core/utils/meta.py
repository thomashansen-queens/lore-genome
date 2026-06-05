"""
Meta utilities for the LoRe framework.
"""
from typing import Any, Iterator

def iter_dsl_attrs(cls: type) -> Iterator[tuple[str, Any]]:
    """
    Yields public (name, value) pairs of attributes across a class's method
    resoultion order (MRO), allowing DSL classes to inherit fields from
    base classes.
    Traverses base classes first (reversed MRO) so parent fields appear at
    the top of the list and can be overridden by child classes if needed.
    """
    for base_class in reversed(cls.__mro__):
        if base_class is object:
            continue

        # vars() preserves order of definition in classes
        for name, value in vars(base_class).items():
            if not name.startswith("_"):
                yield name, value


def has_dsl_fields(cls: type, field_type: type) -> bool:
    """
    Checks if a class or its parents contain fields of a specific type.
    Useful for duck-typing checks to determine if a class is a LoRe DSL
    definition.
    """
    return any(isinstance(v, field_type) for _, v in iter_dsl_attrs(cls))
