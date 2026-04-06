"""
Utility functions for working with Pydantic type hints.
"""
import types
from typing import Any, Union, get_args, get_origin


COLLECTION_TYPES = {list, set, tuple}

def is_collection_type(annotation: Any) -> bool:
    """
    Recursive check if a Pydantic type hint represents a collection (list, set, 
    dict, etc.) even if buried in Optional or Union.
    """
    origin = get_origin(annotation)

    # 1. Base case: Is a collection type
    if annotation in COLLECTION_TYPES or origin in COLLECTION_TYPES:
        return True

    # 2. Recursive case: Optional or Union
    is_union = origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType)
    if is_union:
        return any(is_collection_type(arg) for arg in get_args(annotation))

    return False


def is_optional_type(annotation: Any) -> bool:
    """Check if a type hint includes None (i.e. is Optional / Union with None)."""
    origin = get_origin(annotation)
    is_union = origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType)
    if is_union:
        return type(None) in get_args(annotation)
    return False
