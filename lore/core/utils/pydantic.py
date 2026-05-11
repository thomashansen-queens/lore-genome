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


def get_base_type(annotation: Any) -> Any:
    """
    Recursively drills down through Optionals and Collections to find the core scalar type.
    Example: list[list[MyEnum] | None] -> MyEnum
    """
    # 1. Base Case: We hit the bottom (e.g., int, str, MyEnum)
    if get_origin(annotation) is None:
        return annotation

    # 2. Unwrap Union / Optional
    if is_optional_type(annotation):
        # Filter out NoneType
        non_none_args = [a for a in get_args(annotation) if a is not type(None)]
        if non_none_args:
             return get_base_type(non_none_args[0])

    # 3. Unwrap Collections (list, set, tuple, etc.)
    if is_collection_type(annotation):
        args = get_args(annotation)
        if args:
            return get_base_type(args[0])

    # 4. Fallback
    return annotation
