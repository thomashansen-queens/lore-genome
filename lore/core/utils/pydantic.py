"""
Utility functions for working with Pydantic type hints.
"""
from enum import Enum
import types
from typing import Any, Literal, Union, cast, get_args, get_origin


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
    origin = get_origin(annotation)

    # 1. Base Case: We hit the bottom (e.g., int, str, MyEnum)
    if origin is None:
        return annotation

    # 2. Literal: Literals are terminal types for our purposes
    if origin is Literal:
        return annotation

    # 3. Unwrap Union / Optional
    if is_optional_type(annotation):
        # Filter out NoneType
        non_none_args = [a for a in get_args(annotation) if a is not type(None)]
        if non_none_args:
             return get_base_type(non_none_args[0])

    # 4. Unwrap Collections (list, set, tuple, etc.)
    if is_collection_type(annotation):
        args = get_args(annotation)
        if args:
            return get_base_type(args[0])

    # 5. Fallback
    return annotation


def extract_choices(target_type: Any) -> list[dict[str, Any]] | None:
    """
    Extracts standardized label/value dictionaries from Enums and Literals.
    Returns None if the type is not a choice-based type.
    """
    # 1. Handle Enums
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        enum_class = cast(type[Enum], target_type)
        return [
            {"label": e.name.replace("_", " ").capitalize(), "value": e.value}
            for e in enum_class
        ]

    # 2. Handle Literals
    if get_origin(target_type) is Literal:
        return [
            {"label": str(val).replace("_", " ").capitalize(), "value": val}
            for val in get_args(target_type)
        ]

    return None
