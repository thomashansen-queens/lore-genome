"""
Utility functions for the web application.
"""

import types
from typing import Type, Any, Union, get_args, get_origin, TYPE_CHECKING
from pydantic import BaseModel

from fastapi.datastructures import FormData

from lore.core.utils import clean

if TYPE_CHECKING:
    from lore.core.tasks import TaskDefinition

# --- Form handling utilities ---

def get_form_str(form_data: FormData, key: str, **clean_kwargs) -> str | None:
    """
    Avoids Type UploadFile. Safely retrieve a string value from FormData.
    """
    val = form_data.get(key)
    return clean(val, **clean_kwargs)


def get_form_list(form_data: FormData, key: str, **clean_kwargs) -> list[str]:
    """
    Avoids Type UploadFile. Safely retrieve a list value from FormData.
    """
    # .getlist() returns List[str | UploadFile]
    raw_list = form_data.getlist(key)
    val = [clean(item, **clean_kwargs) for item in raw_list if isinstance(item, str)]
    val = [v for v in val if v is not None]  # Remove empty strings if clean() returns None
    if val:
        return val
    return []


# --- Form to Pydantic parsing ---

def form_to_dict(form_data: FormData, model_class: Type[BaseModel]) -> dict[str, Any]:
    """
    Extracts FormData into a dictionary based on Pydantic model field types.
    Handles HTML form weirdness for list, bool, and optional fields.
    HTML->Python translation layer
    """
    result = {}

    for name, field_info in model_class.model_fields.items():
        # 1. Handle aliases
        form_key = field_info.alias or name
        annotation = field_info.annotation

        # 2. Determine type (List? Bool?)
        origin = get_origin(annotation)
        # Handle Union types (Optional[...] or | None)
        if origin is Union or (hasattr(types, "UnionType") and isinstance(origin, types.UnionType)):
            args = get_args(annotation)
            non_none_type = next((a for a in args if a is not type(None)), None)
            if non_none_type:
                annotation = non_none_type
                origin = get_origin(annotation)
        # Check if it's a list-like type
        is_list = origin in (list, set, tuple) or (isinstance(annotation, type) and issubclass(annotation, list))

        # 3. Extract raw value(s) from FormData and Normalize
        if is_list:
            get_list = getattr(form_data, 'getlist', None)
            main_vals = get_list(form_key) if get_list else [form_data.get(form_key)]
            manual_vals = get_list(f"{form_key}__manual") if get_list else [form_data.get(f"{form_key}__manual")]

            if not any(main_vals) and not any(manual_vals):
                continue # No value provided for this field

            values = []
            for val in filter(None, main_vals + manual_vals):
                if isinstance(val, str) and ("\n" in val or "," in val):
                    # Normalize commas to newlines -> split -> strip
                    normalized = val.replace(",", "\n")
                    values.extend([
                        clean(v) for v in normalized.split("\n")
                        if clean(v, reject=None) is not None and clean(v) != ""
                    ])
                else:
                    v = clean(val)
                    if v is not None and v != "":
                        values.append(v)

            result[name] = values

        elif field_info.annotation == bool:
            # HTML forms send "on" (or "true" or "1") for checked, key not present for unchecked
            value = form_data.get(form_key)
            if value is not None:
                if isinstance(value, str) and value.lower() in ("false", "0", "off"):
                    result[name] = False
                else:
                    result[name] = True
            else:
                result[name] = False

        else:
            # Standard fields: Str, Int, Enum
            main_val = form_data.get(form_key)
            manual_val = form_data.get(f"{form_key}__manual")

            # Belt & suspenders: Resolver will do this, but give manual precedence
            raw_value = manual_val if (isinstance(manual_val, str) and manual_val.strip()) else main_val

            if raw_value is not None:
                try:
                    cleaned = clean(raw_value)
                    if cleaned is not None and cleaned != "":
                        if annotation is int:
                            result[name] = int(cleaned)
                        elif annotation is float:
                            result[name] = float(cleaned)
                        else:
                            result[name] = cleaned
                except ValueError as e:
                    raise ValueError(f"Invalid value for field '{form_key}': {e}") from e

    return result
