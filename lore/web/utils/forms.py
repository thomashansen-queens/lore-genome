"""
Helpers for dealing with HTML forms
"""
import enum
import logging
import re
from typing import Any, Type
from typing_extensions import get_args
from pydantic_core import PydanticUndefined
from pydantic import BaseModel

from fastapi.datastructures import FormData

from lore.core.utils import clean, is_collection_type, is_optional_type
from lore.core.tasks import TaskDefinition

logger = logging.getLogger(__name__)

# --- Python -> Form ---

def format_inputs_for_ui(task_def: TaskDefinition, raw_inputs: dict) -> dict:
    """Helper to transform raw Python types into UI-friendly prefill strings."""
    ui_ready = {}
    model_fields = task_def.input_model.model_fields
    for key, field_info in model_fields.items():
        extra = getattr(field_info, "json_schema_extra", {}) or {}
        widget = extra.get("widget", "text")

        # 1. Priority: User input > Schema default > None
        if key in raw_inputs and raw_inputs[key] is not None:
            val = raw_inputs[key]
        elif field_info.default is not PydanticUndefined:
            val = field_info.default
        else:
            val = None

        # 2. Consider UI widget
        if widget == "artifact_multi_select":
            # Multi-selects must have lists of IDs
            if val is None:
                ui_ready[key] = []
            elif isinstance(val, list):
                ui_ready[key] = [str(v) for v in val]
            else:
                ui_ready[key] = [str(val)]

        elif widget == "artifact_single_select":
            ui_ready[key] = str(val) if val is not None else ""

        elif widget == "radio_group":
            if val is None:
                ui_ready[key] = ""
            elif isinstance(val, enum.Enum):
                ui_ready[key] = val.value
            else:
                ui_ready[key] = str(val)

        elif widget == "checkbox_group":
            # Return a list so the template can use `in` operator to check the boxes
            if val is None:
                ui_ready[key] = []
            elif isinstance(val, list):
                ui_ready[key] = [v.value if isinstance(v, enum.Enum) else str(v) for v in val if v is not None]
            else:
                ui_ready[key] = [val.value if isinstance(val, enum.Enum) else str(val)]

        elif widget == "checkbox":
            # Normalize to Python bool — handles string 'true'/'false' from deserialized manifests
            if isinstance(val, bool):
                ui_ready[key] = val
            elif isinstance(val, str):
                ui_ready[key] = val.lower() in ("true", "1", "on", "yes")
            else:
                ui_ready[key] = bool(val) if val is not None else False

        else:
            # Primitive types: Standard inputs
            if val is None:
                ui_ready[key] = ""
            elif isinstance(val, list):
                # Comma-separate lists for text inputs
                ui_ready[key] = ", ".join(str(v) for v in val if v is not None)
            elif isinstance(val, bool):
                ui_ready[key] = val
            elif isinstance(val, enum.Enum):
                ui_ready[key] = val.value
            else:
                ui_ready[key] = str(val)

    return ui_ready


def format_binding_for_ui(val: Any) -> str:
    """Converts rich Python types into HTML-safe, user-friendly strings."""
    if val is None:
        return ""
    if isinstance(val, enum.Enum):
        return str(val.value)
    if isinstance(val, list):
        # Recursively sanitize items, then join them safely
        return ", ".join(format_binding_for_ui(v) for v in val)
    if isinstance(val, bool):
        return "true" if val else "false"
    
    return str(val)

# --- Form -> Python ---

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


def _extract_raw_values(
        source: FormData | dict, key: str, is_list: bool
    ) -> tuple[list[Any], list[Any]]:
    """
    Extracts 'main' and '__manual' values into lists
    """
    manual_key = f"{key}__manual"

    if isinstance(source, dict):
        main_val = source.get(key)
        manual_val = source.get(manual_key)

        if isinstance(main_val, list):
            main_list = main_val
        else:
            main_list = [main_val] if main_val is not None else []

        if isinstance(manual_val, list):
            manual_list = manual_val
        else:
            manual_list = [manual_val] if manual_val is not None else []

        return main_list, manual_list

    else:
        get_list = getattr(source, 'getlist', None)
        if is_list and get_list:
            return get_list(key), get_list(f"{key}__manual")
        else:
            main_val = source.get(key)
            manual_val = source.get(f"{key}__manual")
            return ([main_val] if main_val is not None else []), ([manual_val] if manual_val is not None else [])


def form_html_to_dict(
    form_data: FormData | dict,
    model_class: Type[BaseModel],
    blank_to_default: bool = False,
    **clean_kwargs,
) -> dict[str, Any]:
    """
    Extracts FormData into a dictionary based on Pydantic model field types.
    Handles HTML form weirdness for list, bool, and optional fields.
    HTML->Python translation layer
    """
    result = {}

    for key, field in model_class.model_fields.items():
        # 1. Handle aliases
        form_key = field.alias or key
        annotation = field.annotation

        # 2. Determine type (List? Bool?)
        is_optional = is_optional_type(annotation)
        is_list = is_collection_type(annotation)

        # Unwrap Optional to get the inner type for int/float casting below
        if is_optional:
            non_none_args = tuple(a for a in get_args(annotation) if a is not type(None))
            if len(non_none_args) == 1:
                annotation = non_none_args[0]

        # 3. Extract raw value(s) from FormData and Normalize
        main_vals, manual_vals = _extract_raw_values(form_data, form_key, is_list)

        field_present = form_key in form_data or f"{form_key}__manual" in form_data
        if not field_present:
            continue

        if is_list:
            raw_items = [v for v in (main_vals + manual_vals) if v not in (None, "")]
            if not raw_items:
                if blank_to_default:
                    result[key] = field.get_default(call_default_factory=True)
                continue

            values = []
            for val in raw_items:
                pieces = re.split(r"[,\n;\r]+", val) if isinstance(val, str) else [val]
                for piece in pieces:
                    if isinstance(piece, str) and not piece.strip():
                        continue

                    cleaned_val = clean(piece, **clean_kwargs)
                    if cleaned_val not in (None, ""):
                        values.append(cleaned_val)

            result[key] = values
            continue

        elif annotation is bool:
            # HTML forms send "on" (or "true" or "1") for checked, key not present for unchecked
            result[key] = any(
                isinstance(v, str) and v.lower() in {"on", "true", "1", "yes"}
                for v in (main_vals + manual_vals)
            )
            continue

        # 4. Scalar fields: prefer __manual override when present
        raw_value = (
            manual_vals[0]
            if manual_vals and isinstance(manual_vals[0], str) and manual_vals[0].strip()
            else main_vals[0] if main_vals else None
        )

        if raw_value is None:
            if blank_to_default:
                result[key] = field.get_default(call_default_factory=True)
            continue

        # 5. Clean and type-cast value (if present) or apply defaults
        try:
            # Literal "None" string should be treated as Python None
            if isinstance(raw_value, str) and raw_value.strip() == "None":
                raw_value = ""
            cleaned = clean(raw_value, **clean_kwargs)

            if cleaned is None and cleaned != "":
                if blank_to_default:
                    result[key] = field.get_default(call_default_factory=True)
                continue

            if annotation is int:
                result[key] = int(cleaned)
            elif annotation is float:
                result[key] = float(cleaned)
            else:
                result[key] = cleaned

        except ValueError as e:
            raise ValueError(f"Invalid value for field '{form_key}': {e}") from e

    return result


def form_json_to_dict(payload: dict | Any, model: type[BaseModel]) -> dict[str, Any]:
    """
    Transforms raw frontend AJAX payload into a clean dictionary for Pydantic validation.
    Handles the __manual companion key emitted by artifact selectors and text overrides.
    """
    result = {}
    raw_dict = dict(payload) if not isinstance(payload, dict) else payload

    for field_name, field_info in model.model_fields.items():
        val = raw_dict.get(field_name)
        manual_val = raw_dict.get(f"{field_name}__manual")

        annotation = field_info.annotation
        is_list_expected = is_collection_type(annotation)

        # Unwrap Optional to check the inner type for bool
        inner = next((a for a in get_args(annotation) if a is not type(None)), annotation)
        is_bool_expected = inner is bool

        # Bool: handle hidden-input trick (always sends "false"; checked also sends "true")
        if is_bool_expected:
            if val is None:
                continue
            if isinstance(val, list):
                val = val[-1]  # last value wins: hidden="false" < checkbox="true"
            if isinstance(val, str):
                val = val.lower() in ("true", "on", "1", "yes")
            result[field_name] = val
            continue

        # List: merge direct values (checkboxes/multi-select) with __manual (textarea).
        # NOTE: val may be None when no checkboxes are checked but __manual has content
        # (e.g. ArtifactInput with no session candidates renders only the __manual textarea).
        if is_list_expected:
            combined = []
            if isinstance(val, list):
                combined.extend(str(v).strip() for v in val if v is not None and str(v).strip())
            elif isinstance(val, str) and val.strip():
                combined.extend(x.strip() for x in re.split(r"[,\n;\r]+", val) if x.strip())
            if isinstance(manual_val, str) and manual_val.strip():
                combined.extend(x.strip() for x in re.split(r"[,\n;\r]+", manual_val) if x.strip())
            if combined:
                result[field_name] = combined
            continue

        # Scalar: prefer __manual when non-empty (lets text override an artifact select)
        actual_val = manual_val if (isinstance(manual_val, str) and manual_val.strip()) else val
        if actual_val is None or actual_val == "":
            continue
        result[field_name] = actual_val

    # logger.debug("form_json_to_dict: result keys=%s", list(result.keys()))
    return result
