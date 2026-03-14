"""
LoRe domain-specific language (DSL) for defining Task inputs and outputs in a 
clear, self-documenting way.
"""

from datetime import datetime
from enum import Enum
import types
from typing import Any, get_args, get_origin, Type, Union
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

# --- Data handling ---

class Widget(str, Enum):
    """
    Defines the type of UI widget to render for a TaskInput. Will be inferred
    unless manually overridden (e.g. for Enums)
    """
    # Artifacts
    ARTIFACT_SINGLE = "artifact_single_select"
    ARTIFACT_MULTI = "artifact_multi_select"

    # Enums / Choices
    SELECT = "select"  # Standard dropdown
    CHECKBOX_GROUP = "checkbox_group"  # Multi-select enums
    RADIO_GROUP = "radio_group"  # Single-select enums (alternative to select)
    SEGMENTED_RADIO = "segmented_radio"  # More visual radio option, good for small number of options

    # Primitives
    TEXT = "text"
    TEXTAREA = "textarea"
    SLIDER = "slider"
    NUMBER = "number"
    FLOAT = "float"
    INTEGER = "integer"
    CHECKBOX = "checkbox"  # Standard boolean toggle
    DATE = "date"
    DATETIME = "date"  # could be datetime for YYYY-MM-DD hh:mm:ss


class Cardinality(str, Enum):
    """
    Defines the expected number of Artifacts for a Task input. Useful for UI and
    type hints. Task handler should still defensively check, and the Execution
    enforces this contract when resolving inputs.
    """
    OPTIONAL_SINGLE = "optional"     # 0 or 1
    OPTIONAL_MULTIPLE = "optional_multiple"  # 0 or more (Optional list)
    SINGLE = "single"         # Exactly 1 (Required)
    MULTIPLE = "multiple"     # 1 or more (Required)

    @property
    def is_required(self) -> bool:
        """One or more Artifacts must be provided."""
        return self in {self.SINGLE, self.MULTIPLE}

    @property
    def allows_multiple(self) -> bool:
        """Could be multiple Artifacts, so should be provided as a list."""
        return self in {self.MULTIPLE, self.OPTIONAL_MULTIPLE}

    def ui_widget(self) -> Widget:
        """Returns the appropriate UI widget type based on cardinality."""
        return Widget.ARTIFACT_MULTI if self.allows_multiple else Widget.ARTIFACT_SINGLE


class Materialization(str, Enum):
    """
    How the Task input should be materialized for the handler function. ID and 
    PATH are provided directly to the handler as strings, while CONTENT, STREAM,
    and PREVIEW will pass through the Adapter layer to load the data.
    """
    # Pointer (handler can choose how to load Artifact)
    ARTIFACT = "artifact"      # Provide the Artifact record
    PATH = "path"              # Provide the file path(s)
    # Full RAM (Execution loads content to memory and passes to handler)
    RAW = "raw"          # Parse file(s) into records but don't adapt
    ADAPTED = "ADAPTED"        # Load the file(s) to memory and adapt
    # Chunked RAM (Execution passes a generator to handler)
    RAW_STREAM = "raw_stream"  # Iterator yields parsed records
    ADAPTED_STREAM = "adapated_stream"          # Iterator yields adapted records
    # Preview (Execution provides a small piece of content to handler)
    PREVIEW = "preview"        # Provide a small adapted preview of the content (e.g. first 100 lines)

# --- Task inputs ---

class TaskInput:
    """
    The Base DSL Class.
    Use this to define the inputs for a Task in a clear, self-documenting way.
    """
    def __init__(
        self,
        description: str = "",
        default: Any = PydanticUndefined,
        label: str | None = None,
        examples: list[Any] | None = None,
        **pydantic_kwargs,
    ):
        self.description = description
        self.default = default
        self.label = label
        self.examples = examples
        self.pydantic_kwargs = pydantic_kwargs

    def to_field_info(self) -> FieldInfo:
        """
        Compiles LoRe DSL args to Pydantic Field's internal args.
        Smuggled in through json_schema_extra
        """
        field_kwargs = {
            "default": self.default,
            "description": self.description,
            "title": self.label, # Pydantic uses 'title' for labels in JSON Schema
            "examples": self.examples,
            **self.pydantic_kwargs,
        }
        lore_dsl = {
            "is_task_input": True,  # Hint for UI
            "label": self.label or "",  # Label for UI
            "widget": Widget.TEXT.value,  # Default, subclasses override
        }
        self._enrich_schema_extra(lore_dsl)
        return Field(
            **field_kwargs,
            json_schema_extra=lore_dsl,
        )

    def _enrich_schema_extra(self, extra: dict[str, Any]):
        """Hook for subclasses to extend JSON Schema extra field (e.g. options, Cardinality)."""
        pass

    def get_type_annotation(self) -> Type:
        """Returns the Python type expected for this input (e.g. str, int, list[str])"""
        raise NotImplementedError

    @property
    def safe_default(self):
        """Used by UI elements to show a default value only if provided"""
        if self.default in (PydanticUndefined, None):
            return ''
        if isinstance(self.default, datetime):
            return self.default.strftime('%Y-%m-%d')
        return self.default


class ArtifactInput(TaskInput):
    """
    A single Artifact or list of Artifacts as input to a Task.
    The UI can use the metadata in json_schema_extra to render the appropriate
    widget and enforce constraints:
    - select: number of Artifacts ("cardinality" is exposed as "select" for clarity)
    - load_as: how to pass the Artifact to the handler ("materialization" is exposed as "load_as" for clarity)
    - Accepted data: Fuzzy search for LoRe data type (e.g. "protein_fasta"),
      file format (e.g. "fasta") or specific slice of data by key (e.g. "protein_sequences")
    """
    def __init__(
        self,
        description: str = "",
        # Metadata
        select: Cardinality = Cardinality.SINGLE,
        load_as: Materialization = Materialization.ADAPTED,
        # Fuzzy Matching: ["json", "ncbi", "genome_accessions"]
        accepted_data: str | list[str] | None = "*",
        # Pydantic pass-throughs
        default: Any = PydanticUndefined,
        label: str | None = None,
        examples: list[Any] | None = None,
    ):
        super().__init__(description=description, default=default, label=label, examples=examples)
        self.cardinality = select
        self.materialization = load_as
        self.accepted_data = accepted_data if isinstance(accepted_data, list) else [accepted_data]

    def _enrich_schema_extra(self, extra: dict[str, Any]):
        """Artifact-specific metadata for UI rendering and validation"""
        extra.update({
            "is_artifact": True,
            "widget": self.cardinality.ui_widget().value,
            "select": self.cardinality.value,
            "load_as": self.materialization.value,
            "accepted_data": self.accepted_data,
        })

    def get_type_annotation(self) -> Type:
        """
        Smart Typing: If cardinality allows multiple, we expect a List of IDs.
        Otherwise, a single ID string.
        """
        if self.cardinality.allows_multiple:
            return list[str]
        return str


class ValueInput(TaskInput):
    """
    A single primitive (e.g. str, list) or Enum value as input to a Task.
    """
    def __init__(
        self,
        annotated_type: Any,  # primitive or Enum. Accepts UnionTypes like "int | None"
        description: str = "",
        default: Any = PydanticUndefined,
        label: str | None = None,
        examples: list[Any] | None = None,
        options: list[Any] | None = None,  # Manual options override
        widget: Widget | str | None = None,
        min: float | int | None = None,  # pydantic alias
        max: float | int | None = None,  # pydantic alias
        step: float | int | None = None, # pydantic alias
        **pydantic_kwargs,
    ):
        if min is not None: pydantic_kwargs["ge"] = min
        if max is not None: pydantic_kwargs["le"] = max
        if step is not None: pydantic_kwargs["multiple_of"] = step

        super().__init__(
            description=description,
            default=default,
            label=label,
            examples=examples,
            **pydantic_kwargs,
        )
        self.annotated_type = annotated_type
        self.options = options
        self.widget_override = widget

    def get_type_annotation(self) -> Type:
        return self.annotated_type

    def _enrich_schema_extra(self, extra: dict[str, Any]):
        """If options provided, add to schema extra for UI dropdown rendering."""
        target_type = self.annotated_type
        is_optional = False

        origin = get_origin(self.annotated_type)

        # 1. Unwrap Union types to get the core type (int | None -> int)
        if origin is Union or origin is types.UnionType:
            args = get_args(target_type)
            is_optional = type(None) in args

            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                target_type = non_none_args[0]
                origin = get_origin(target_type)

        # 2. Unwrap collections (e.g. list[str] -> str) to get item type for Enums
        is_list = origin in (list, tuple, set)
        if is_list:
            args = get_args(target_type)
            target_type = args[0] if args else target_type

        # 3. Interpret Pydantic kwargs
        if "ge" in self.pydantic_kwargs:
            extra["min"] = self.pydantic_kwargs["ge"]
        if "le" in self.pydantic_kwargs:
            extra["max"] = self.pydantic_kwargs["le"]
        if "multiple_of" in self.pydantic_kwargs:
            extra["step"] = self.pydantic_kwargs["multiple_of"]

        # 4. Inject None to optional Enums
        enums = None
        if isinstance(target_type, type) and issubclass(target_type, Enum):
            enums = [{"label": e.name.replace("_", " ").capitalize(),
                      "value": e.value} for e in target_type]
            if is_optional and "None" not in [e["value"] for e in enums]:
                enums.insert(0, {"label": "None (default)", "value": ""})

        # 5. Assign widget types
        if isinstance(target_type, type) and issubclass(target_type, Enum):
            extra["options"] = enums
            extra["widget"] = Widget.CHECKBOX_GROUP.value if is_list else Widget.SELECT.value

        elif target_type is bool:
            extra["widget"] = Widget.CHECKBOX.value

        elif target_type is int:
            extra["widget"] = Widget.INTEGER.value

        elif target_type is float:
            extra["widget"] = Widget.FLOAT.value

        elif target_type is datetime:
            extra["widget"] = Widget.DATETIME.value

        elif target_type is str:
            # Base class sets Widget.TEXT.value by default, but explicit is better than implicit
            extra["widget"] = Widget.TEXT.value

        # 6. Handle manual overrides for fixed options and widget type
        if self.options:
            normalized_options = _normalize_options(self.options)

            extra["options"] = normalized_options
            # If manual options are given for a string, default to be a dropdown
            if extra.get("widget") in (Widget.TEXT.value, Widget.INTEGER.value, Widget.FLOAT.value):
                extra["widget"] = Widget.SELECT.value

        # 7. Manual widget override (e.g. for Enums to radio instead of dropdown)
        if self.widget_override:
            extra["widget"] = getattr(self.widget_override, "value", self.widget_override)


def _normalize_options(options: str | list[str]) -> list[dict[str, Any]]:
    """
    Helper to convert simple options lists into UI-ready dicts.
    key: Label. value: Raw option value passed to handler
    """
    normalized_options = []

    if isinstance(options, str):
        options = [opt.strip() for opt in options.split(",")]

    for opt in options:
        if isinstance(opt, dict) and "value" in opt:
            if "label" not in opt:
                opt["label"] = str(opt.get("value")).title()
            normalized_options.append(opt)
        else:
            normalized_options.append({
                "label": str(opt).title(),
                "value": opt,
            })

    return normalized_options

# --- Task output ---

class TaskOutput(BaseModel):
    """
    A named output slot for a Task
    """
    data_type: str = Field(..., description="LoRe data type (e.g. 'genome_report', 'phylo_tree')")
    yields: Cardinality = Field(default=Cardinality.SINGLE, description="Expected number of Artifacts for this output")
    label: str = Field(..., description="Human-readable label for this output (used in default file names)")
    description: str = Field(default = "", description="Optional details about this output")
    is_primary: bool = Field(default=False, description="Whether this is the high-priority output of the Task")
    is_artifact: bool = Field(default=True, description="Whether this output is an Artifact ID (as opposed to a primitive value)")
