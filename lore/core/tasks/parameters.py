"""
LoRē domain-specific language (DSL) for defining Task inputs and outputs.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import types
from typing import Any, get_args, get_origin, Type, Union
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from lore.core.topology import traits
from lore.core.utils import is_collection_type, is_optional_type
from lore.core.utils.pydantic import get_base_type

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
    SELECT = "select"
    CHECKBOX_GROUP = "checkbox_group"
    RADIO = "radio"
    SEGMENTED_RADIO = "segmented_radio"  # Like radio but as a visual control

    # Primitives
    TEXT = "text"
    TEXTAREA = "textarea"
    SLIDER = "slider"
    FLOAT = "float"
    INTEGER = "integer"
    CHECKBOX = "checkbox"  # Boolean toggle, different from CHECKBOX_GROUP
    DATE = "date"
    DATETIME = "datetime"

_WIDGET_VALUES = {w.value for w in Widget}


class Cardinality(str, Enum):
    """
    Defines the expected number of Artifacts for a Task input. Useful for UI and
    type hints. Task handler should still defensively check, and the Execution
    enforces this contract when resolving inputs.
    """

    OPTIONAL_SINGLE = "optional"
    OPTIONAL_MULTIPLE = "optional_multiple"
    SINGLE = "single"
    MULTIPLE = "multiple"

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
    How the Task input should be materialized for the handler function.

    Pointer: Handler receives a reference and chooses what to do with it.
    Full RAM: Pass source data (RAW) or transformed source data (ADAPTED)
    Streamed: Same as Full RAM but passed as an iterator/generator
    Preview: Gives a small piece of content (e.g. first 100 lines) to the handler
    """

    # Pointer
    ARTIFACT = "artifact"
    PATH = "path"
    # Full RAM
    RAW = "raw"
    ADAPTED = "adapted"
    # Streamed
    RAW_STREAM = "raw_stream"
    ADAPTED_STREAM = "adapted_stream"
    # Preview
    PREVIEW = "preview"


# --- Task inputs ---


class TaskInput:
    """
    The Base DSL Class.
    This is not a Pydantic model itself, but is a factory for generating Pydantic FieldInfo with 
    with LoRe metadata at task registration time.
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
        Compiles LoRē DSL args to Pydantic Field's internal args.
        Smuggled in through json_schema_extra
        """
        field_kwargs = {
            "default": self.default,
            "description": self.description,
            "title": self.label,  # Pydantic uses 'title' for labels in JSON Schema
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
            return ""
        if isinstance(self.default, datetime):
            return self.default.strftime("%Y-%m-%dT%H:%M")
        return self.default


class ArtifactInput(TaskInput):
    """
    A topological "Wanted Ad" defining the data requirements for a Task.

    ArtifactInputs are flexible contracts. They define the shapes of data that are
    acceptable for a Task. The semantic matcher engine in LoRē will discover which 
    upstream Artifacts and/or task ouputs can satisfy the contract (via the Adapter 
    layer).

    The UI uses metadata in Pydantic's `json_schema_extra` to render the appropriate
    widget and enforce constraints:

    Attributes:
        - select (Cardinality): The number of Artifacts allowed (e.g. SINGLE, 
        OPTIONAL_MULTIPLE)
        - load_as (Materialization): How the data is delivered to the execution 
        handler (e.g. ADAPTED to a DataFrame, RAW_STREAM as an iterator)
        - Accepted data: The contract. Fuzzy search for LoRē data type which can be a 
        broad trait (e.g. lore.TABULAR), a specific type (e.g. "genome_annotations") 
        or a slice of from a table (e.g. "genome_accessions"). If provided as a list, 
        any one match is sufficient.
    """

    def __init__(
        self,
        description: str = "",
        select: Cardinality = Cardinality.SINGLE,
        load_as: Materialization = Materialization.ADAPTED,
        # Fuzzy Matching: ["json", "ncbi", "genome_accessions"]
        accepted_data: str | traits.DataTrait | list[str | traits.DataTrait] | None = traits.ANY,
        # Pydantic pass-throughs
        default: Any = PydanticUndefined,
        label: str | None = None,
        examples: list[Any] | None = None,
    ):
        super().__init__(description=description, default=default, label=label, examples=examples)
        self.cardinality = select
        self.materialization = load_as
        if accepted_data is None:
            self.accepted_data = [traits.ANY]
        elif isinstance(accepted_data, list):
            self.accepted_data = accepted_data
        else:
            self.accepted_data = [accepted_data]

    def _enrich_schema_extra(self, extra: dict[str, Any]):
        """Artifact-specific metadata for UI rendering and validation"""
        extra.update(
            {
                "is_artifact": True,
                "widget": self.cardinality.ui_widget().value,
                "select": self.cardinality.value,
                "load_as": self.materialization.value,
                # TODO: Stringify traits for serialization/UI
                "accepted_data": self.accepted_data,
            }
        )

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
        step: float | int | None = None,  # pydantic alias
        **pydantic_kwargs,
    ):
        if min is not None:
            pydantic_kwargs["ge"] = min
        if max is not None:
            pydantic_kwargs["le"] = max
        if step is not None:
            pydantic_kwargs["multiple_of"] = step

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

    def get_type_annotation(self) -> Any:
        return self.annotated_type

    def _enrich_schema_extra(self, extra: dict[str, Any]):
        """If options provided, add to schema extra for UI dropdown rendering."""
        # 1. Unwrap Union types to get the core type (int | None -> int)
        is_optional = is_optional_type(self.annotated_type)

        # 2. Unwrap collections (e.g. list[str] -> str) to get item type for Enums
        is_list = is_collection_type(self.annotated_type)

        # 3. Get the base scalar type for widget inference/Enum handling
        target_type = get_base_type(self.annotated_type)

        # 4. Interpret Pydantic kwargs
        if "ge" in self.pydantic_kwargs:
            extra["min"] = self.pydantic_kwargs["ge"]
        if "le" in self.pydantic_kwargs:
            extra["max"] = self.pydantic_kwargs["le"]
        if "multiple_of" in self.pydantic_kwargs:
            extra["step"] = self.pydantic_kwargs["multiple_of"]

        # 5. Inject None to optional Enums
        enums = None
        if isinstance(target_type, type) and issubclass(target_type, Enum):
            enums = [
                {"label": e.name.replace("_", " ").capitalize(), "value": e.value}
                for e in target_type
            ]
            if is_optional and "None" not in [e["value"] for e in enums]:
                enums.insert(0, {"label": "None (default)", "value": ""})

        # 6. Assign widget types
        if isinstance(target_type, type) and issubclass(target_type, Enum):
            extra["options"] = enums
            extra["widget"] = Widget.CHECKBOX_GROUP if is_list else Widget.SELECT

        elif target_type is bool:
            extra["widget"] = Widget.CHECKBOX

        elif target_type is int:
            extra["widget"] = Widget.INTEGER

        elif target_type is float:
            extra["widget"] = Widget.FLOAT

        elif target_type is datetime:
            extra["widget"] = Widget.DATETIME

        elif target_type is str:
            # Base class sets Widget.TEXT.value by default, but explicit is better than implicit
            extra["widget"] = Widget.TEXT

        # 7. Handle manual overrides for fixed options and widget type
        if self.options:
            normalized_options = _normalize_options(self.options)

            extra["options"] = normalized_options
            # If manual options are given for a string, default to be a dropdown
            if extra.get("widget") in (Widget.TEXT, Widget.INTEGER, Widget.FLOAT):
                extra["widget"] = Widget.SELECT

        # 8. Manual widget override (e.g. for Enums to radio instead of dropdown)
        if self.widget_override and self.widget_override in _WIDGET_VALUES:
            extra["widget"] = self.widget_override


def _normalize_options(options: str | list[Any]) -> list[dict[str, Any]]:
    """
    Helper to convert simple options lists into UI-ready dicts.
    key: Label. value: Raw option value passed to handler
    """
    normalized_options = []

    if isinstance(options, str):
        options = [opt.strip() for opt in options.split(",")]

    for opt in options:
        if isinstance(opt, dict) and "value" in opt:
            opt_dict = opt.copy()
            if "label" not in opt_dict:
                opt_dict["label"] = str(opt_dict.get("value")).title()
            normalized_options.append(opt_dict)
        else:
            normalized_options.append(
                {
                    "label": str(opt).title(),
                    "value": opt,
                }
            )

    return normalized_options

# --- Task output ---


@dataclass(frozen=True)
class Passthrough:
    """
    A declarative marker for a TaskOutput to inherit semantic data type and 
    schema from one of its inputs. This is useful for Tasks that transform data 
    but don't fundamentally change its nature (e.g. format conversions, slicing, 
    filtering).
    
    Usage:
        filtered_data = TaskOutput(
            data_type=Passthrough("source"),
            label="Filtered data",
        )    
    """
    slot: str


class TaskOutput(BaseModel):
    """
    A named output slot for a Task

    Attributes:
        data_type: LoRē data type for this output (e.g. "genome_report", "phylo_tree").
        This can also be a Passthrough to inherit data type from an input slot (e.g. 
        Passthrough("fasta_input") to say "same data type as the 'fasta_input' slot")
        yields: Expected number of Artifacts for this output
        label: Human-readable label for this output (used in UI, default file names)
        description: Optional details about this output (used in UI)
        is_primary: At least one output must be marked as the 'primary' output.
        is_artifact: Whether this output is an Artifact (True, default) or primitive value (False).
    """

    data_type: str | Passthrough
    label: str
    yields: Cardinality = Cardinality.SINGLE
    description: str = ""
    is_primary: bool = False
    is_artifact: bool = True

    # Engine-facing `cardinality` alias
    @property
    def cardinality(self) -> Cardinality:
        return self.yields
