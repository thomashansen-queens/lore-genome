"""
Core definitions for a Task registry system.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Type, get_args, get_origin
from pydantic import BaseModel, create_model, Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

# --- Task inputs ---

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

    # Primitives
    TEXT = "text"
    TEXTAREA = "textarea"
    NUMBER = "number"
    FLOAT = "float"
    INTEGER = "integer"
    CHECKBOX = "checkbox"  # Standard boolean toggle
    DATE = "date"
    DATETIME = "date"  # could be datetime for YYYY-MM-DD hh:mm:ss


class Cardinality(str, Enum):
    """
    Defines the expected number of Artifacts for a Task input. Useful for UI and
    type hints. Task handler should still defensively check, and the Executor
    enforces this contract when resolving inputs.
    """
    OPTIONAL = "optional"     # 0 or 1
    SINGLE = "single"         # Exactly 1 (Required)
    ONE_OR_MORE = "multiple"  # 1 or more (Required)
    ANY = "any"               # 0 or more (Optional list)
    PAIR = "pair"             # Exactly 2
    TWO_OR_MORE = "two_or_more" # 2 or more (Required list)

    @property
    def is_required(self) -> bool:
        """One or more Artifacts must be provided."""
        return self in {self.SINGLE, self.ONE_OR_MORE, self.PAIR, self.TWO_OR_MORE}

    @property
    def allows_multiple(self) -> bool:
        """Could be multiple Artifacts, so should be provided as a list."""
        return self in {self.ONE_OR_MORE, self.ANY, self.PAIR, self.TWO_OR_MORE}

    def ui_widget(self) -> Widget:
        """Returns the appropriate UI widget type based on cardinality."""
        return Widget.ARTIFACT_MULTI if self.allows_multiple else Widget.ARTIFACT_SINGLE


class Materialization(str, Enum):
    """How the Task input should be materialized for the handler function"""
    ID = "id"            # Provide the Artifact ID(s) to the handler
    PATH = "path"        # Provide the file path(s) to the handler
    CONTENT = "content"  # Load the file(s) to memory, provide the content (e.g. JSON, text)
    STREAM = "stream"    # Provide a file-like stream to the handler (for large files)
    PREVIEW = "preview"  # Provide a small preview of the content (e.g. first 100 lines)


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
    ):
        self.description = description
        self.default = default
        self.label = label
        self.examples = examples

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

    def _enrich_schema_extra(self, extra: dict[str, Any]):  # pylint: disable=unused-argument
        """Hook for subclasses to extend JSON Schema extra field (e.g. options, Cardinality)."""
        pass  # pylint: disable=unnecessary-pass

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
    - Cardinality: number of Artifacts
    - Materialization: how to pass the Artifact to the handler
    - Accepted data: Fuzzy search for LoRe data type (e.g. "protein_fasta"),
      file format (e.g. "fasta") or specific slice of data by key (e.g. "protein_sequences")
    """
    def __init__(
        self,
        description: str = "",
        # Metadata
        cardinality: Cardinality = Cardinality.SINGLE,
        load_as: Materialization = Materialization.CONTENT,
        # Fuzzy Matching: ["json", "ncbi", "genome_accessions"]
        accepted_data: list[str] | None = None,
        # Pydantic pass-throughs
        default: Any = PydanticUndefined,
        label: str | None = None,
        examples: list[Any] | None = None,
    ):
        super().__init__(description=description, default=default, label=label, examples=examples)
        self.cardinality = cardinality
        self.load_as = load_as
        self.accepted_data = accepted_data or ["*"]

    def _enrich_schema_extra(self, extra: dict[str, Any]):
        """Artifact-specific metadata for UI rendering and validation"""
        extra.update({
            "is_artifact": True,
            "widget": self.cardinality.ui_widget().value,
            "cardinality": self.cardinality.value,
            "load_as": self.load_as.value,
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
        annotated_type: Type,  # primitive e.g. int, bool, or Enum e.g. SamplingStrategy
        description: str = "",
        default: Any = PydanticUndefined,
        label: str | None = None,
        examples: list[Any] | None = None,
        options: list[Any] | None = None,  # Manual options override
        widget: Widget | str | None = None,
    ):
        super().__init__(description=description, default=default, label=label, examples=examples)
        self.annotated_type = annotated_type
        self.options = options
        self.widget_override = widget

    def get_type_annotation(self) -> Type:
        return self.annotated_type

    def _enrich_schema_extra(self, extra: dict[str, Any]):
        """If options provided, add to schema extra for UI dropdown rendering."""
        origin = get_origin(self.annotated_type)
        args = get_args(self.annotated_type)
        is_list = origin in (list, tuple, set)
        target_type = args[0] if is_list and args else self.annotated_type

        if isinstance(target_type, type) and issubclass(target_type, Enum):
            extra["options"] = [e.value for e in target_type]
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

        # 3. Handle manual overrides for fixed options and widget type
        if self.options:
            extra["options"] = self.options
            # If manual options are given for a string, default to be a dropdown
            if extra.get("widget") == Widget.TEXT.value:
                extra["widget"] = Widget.SELECT.value

        if self.widget_override:
            # Handle both Enum members and raw strings
            extra["widget"] = getattr(self.widget_override, "value", self.widget_override)

# --- Task output ---

class TaskOutput(BaseModel):
    """
    A named output slot for a Task
    """
    data_type: str = Field(..., description="LoRe data type (e.g. 'genome_report', 'phylo_tree')")
    label: str = Field(..., description="Human-readable label for this output (used in default file names)")
    description: str = Field(default = "", description="Optional details about this output")
    is_primary: bool = Field(default=False, description="Whether this is the high-priority output of the Task")
    is_artifact: bool = Field(default=True, description="Whether this output is an Artifact ID (as opposed to a primitive value)")

# --- Task definition ---

@dataclass(frozen=True)
class TaskDefinition:
    """
    Each Task is a contract to perform a single unit of work in LoRe, with 
    defined inputs, outputs, and a handler function that does the science.

    Attributes:
        key: Unique identifier for the Task (e.g. "ncbi.fetch_genome_reports").
        name: Human-readable name of the Task. (e.g. "Fetch Genome Reports from NCBI").
        handler: Python function that implements the Task logic. (ctx, config) -> result.
        input_model: Pydantic model defining the Inputs for the Task.
        output_model: Pydantic model defining the Outputs for the Task
        description: Human-readable description of the Task.
        category: Category or grouping for the Task (e.g. "NCBI", "Phylogeny").
        icon: Emoji or symbol representing the Task visually.
    """
    key: str
    handler: Callable
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    description: str = ""
    name: str = ""
    category: str = ""
    icon: str = "⚡"


class TaskRegistry:
    """
    Registry for managing Task definitions. Use the `register` method as a decorator
    to register Task handler functions.
    """
    def __init__(self):
        self._tasks: dict[str, TaskDefinition] = {}

    def register(
        self,
        key: str,
        inputs: Type[BaseModel] | Type[Any],
        outputs: Type[BaseModel] | Type[Any],
        name: str | None = None,
        category: str | None = None,
        icon: str | None = None,
    ):
        """
        Decorator to registers a TaskDefinition with a unique key.
        Internally compiles LoRe TaskInput DSL to Pydantic model for validation 
        and UI generation. Also allows for raw Pydantic models for power users.
        """
        def _compile_dsl_to_pydantic(task_key: str, dsl_model: Type[Any]) -> Type[BaseModel]:
            """
            Compiles a LoRe TaskInput DSL class to a Pydantic BaseModel.
            Inspects the fields of the provided dataclass, converts TaskInput 
            fields to Pydantic Fields with appropriate metadata, and constructs
            a new Pydantic model class dynamically
            """
            fields = {}

            # Iterate to convert TaskInput fields to Pydantic field definitions
            # Using dir() to walk the Method Resolution Order, allowing inheritance
            for attr_name in dir(dsl_model):
                if attr_name.startswith("__"):
                    continue  # Skip dunder methods and attributes
                attr_value = getattr(dsl_model, attr_name)
                if isinstance(attr_value, TaskInput):
                    py_type = attr_value.get_type_annotation()
                    field_info = attr_value.to_field_info()
                    fields[attr_name] = (py_type, field_info)

            # Dynamic model creation with a unique name base on the task key
            safe_name = f"{task_key.replace(".", "_")}_InputModel"
            model = create_model(safe_name, **fields)
            model.__doc__ = dsl_model.__doc__  # Preserve docstring for the model

            return model

        def _compile_outputs_to_pydantic(task_key: str, dsl_outputs: Type[Any]) -> Type[BaseModel]:
            """
            Turns a list of TaskOutput definitions into a Pydantic model for 
            documentation and validation.
            """
            fields = {}
            for attr_name, attr_value in dsl_outputs.__dict__.items():
                if isinstance(attr_value, TaskOutput):
                    fields[attr_name] = (str, Field(
                        description=attr_value.description,
                        json_schema_extra={
                            "data_type": attr_value.data_type,
                            "description": attr_value.description,
                            "is_primary": attr_value.is_primary,
                            "is_output": True,  # Hint for UI
                        }
                    ))

            model = create_model(f"{task_key.replace(".", "_")}_OutputModel", **fields)
            model.__doc__ = dsl_outputs.__doc__
            return model

        def wrapper(func):
            if key in self._tasks:
                raise ValueError(f"Task with key '{key}' is already registered.")

            # Check if LoRe TaskInput fields are in input model (including inherited)
            is_dsl = any(
                isinstance(getattr(inputs, attr), TaskInput)
                for attr in dir(inputs)
                if not attr.startswith("__")
            )

            if is_dsl:
                final_input_model = _compile_dsl_to_pydantic(key, inputs)
            else:
                # Power user mode: Allow raw pydantic models, which allows for
                # custom validation and non-TaskInput fields, but requires more 
                # boilerplate and won't have the nice UI features
                if not issubclass(inputs, BaseModel):
                    raise ValueError(f"Inputs for {key} must be LoRe TaskInput or Pydantic BaseModel")
                final_input_model = inputs

            # Similar logic for outputs
            is_output_dsl = any(isinstance(v, TaskOutput) for v in outputs.__dict__.values())
            if is_output_dsl:
                final_output_model = _compile_outputs_to_pydantic(key, outputs)
            else:
                if not issubclass(outputs, BaseModel):
                    raise ValueError(f"Outputs for {key} must be LoRe TaskOutput or Pydantic BaseModel")
                final_output_model = outputs

            # Auto-generate metadata if not provided
            final_name = name or key.split(".")[-1].replace("_", " ").capitalize()
            final_category = category or (key.split(".")[0] if "." in key else "General")
            final_icon = icon or "⚡"

            self._tasks[key] = TaskDefinition(
                key=key,
                handler=func,
                input_model=final_input_model,
                output_model=final_output_model,
                description=func.__doc__ or "",
                name=final_name,
                category=final_category,
                icon=final_icon,
            )
            return func
        return wrapper

    def __getitem__(self, key: str) -> TaskDefinition:
        """Allows dict-like access to Task definitions"""
        if key not in self._tasks:
            raise KeyError(f"Task with key '{key}' not found.")
        return self._tasks[key]

    def get(self, key: str) -> TaskDefinition | None:
        """Retrieve a Task definition by its key."""
        return self._tasks.get(key)

    @property
    def all(self) -> dict[str, TaskDefinition]:
        """Get all registered Task definitions."""
        return self._tasks


# Global Task registry instance
task_registry = TaskRegistry()

# --- Task execution ---

class Task(BaseModel):
    """
    A single unit of work executed by LoRe Genome.
    """
    id: str
    registry_key: str
    name: str | None = None
    status: str

    # Execution data
    exec_config: dict[str, Any] = Field(default_factory=dict)  # Execution config (e.g. retry_count, timeout)
    inputs: dict[str, Any] = Field(default_factory=dict)  # Args in
    outputs: dict[str, Any] = Field(default_factory=dict)  # Artifact IDs out

    # Lineage
    parent_artifact_ids: list[str] = Field(default_factory=list)
    error: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration(self) -> float | None:
        """Calculate duration from started_at and completed_at timestamps."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def modified_at(self) -> datetime:
        """Returns the most recent timestamp"""
        return self.completed_at or self.started_at or self.created_at

    def validate_and_serialize(self) -> dict[str, Any]:
        """
        Validate and coerce Task input data via TaskDefinition's InputModel with
        a round-trip through the Pydantic model
        Ignores extra keys that aren't in the model (like web-specific form fields).
        """
        try:
            task_def = task_registry[self.registry_key]
            validated = task_def.input_model.model_validate(self.inputs, extra="ignore")
            return validated.model_dump(mode='json')
        except Exception as e:
            raise ValueError(f"Task Input Validation Error: {e}") from e
