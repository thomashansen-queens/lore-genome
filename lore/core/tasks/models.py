"""
Defines Task-related data structures and models, such as TaskDefinition, which 
encapsulates the contract for a unit of work in LoRe.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Type
from pydantic import BaseModel, Field

from lore.core.tasks.dsl import Cardinality

# --- Task Definition ---

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
    live_preview: bool = False

    def field_meta(self, key: str) -> tuple[Any, dict[str, Any]]:
        """
        Validates and extracts metadata for a given input field in a TaskDefinition.
        :returns: tuple[pydantic FieldInfo, dict]
        """
        model_fields = self.input_model.model_fields
        field_info = model_fields.get(key)
        if field_info is None:
            raise ValueError(f"Input key '{key}' not defined in '{self.key}' input model.")

        extra = getattr(field_info, "json_schema_extra", None)
        if extra is None or not isinstance(extra, dict):
            return field_info, {}
        return field_info, extra

    @property
    def primary_output_key(self) -> str | None:
        """Scans the output model for the field marked is_primary=True"""
        for key in self.output_model.model_fields.keys():
            meta = self.output_model.model_fields[key].json_schema_extra
            if meta is None:
                continue
            if meta.get("is_primary"):
                return key

        return None

# --- Task execution ---

class AdapterConfig(BaseModel):
    """Task config for the Adapter and UI presentation layer"""
    view_state: dict[str, Any] = Field(default_factory=dict, description="UI view state (e.g. {'sort_by': 'score'})")
    strategy: str = Field(default="auto", description="How the Adapter should load the data for the Task. 'auto' streams if possible, 'lazy' forces streaming, 'eager' loads everything")
    # FUTURE: UI settings can go here too (e.g. whether to show a table with pagination, or a simple list of results)


class ExecutionConfig(BaseModel):
    """Task config for the Engine's execution behaviour"""
    pass
    # FUTURE: Settings like "force_recommpute", "timeout" etc.


class ExecConfig(BaseModel):
    """Namespaced configuration for a Task execution"""
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    task: dict[str, Any] = Field(default_factory=dict)  # For arbitrary Task-specific config (e.g. instructions for the handler)


class Task(BaseModel):
    """
    A single unit of work executed by LoRe Genome.
    """
    id: str
    registry_key: str
    name: str | None = None
    status: str

    # Execution data
    exec_config: dict[str, Any] = Field(default_factory=dict)  # Namespaced config for execution
    inputs: dict[str, Any] = Field(default_factory=dict)  # Args in
    outputs: dict[str, list[str]] = Field(default_factory=dict)  # Artifact IDs out

    # Lineage
    parent_artifact_ids: list[str] = Field(default_factory=list)
    error: str | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

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

    def validate_config(self, raw_config: dict) -> dict[str, Any]:
        """
        Validate and coerce Task execution config via the ExecConfig model.
        Returns a clean config dictionary.
        """
        try:
            validated = ExecConfig.model_validate(raw_config, extra="ignore")
            return validated.model_dump(mode="json")
        except Exception as e:
            raise ValueError(f"Config Validation Error: {e}") from e

    def validate_and_serialize(self, raw_inputs: dict) -> dict[str, Any]:
        """
        Validate and coerce Task input data via TaskDefinition's InputModel
        Returns a clean Task-specific dictionary without mutating Task state
        """
        from lore.core.tasks import task_registry  # avoid circular import
        task_def = task_registry[self.registry_key]
        try:
            validated = task_def.input_model.model_validate(raw_inputs, extra="ignore")
            return validated.model_dump(mode="json")
        except Exception as e:
            raise ValueError(f"Task Input Validation Error: {e}") from e

    def update(
        self,
        inputs: dict | None = None,
        exec_config: dict | None = None,
        action: str | None = None,
        name: str | None = None,
        update_inputs: bool = True,
    ) -> bool:
        """
        Mutate the Task in place. Handles DRAFT/PENDING status. Returns True if
        the Task should proceed to execution
        """
        if name:
            self.name = name

        # 1. Update the raw stored dictionaries
        if inputs is not None:
            self.inputs = inputs
        if exec_config is not None:
            self.exec_config = exec_config

        if not update_inputs:
            return action == "RUN" and self.status == "PENDING"

        # 2. Gatekeep: Only allow valid Tasks to become PENDING
        try:
            clean_inputs = self.validate_and_serialize(self.inputs)
            clean_config = self.validate_config(self.exec_config)

            self.inputs = clean_inputs
            self.exec_config = clean_config
            self.status = "PENDING"

        except ValueError as e:
            self.status = "DRAFT"
            self.error = str(e)

        return action == "RUN" and self.status == "PENDING"


class TaskResults:
    """
    A strict container for Task outputs. All slots are lists to simplify engine 
    logic, but cardinality rules are enforced at the time of adding results.
    """
    # _allowed_keys: tuple[str, ...]
    _primary_key: str | None
    _key_cardinality: dict[str, Cardinality | None]

    def __init__(self, task_def: "TaskDefinition"):
        object.__setattr__(self, "_primary_key", task_def.primary_output_key)
        object.__setattr__(self, "_key_cardinality", {})

        if task_def.output_model:
            for key, field in task_def.output_model.model_fields.items():
                task_output = field.json_schema_extra
                if isinstance(task_output, dict):
                    cardinality = task_output.get("cardinality", Cardinality.SINGLE)

                if not isinstance(cardinality, Cardinality):
                    raise ValueError(f"Invalid cardinality for output '{key}' in Task '{task_def.key}'. Must be a Cardinality enum value.")

                self._key_cardinality[key] = cardinality

                # Initialize all output slots as empty lists (allows ordering; simplifies engine logic)
                object.__setattr__(self, key, [])

    def __setattr__(self, name: str, value: Any):
        if name in ("_key_cardinality", "_primary_key"):
            raise AttributeError(f"{name} is immutable after initialization")

        try:
            allowed = self._key_cardinality
        except AttributeError:
            object.__setattr__(self, name, value)
            return

        if name not in allowed:
            raise KeyError(
                f"Cannot set unknown output key: '{name}'\n"
                f"Allowed keys are: {', '.join(allowed)}"
            )

        self.add(name, value)

    def __getitem__(self, key: str) -> Any:
        """Allows keyed access to Task results."""
        if key not in self._key_cardinality:
            raise KeyError(f"Result with key '{key}' not found.")
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Supports the 'in' operator."""
        if key not in self._key_cardinality:
            return False
        return len(getattr(self, key)) > 0

    def __repr__(self) -> str:
        """Human-readable representation of TaskResults showing keys and their data types."""
        contents = ", ".join(
            f"{k}={getattr(self, k)!r}" for k in self._key_cardinality
        )
        return f"TaskResults({contents})"

    @property
    def primary_key(self) -> str | None:
        """Returns the primary output key or first if undefined"""
        if self._primary_key and self._primary_key in self._key_cardinality:
            return self._primary_key
        return list(self._key_cardinality.keys())[0] if self._key_cardinality else None

    @property
    def primary_data(self) -> Any:
        """Returns the data payload of the primary key."""
        key = self.primary_key
        return getattr(self, key) if key else None

    def next_empty_key(self) -> str | None:
        """Returns the next empty output slot key, or None if all are filled"""
        for k in self._key_cardinality:
            val = getattr(self, k)
            if val is None or (isinstance(val, list) and len(val) == 0):
                return k
        return None

    def add(self, name: str, value: Any):
        """
        Primary way for the engine to add results. Handles cardinality rules.
        Appends to lists if allows_multiple, otherwise does not overwrite.
        """
        if name not in self._key_cardinality:
            raise KeyError(
                f"Cannot set unknown output key: '{name}'. "
                f"Allowed keys are {', '.join(self._key_cardinality.keys())}"
            )

        cardinality = self._key_cardinality[name]
        current_list = getattr(self, name)
        if cardinality == Cardinality.SINGLE and len(current_list) >= 1:
            raise ValueError(
                f"Output key '{name}' allows only a single value, but multiple "
                "were added."
            )

        current_list.append(value)

    def load(self, name: str, value: Any):
        """Bypass cardinality rules for deserialization. Use with care!"""
        if name not in self._key_cardinality:
            raise KeyError(
                f"Cannot load unknown output key: '{name}'. "
                f"Allowed keys are {', '.join(self._key_cardinality.keys())}"
            )
        object.__setattr__(self, name, value)

    def to_dict(self) -> dict:
        """Serialize results to a dict for storage in Manifest"""
        serialized = {}
        for k in self._key_cardinality:
            items = getattr(self, k)
            serialized[k] = [item.id if hasattr(item, "id") else item for item in items]
        return serialized
