"""
The actual Task instances that are created, mutated, and executed within
a Session. The TaskDefinition is the immutable contract, while the Task is
the mutable instance.
"""
from datetime import datetime, timezone
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any, cast, TYPE_CHECKING

from lore.core.bindings import (
    Binding,
    LiteralBinding,
    ReferenceBinding,
    UserInputBinding,
    MissingUserInputError,
    UnresolvedReferenceError,
)
from lore.core.utils import is_collection_type
from .parameters import Cardinality, Passthrough
from .state import TaskIntegrity, TaskStatus

if TYPE_CHECKING:
    from .definition import TaskDefinition

# --- Per-Task execution data ---

class AdapterStrategy(StrEnum):
    """Defines how the materializer and adapter should load the data for a task."""
    AUTO = "auto"
    LAZY = "lazy"  # Force streaming if possible (O(1) memory)
    EAGER = "eager"  # Force full loading into memory
    PEEK = "peek"  # UI mode: Fast, 100-row limit
    FULL = "full"  # UI mode: full execution preview (may be slow)


class AdapterConfig(BaseModel):
    """Task config for the Adapter and UI presentation layer"""
    view_state: dict[str, Any] = Field(
        default_factory=dict, description="UI view state (e.g. {'sort_by': 'score'})"
    )
    strategy: AdapterStrategy = Field(
        default=AdapterStrategy.AUTO,
        description="How the Adapter should load the data for the Task. 'auto' streams if possible, 'lazy' forces streaming, 'eager' loads everything",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="kwargs to pass to the Adapter e.g. {'separator': '\\t'}",
    )


class ExecutionConfig(BaseModel):
    """Task config for the Engine's execution behaviour"""
    pass
    # FUTURE: Settings like "force_recommpute", "timeout" etc.


class TaskConfig(BaseModel):
    """Namespaced configuration for a Task execution"""
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    task: dict[str, Any] = Field(default_factory=dict)  # arbitrary Task-specific config

# --- The Task itself ---

class Task(BaseModel):
    """
    A concrete unit of work within a Session. Inputs are validated on assignment.
    """
    # Identity
    model_config = ConfigDict(validate_assignment=True)
    id: str
    registry_key: str
    name: str | None = None
    description: str | None = None

    # State
    status: TaskStatus = Field(default=TaskStatus.DRAFT)
    integrity: TaskIntegrity = Field(default=TaskIntegrity.UNKNOWN)

    # Execution data
    exec_config: dict[str, Any] = Field(default_factory=dict)  # Namespaced config for execution
    inputs: dict[str, list[Binding]] = Field(default_factory=dict)
    outputs: dict[str, list[Any]] = Field(default_factory=dict)  # Artifact IDs or python objects

    # Lineage
    parent_artifact_ids: list[str] = Field(default_factory=list)
    error: str | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @field_validator("inputs", mode="before")
    @classmethod
    def inputs_to_bindings(cls, v: Any) -> Any:
        """Ensures any raw dictionaries passed to `inputs` are strictly wrapped in Bindings."""
        if isinstance(v, dict):
            from lore.core.bindings import wrap_in_bindings

            return wrap_in_bindings(v)
        return v

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
            validated = TaskConfig.model_validate(raw_config, extra="ignore")
            return validated.model_dump(mode="json")
        except Exception as e:
            raise ValueError(f"Config Validation Error: {e}") from e

    def validate_and_serialize(self) -> dict[str, Any]:
        """
        Validate and coerce Task input Bindings via TaskDefinition's InputModel
        Returns a clean Task-specific dictionary without mutating Task state
        """
        from lore.core.tasks import task_registry

        task_def = task_registry[self.registry_key]

        # 1. Temporarily unwrap the Bindings for validation
        unwrapped_inputs = {}
        for key, bindings in self.inputs.items():
            field_info, extra = task_def.field_meta(key)
            allows_multiple = extra.get("cardinality", Cardinality.SINGLE).allows_multiple
            expects_collection = allows_multiple or is_collection_type(field_info.annotation)

            unwrapped_list = []
            for b in bindings:
                if isinstance(b, UserInputBinding):
                    raise MissingUserInputError(f"Missing user input: {key}")
                elif isinstance(b, ReferenceBinding):
                    if b.artifact_id:
                        # Pinned: Concrete Artifact ID provided
                        unwrapped_list.append(b.artifact_id)
                    else:
                        # Unpinned: Must resolve at runtime
                        # This string is meaningless and exists only to satisfy Pydantic;
                        # the materializer will resolve the ReferenceBinding to concrete data
                        unwrapped_list.append(f"<promise:{b.source_id}.{b.output_key}>")
                elif isinstance(b, LiteralBinding):
                    unwrapped_list.append(b.value)
                else:
                    raise ValueError(f"Invalid binding type for input '{key}': {type(b)}")

            # 2. Shape the data to match the schema's expectations
            if expects_collection:
                # Type is e.g. list[str], or ArtifactInput with cardinality.allow_multiple
                unwrapped_inputs[key] = unwrapped_list
            else:
                # Type is e.g. str, or ArtifactInput with cardinality.single; take first only
                unwrapped_inputs[key] = unwrapped_list[0] if unwrapped_list else None

        # 3. Validate against the TaskDefinition's InputModel
        try:
            validated = task_def.input_model.model_validate(unwrapped_inputs, extra="ignore")
            return validated.model_dump(mode="json")
        except Exception as e:
            raise ValueError(f"Task Input Validation Error: {e}") from e

    def update(
        self,
        inputs: dict[str, list[Binding]] | None = None,
        exec_config: dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """
        Mutate the Task in place. Handles DRAFT/READY status.
        """
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description

        # 1. Update the raw stored dictionaries
        if inputs is not None:
            self.inputs = inputs
        if exec_config is not None:
            self.exec_config = exec_config

        is_mutated = (inputs is not None) or (exec_config is not None)
        if is_mutated and self.status == TaskStatus.COMPLETED:
            self.status = TaskStatus.DRAFT
            self.integrity = TaskIntegrity.PENDING

        # 2. Gatekeep: Only allow valid Tasks to become READY
        try:
            self.validate_and_serialize()
            clean_config = self.validate_config(self.exec_config)

            self.exec_config = clean_config
            if self.status != TaskStatus.COMPLETED:
                self.status = TaskStatus.READY
                self.error = None

        except MissingUserInputError as e:
            # Not an error, just waiting on a human
            self.status = TaskStatus.DRAFT
            self.error = None
        except UnresolvedReferenceError as e:
            # Not an error per se, just waiting
            self.status = TaskStatus.QUEUED
            self.error = None
        except ValueError as e:
            # Genuine validation error
            self.status = TaskStatus.DRAFT
            self.error = str(e)

    def resolve_output_type(self, output_key: str, container: Any) -> dict[str, Any]:
        """
        Calculates the semantic data type and schema of an output slot.
        Traverses the DAG if the output is a Passthrough to its input slot,
        which can chain together multiple TaskDefinitions and Adapters to 
        resolve the final data type of an output.
        """
        from lore.core.tasks.registry import task_registry

        task_def = task_registry[self.registry_key]
        out_def = task_def.output_model.model_fields.get(output_key)
        if not out_def:
            raise KeyError(
                f"Output key '{output_key}' not found in TaskDefinition '{self.registry_key}'"
            )
        _, meta = task_def.field_meta(output_key, is_output=True)
        data_type = meta.get("data_type", "unknown")

        # 1. Base case: Static type (e.g. "protein_sequence")
        if not isinstance(data_type, Passthrough):
            return {"data_type": data_type}

        # 2. Recursive case: Passthrough to an input slot
        passthrough_slot = data_type.slot
        bindings = self.inputs.get(passthrough_slot, [])

        # 3. Empty slots: nothing plugged in
        if not bindings:
            # Fall back to what the slot theoretically accepts based on the 
            # TaskDefinitions's input model
            _, input_meta = task_def.field_meta(passthrough_slot)
            accepted = input_meta.get("accepted_data", ["*"])
            fallback_type = accepted[0] if accepted else "*"
            return {"data_type": fallback_type}

        # 4. Traverse the bindings
        for b in bindings:
            if isinstance(b, ReferenceBinding):
                upstream_task = container.get_task(b.source_id)
                if upstream_task:
                    return upstream_task.resolve_output_type(b.output_key, container)

            elif isinstance(b, LiteralBinding):
                # May be a literal binding to an Artifact
                if isinstance(b.value, str) and container.get_artifact(b.value):
                    artifact = container.get_artifact(b.value)
                    return {"data_type": artifact.data_type}
                else:
                    return {"data_type": type(b.value).__name__}

        return {"data_type": "*"}


class TaskResults:
    """
    A strict container for Task outputs. All slots are lists to simplify engine
    logic, but cardinality rules are enforced at the time of adding results.
    """
    _primary_key: str | None
    _allows_multiple: dict[str, bool]

    def __init__(self, task_def: "TaskDefinition"):
        object.__setattr__(self, "_primary_key", task_def.primary_output_key)
        object.__setattr__(self, "_allows_multiple", {})

        if task_def.output_model:
            for key, field in task_def.output_model.model_fields.items():
                extra = field.json_schema_extra or {}

                raw_card = cast(Cardinality, extra.get("cardinality", Cardinality.SINGLE))
                self._allows_multiple[key] = raw_card.allows_multiple

                # Initialize all output slots as empty ordered lists
                object.__setattr__(self, key, [])

    @property
    def output_keys(self):
        """Registered output slot names in definition order."""
        return self._allows_multiple.keys()

    def __setattr__(self, name: str, value: Any):
        if name in ("_output_keys", "_primary_key", "_allows_multiple"):
            raise AttributeError(f"{name} is immutable after initialization")

        try:
            allowed = self.output_keys
        except AttributeError:
            object.__setattr__(self, name, value)
            return

        if name not in allowed:
            raise KeyError(
                f"Cannot set unknown output key: '{name}'\nAllowed keys are: {', '.join(allowed)}"
            )

        self.add(name, value)

    def __getitem__(self, key: str) -> Any:
        """Allows keyed access to Task results."""
        if key not in self.output_keys:
            raise KeyError(f"Result with key '{key}' not found.")
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Supports the 'in' operator."""
        if key not in self.output_keys:
            return False
        return len(getattr(self, key)) > 0

    def __repr__(self) -> str:
        """Human-readable representation of TaskResults showing keys and their data types."""
        contents = ", ".join(f"{k}={getattr(self, k)!r}" for k in self.output_keys)
        return f"TaskResults({contents})"

    @property
    def primary_key(self) -> str | None:
        """Returns the primary output key or first if undefined"""
        if self._primary_key and self._primary_key in self.output_keys:
            return self._primary_key
        return list(self.output_keys)[0] if self.output_keys else None

    @property
    def primary_data(self) -> Any:
        """Returns the data payload of the primary key."""
        key = self.primary_key
        return getattr(self, key) if key else None

    def next_empty_key(self) -> str | None:
        """Returns the next empty output slot key, or None if all are filled"""
        for k in self.output_keys:
            val = getattr(self, k)
            if val is None or (isinstance(val, list) and len(val) == 0):
                return k
        return None

    def add(self, name: str, value: Any):
        """
        Primary way for the engine to add results. Handles cardinality rules.
        Appends to lists if allows_multiple, otherwise does not overwrite.
        """
        if name not in self.output_keys:
            raise KeyError(
                f"Cannot set unknown output key: '{name}'. "
                f"Allowed keys are {', '.join(self.output_keys)}"
            )

        allows_multiple = self._allows_multiple[name]
        current_list = getattr(self, name)
        if not allows_multiple and len(current_list) >= 1:
            raise ValueError(
                f"Output key '{name}' allows only a single value, but multiple were added."
            )

        current_list.append(value)

    def load(self, name: str, value: Any):
        """Bypass cardinality rules for deserialization. Use with care!"""
        if name not in self.output_keys:
            raise KeyError(
                f"Cannot load unknown output key: '{name}'. "
                f"Allowed keys are {', '.join(self.output_keys)}"
            )
        object.__setattr__(self, name, value)

    def to_dict(self) -> dict:
        """
        Serialize results to a dict for storage in Manifest. Artifacts are serialized by their IDs,
        while other objects are serialized as-is (assuming they are JSON-serializable).
        """
        serialized = {}
        for k in self.output_keys:
            items = getattr(self, k)
            # Duck type Artifacts: if an item has an 'id' attribute, serialize by its ID
            serialized[k] = [item.id if hasattr(item, "id") else item for item in items]
        return serialized
