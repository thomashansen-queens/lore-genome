"""
The Task Registry is initialized at Runtime and serves as the global source of truth for all
available Task definitions.
"""

from typing import Type, Any
from pydantic import BaseModel, Field, create_model

from lore.core.tasks.parameters import TaskInput, TaskOutput
from lore.core.tasks.models import TaskDefinition


class TaskRegistry:
    """
    Registry for managing Task definitions. Use the `register` method as a decorator to register
    Task handler functions.
    """

    def __init__(self):
        self._tasks: dict[str, TaskDefinition] = {}
        # O(1) reverse lookup for UI suggestions
        self._type_index: dict[str, set[TaskDefinition]] = {}
        self._universal_tasks: set[TaskDefinition] = set()

    # --- Dict-like behaviour ---

    def __getitem__(self, key: str) -> TaskDefinition:
        """Allows dict-like access to Task definitions"""
        if key not in self._tasks:
            raise KeyError(f"Task with key '{key}' not found.")
        return self._tasks[key]

    def __contains__(self, key: str) -> bool:
        """Supports the 'in' operator."""
        return key in self._tasks

    def __delitem__(self, key: str) -> None:
        """Supports the 'del' operator."""
        if key not in self._tasks:
            raise KeyError(f"Task with key '{key}' not found.")

        task_def = self._tasks.pop(key)

        # Clean up the UI reverse index to prevent memory leaks
        self._universal_tasks.discard(task_def)
        for dtype_set in self._type_index.values():
            dtype_set.discard(task_def)

    def __iter__(self):
        """Allows iterating over the registry (e.g. for key in task_registry:)."""
        return iter(self._tasks)

    # --- Task registry logic ---

    def register(
        self,
        key: str,
        inputs: Type[BaseModel] | Type[Any],
        outputs: Type[BaseModel] | Type[Any],
        name: str | None = None,
        category: str | None = None,
        icon: str | None = None,
        live_preview: bool = False,
    ):
        """
        Decorator to registers a TaskDefinition with a unique key.
        Internally compiles LoRe TaskInput DSL to Pydantic model for validation
        and UI generation. Also allows for raw Pydantic models for power users.
        """

        def _compile_inputs_to_pydantic(task_key: str, input_model: Type[Any]) -> Type[BaseModel]:
            """
            Compiles a LoRe TaskInput DSL class to a Pydantic BaseModel.
            Inspects the fields of the provided dataclass, converts TaskInput
            fields to Pydantic Fields with appropriate metadata, and constructs
            a new Pydantic model class dynamically
            """
            fields = {}

            # Iterate to convert TaskInput fields to Pydantic field definitions
            # Using dir() to walk the Method Resolution Order, allowing inheritance
            for attr_name in dir(input_model):
                if attr_name.startswith("__") or callable(getattr(input_model, attr_name)):
                    continue  # Skip dunder methods and attributes
                attr_value = getattr(input_model, attr_name)
                if isinstance(attr_value, TaskInput):
                    py_type = attr_value.get_type_annotation()
                    field_info = attr_value.to_field_info()
                    fields[attr_name] = (py_type, field_info)
                else:
                    raise ValueError(
                        f"Attribute '{attr_name}' in {input_model.__name__} is not a TaskInput."
                        f"Only TaskInput fields are allowed in DSL input models."
                    )

            # Dynamic model creation with a unique name base on the task key
            safe_name = f"{task_key.replace('.', '_')}_InputModel"
            model = create_model(safe_name, **fields)
            model.__doc__ = input_model.__doc__  # Preserve docstring for the model

            return model

        def _compile_outputs_to_pydantic(task_key: str, dsl_outputs: Type[Any]) -> Type[BaseModel]:
            """
            Turns a list of TaskOutput definitions into a Pydantic model for
            documentation and validation.
            """
            fields = {}
            for attr_name in dir(dsl_outputs):
                if attr_name.startswith("__") or callable(getattr(dsl_outputs, attr_name)):
                    continue
                attr_value = getattr(dsl_outputs, attr_name)
                if isinstance(attr_value, TaskOutput):
                    fields[attr_name] = (
                        str,
                        Field(
                            description=attr_value.description,
                            json_schema_extra={
                                "data_type": attr_value.data_type,
                                "description": attr_value.description,
                                "is_primary": attr_value.is_primary,
                                "cardinality": attr_value.cardinality,
                                "is_artifact": attr_value.is_artifact,
                                "is_output": True,  # Hint for UI
                            },
                        ),
                    )
                else:
                    raise ValueError(
                        f"Attribute '{attr_name}' in {dsl_outputs.__name__} is not a TaskOutput."
                        f"Only TaskOutput fields are allowed in DSL output models."
                    )

            model = create_model(f"{task_key.replace('.', '_')}_OutputModel", **fields)
            model.__doc__ = dsl_outputs.__doc__
            return model

        def wrapper(func):
            if key in self._tasks:
                raise ValueError(f"Task with key '{key}' is already registered.")

            # 1. Check if LoRe TaskInput fields are in input model (including inherited)
            is_dsl = any(
                isinstance(getattr(inputs, attr), TaskInput)
                for attr in dir(inputs)
                if not attr.startswith("__")
            )

            if is_dsl:
                final_input_model = _compile_inputs_to_pydantic(key, inputs)
            else:
                raise ValueError(f"Inputs for {key} must be a Class of TaskInput objects.")

            # 2. Similar logic for outputs
            is_output_dsl = any(isinstance(v, TaskOutput) for v in outputs.__dict__.values())
            if is_output_dsl:
                final_output_model = _compile_outputs_to_pydantic(key, outputs)
            else:
                raise ValueError(f"Outputs for {key} must be a Class of TaskOutput objects.")

            # 3. Auto-generate metadata if not provided
            final_name = name or key.split(".")[-1].replace("_", " ").capitalize()
            final_category = category or (key.split(".")[0] if "." in key else "General")
            final_icon = icon or "⚡"

            task_def = TaskDefinition(
                key=key,
                handler=func,
                input_model=final_input_model,
                output_model=final_output_model,
                description=" ".join(
                    [line.strip() for line in func.__doc__.split("\n") if line.strip()]
                )
                or "",
                name=final_name,
                category=final_category,
                icon=final_icon,
                live_preview=live_preview,
            )

            self._tasks[key] = task_def

            # 4. Build reverse index for UI to match Artifacts to Tasks
            for field_name in task_def.input_model.model_fields.keys():
                _, extra = task_def.field_meta(field_name)
                accepted_data = extra.get("accepted_data", [])

                if "*" in accepted_data:
                    self._universal_tasks.add(task_def)
                else:
                    for data_type in accepted_data:
                        if data_type not in self._type_index:
                            self._type_index[data_type] = set()
                        self._type_index[data_type].add(task_def)

            return func

        return wrapper

    def get(self, key: str, default: Any = None) -> TaskDefinition | None:
        """Retrieve a Task definition by its key."""
        return self._tasks.get(key, default)

    def get_safe(self, key: str) -> TaskDefinition:
        """
        Retrieve a TaskDefinition, or generate a placeholder if not found. Useful for imported 
        workflows that reference unavailable tasks.
        """
        if key in self._tasks:
            return self._tasks[key]

        return TaskDefinition(
            name=f"Unavailable Task ({key})",
            description="",
            key=key,
            handler=lambda: None,  # No-op handler
            input_model=create_model(f"{key}_InputModel"),
            output_model=create_model(f"{key}_OutputModel"),
        )

    @property
    def all(self) -> dict[str, TaskDefinition]:
        """Get all registered Task definitions."""
        return self._tasks

    def compatible_tasks(self, resolvable_types: set[str]) -> list[TaskDefinition]:
        """
        Given a set of provided data types, return a list of TaskDefinitions that
        can accept at least one of those types as input. Used for UI suggestions.
        """
        compatible: set[TaskDefinition] = set(self._universal_tasks)

        for dtype in resolvable_types:
            if dtype in self._type_index:
                compatible.update(self._type_index[dtype])

        return list(compatible)


# Global Task registry instance
task_registry = TaskRegistry()
