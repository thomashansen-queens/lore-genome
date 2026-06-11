"""
The Task Registry is initialized at Runtime and serves as the global source of truth for all
available Task definitions.
"""
from typing import Any, Type
from pydantic import BaseModel, Field, create_model

from .definition import PreviewMode, PreviewModeLiteral, TaskDefinition
from .parameters import TaskInput, TaskOutput
from lore.core.utils import iter_dsl_attrs, has_field_type


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
        inputs: type["BaseModel"] | type,
        outputs: type["BaseModel"] | type,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        icon: str | None = None,
        preview_mode: PreviewModeLiteral | PreviewMode = PreviewMode.NONE,
    ):
        """
        Decorator to registers a TaskDefinition with a unique key.
        Internally compiles LoRē TaskInput DSL to Pydantic model for validation
        and UI generation. Also allows for raw Pydantic models for power users.

        1. Validates uniqueness of the task key.
        2. Compiles input and output models from LoRē DSL to Pydantic if necessary.
        3. Auto-generates metadata like name and description if not provided.

        Previews can be enabled by setting the `preview_mode` parameter, which 
        controls how the preview is generated and displayed (defaults tp "none" for
        safety).
        """
        def _compile_inputs_to_pydantic(task_key: str, input_model: type) -> type[BaseModel]:
            """
            Compiles a LoRē TaskInput DSL class to a Pydantic BaseModel.
            Inspects the fields of the provided dataclass, converts TaskInput
            fields to Pydantic Fields with appropriate metadata, and constructs
            a new Pydantic model class dynamically
            """
            fields = {}

            # Iterate to convert TaskInput fields to Pydantic field definitions
            for attr_name, attr_value in iter_dsl_attrs(input_model):
                if callable(attr_value):
                    continue  # Skip methods

                if isinstance(attr_value, TaskInput):
                    field_type = attr_value.get_type_annotation()
                    field_info = attr_value.to_field_info()
                    fields[attr_name] = (field_type, field_info)
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

        def _compile_outputs_to_pydantic(task_key: str, dsl_outputs: type) -> Type["BaseModel"]:
            """
            Turns a list of TaskOutput definitions into a Pydantic model for
            documentation and validation.
            """
            fields = {}
            for attr_name, attr_value in iter_dsl_attrs(dsl_outputs):
                if callable(attr_value):
                    continue

                if isinstance(attr_value, TaskOutput):
                    fields[attr_name] = (
                        str,
                        Field(
                            description=attr_value.description,
                            # Pydantic will serialize Enums at runtime; ignoring JsonDict type req
                            json_schema_extra={  # pyright: ignore[reportArgumentType]
                                "data_type": attr_value.data_type,
                                "description": attr_value.description,
                                "is_primary": attr_value.is_primary,
                                "cardinality": attr_value.cardinality.value,
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
            # 1. Guards
            if key in self._tasks:
                raise ValueError(f"Task with key '{key}' is already registered.")

            try:
                preview_mode_enum = PreviewMode(preview_mode)
            except ValueError:
                raise ValueError(
                    f"Invalid preview_mode '{preview_mode}' for task '{key}'. "
                    f"Must be one of {", ".join([mode.value for mode in PreviewMode])}."
                )

            # 2. Resolve input model (LoRe TaskInput or Pydantic BaseModel)
            if isinstance(inputs, type) and issubclass(inputs, BaseModel):
                final_input_model = inputs
            elif has_field_type(inputs, TaskInput):
                final_input_model = _compile_inputs_to_pydantic(key, inputs)
            else:
                raise ValueError(
                    f"Inputs for {key} must be a Class of TaskInput objects or a "
                    f"Pydantic BaseModel."
                )

            # 3. Similar logic for outputs
            if isinstance(outputs, type) and issubclass(outputs, BaseModel):
                final_output_model = outputs
            elif has_field_type(outputs, TaskOutput):
                final_output_model = _compile_outputs_to_pydantic(key, outputs)
            else:
                raise ValueError(f"Outputs for {key} must be a Class of TaskOutput objects.")

            # 4. Auto-generate metadata if not provided
            final_name = name or key.split(".")[-1].replace("_", " ").capitalize()
            final_category = category or (key.split(".")[0] if "." in key else "General")
            final_icon = icon or "⚡"
            docstring = func.__doc__ or ""
            final_description = description or " ".join(
                [line.strip() for line in docstring.split("\n") if line.strip()]
            )

            task_def = TaskDefinition(
                key=key,
                handler=func,
                input_model=final_input_model,
                output_model=final_output_model,
                description=final_description,
                name=final_name,
                category=final_category,
                icon=final_icon,
                preview_mode=preview_mode_enum,
            )

            self._tasks[key] = task_def

            # 5. Build reverse index for UI to match Artifacts to Tasks
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
            description=f"Auto-generated placeholder for an unavailable Task: {key}",
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
