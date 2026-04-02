"""
Resolves inputs and outputs from Tasks
"""

from typing import Any
from pydantic import BaseModel

from lore.core.artifacts import BaseArtifact, FutureArtifact
from lore.core.bindings import LiteralBinding, ReferenceBinding, UserInputBinding
from lore.core.tasks import task_registry


class ResolvedField(BaseModel):
    """A strict data contract representing a Task input or output for the UI/Planner."""
    key: str
    label: str
    description: str
    data_type: str
    is_primary: bool
    accepts_artifact: bool
    value: Any | list[BaseArtifact] | list[str]


def resolve_task_inputs(session, task_id: str) -> dict[str, ResolvedField]:
    """
    Resolves Task Inputs for UI presentation.
    Dynamically generates FutureArtifacts for Workflow references.
    """
    task = session.get_task(task_id)
    if not task:
        raise ValueError(f"Task with ID {task_id} not found in session.")

    task_def = task_registry.get(task.registry_key)
    if not task_def:
        raise ValueError(f"Task definition for key {task.registry_key} not found.")

    resolved_inputs = {}

    for key, bindings in task.inputs.items():
        _, extra = task_def.field_meta(key)
        label = key.replace("_", " ").capitalize()
        description = extra.get("description", "")
        data_type = extra.get("data_type", "unknown")
        is_primary = extra.get("is_primary", False)
        accepts_artifact = extra.get("is_artifact", False)

        cardinality = extra.get("cardinality")
        allows_multiple = cardinality.allows_multiple if cardinality is not None else False

        resolved_list = []

        # 2. Evaluate every binding and resolve references
        for binding in bindings:
            if isinstance(binding, ReferenceBinding):
                if accepts_artifact:
                    # A. Future Artifact
                    resolved_list.append(FutureArtifact(
                        id=f"ref:{binding.source_id}.{binding.output_key}",
                        data_type=data_type,
                        source_step_id=binding.source_id,
                        source_output_key=binding.output_key,
                    ))
                else:
                    # B. Future primitive value
                    resolved_list.append(binding)
            elif isinstance(binding, UserInputBinding):
                resolved_list.append(binding)
            elif isinstance(binding, LiteralBinding):
                if accepts_artifact and isinstance(binding.value, str) and session.get_artifact(binding.value):
                    # C. Concrete Artifact ID
                    resolved_list.append(session.get_artifact(binding.value))
                else:
                    # D. Literal input
                    resolved_list.append(binding.value)
            else:
                raise ValueError(f"Unsupported input binding type for key '{key}': {type(binding)}")

        # 3. De-normalize single-value lists
        if allows_multiple:
            final_value = resolved_list
        else:
            final_value = resolved_list[0] if resolved_list else None

        resolved_inputs[key] = ResolvedField(
            key=key, label=label, description=description, data_type=data_type,
            is_primary=is_primary, accepts_artifact=accepts_artifact, value=final_value,
        )

    return resolved_inputs


def resolve_task_outputs(session, task_id: str) -> dict[str, ResolvedField]:
    """
    Resolves Task outputs for UI presentation.
    """
    # 1. Fetch and guard
    task = session.get_task(task_id)
    if task is None:
        raise ValueError(f"Cannot resolve outputs for non-existent task ID: {task_id}")

    if not task.outputs:
        return {}

    task_def = task_registry[task.registry_key]
    schema_fields = task_def.output_model.model_fields if (task_def and task_def.output_model) else {}

    resolved = {}
    for key, val in task.outputs.items():
        field_info = schema_fields.get(key)

        # 2. Sensible defaults in case of sparse schema
        is_artifact = False
        label = key.replace("_", " ").capitalize()
        description = ""
        is_primary = False
        data_type = "unknown"

        # 3. Get rich metadata
        if field_info:
            label = field_info.title or label
            description = field_info.description or description

            # LoRe specific metadata
            extra = getattr(field_info, "json_schema_extra", None)
            if extra is not None and isinstance(extra, dict):
                is_artifact = extra.get("is_artifact", True)
                is_primary = extra.get("is_primary", False)
                data_type = extra.get("data_type", "unknown")
                label = extra.get("label", label)

        # 4. Resolve Artifacts if applicable — val is always list[str] of artifact IDs
        if not is_artifact:
            resolved_value = val
        else:
            resolved_value = []
            for artifact_id in (val or []):
                artifact = session.get_artifact(artifact_id)
                if artifact is None:
                    session.logger.warning(
                        "Output '%s' of task '%s' (ID: %s) references artifact ID '%s' that cannot be found.",
                        key, task.name, task.id, artifact_id
                    )
                    resolved_value.append("MISSING")
                else:
                    resolved_value.append(artifact)

        # 5. Package the resolved output with its metadata for convenient presentation
        resolved[key] = {
            "key": key,
            "label": label,
            "description": description,
            "data_type": data_type,
            "is_primary": is_primary,
            "is_artifact": is_artifact,
            "value": resolved_value,
        }

    return resolved
