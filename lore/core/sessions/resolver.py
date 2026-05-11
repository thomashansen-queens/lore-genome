"""
Resolves inputs and outputs from Tasks
"""

from typing import Any, Literal
from pydantic import BaseModel

from lore.core.artifacts import BaseArtifact
from lore.core.bindings import LiteralBinding, ReferenceBinding, UserInputBinding
from lore.core.tasks import task_registry


class ResolvedField(BaseModel):
    """
    A strict data contract representing a Task input or output for the UI/Planner.
    UI:
        binding_type: literal, reference, user_input, output
        value: the literal value, or JIT-resolved peek value
        is_ready: True if upstream reference is COMPLETED
        source_ref: e.g. "Step 1 (report)" for UI rendering
    """
    key: str
    label: str
    description: str
    data_type: str
    is_primary: bool
    accepts_artifact: bool

    # UI exposed fields
    binding_type: Literal["literal", "reference", "user_input", "output"]
    value: Any | list[BaseArtifact] | list[str]
    is_ready: bool
    ui_label: str | None = None


def _format_reference_label(session, binding: ReferenceBinding) -> str:
    """Helper to generate a human-readable label for a ReferenceBinding."""
    upstream_task = session.get_task(binding.source_id)
    task_name = upstream_task.name if upstream_task else binding.source_id[:8]
    return f"{task_name} ({binding.output_key})"


def resolve_task_inputs(session, task_id: str) -> dict[str, ResolvedField]:
    """
    Resolves Task Inputs for UI presentation.
    Performs JIT resolution on ReferenceBindings to peek at upstream data.
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

        cardinality = extra.get("cardinality")
        allows_multiple = cardinality.allows_multiple if cardinality is not None else False

        resolved_list = []
        is_ready = True
        binding_type = "literal"  # default: assume literal
        source_ref = None

        # 2. Evaluate every binding and resolve references
        ui_label = None
        for binding in bindings:
            if isinstance(binding, ReferenceBinding):
                binding_type = "reference"
                upstream_task = session.get_task(binding.source_id)
                ui_label = _format_reference_label(session, binding)
                
                if upstream_task and upstream_task.status == "COMPLETED":
                    # Peek: Look at the data from the completed upstream task
                    peek_values = upstream_task.get(binding.output_key, [])
                    if extra.get("is_artifact"):
                        resolved_list.extend([session.get_artifact(aid) for aid in peek_values])
                    else:
                        resolved_list.extend(peek_values)
                
                else:
                    # Upstream is not done yet
                    is_ready = False
                    resolved_list.append(None)

            elif isinstance(binding, UserInputBinding):
                binding_type = "user_input"
                is_ready = False
                resolved_list.append(binding)

            elif isinstance(binding, LiteralBinding):
                if extra.get("is_artifact") and session.get_artifact(binding.value) and isinstance(binding.value, str):
                    # C. Concrete Artifact ID
                    resolved_list.append(session.get_artifact(binding.value))
                else:
                    # D. Literal input
                    resolved_list.append(binding.value)
            else:
                raise ValueError(f"Unsupported input binding type for key '{key}': {type(binding)}")

        # 3. De-normalize single-value lists
        if allows_multiple:
            final_value = [v for v in resolved_list if v is not None]
        else:
            # If multiple bindings are provided but only one is allowed, take the first
            valid_items = [v for v in resolved_list if v is not None]
            final_value = valid_items[0] if valid_items else None

        resolved_inputs[key] = ResolvedField(
            key=key, 
            label=key.replace("_", " ").capitalize(), 
            description=extra.get("description", ""), 
            data_type=extra.get("data_type", "unknown"),
            is_primary=extra.get("is_primary", False), 
            accepts_artifact=extra.get("is_artifact", False), 
            binding_type=binding_type,
            value=final_value,
            is_ready=is_ready,
            ui_label=ui_label,
        )

    return resolved_inputs


def resolve_task_outputs(session, task_id: str) -> dict[str, ResolvedField]:
    """
    Resolves Task outputs into standard ResolvedFields for UI presentation.
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
        resolved[key] = ResolvedField(
            key=key,
            label=label,
            description=description,
            data_type=data_type,
            is_primary=is_primary,
            accepts_artifact=is_artifact,
            binding_type="output",
            value=resolved_value,
            is_ready=True,  # Outputs existing means they are ready
            ui_label=None,  # Outputs don't need a UI label
        )

    return resolved
