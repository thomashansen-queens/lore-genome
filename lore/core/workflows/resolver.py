"""
Workflow version of the input resolver. Since workflows are templates, this only resolves the 
"shape" of the inputs for UI rendering. No data peeking or artifact candidates.
"""
from lore.core.bindings import ReferenceBinding, LiteralBinding, UserInputBinding
from lore.core.sessions.resolver import ResolvedField
from lore.core.tasks.registry import task_registry


def resolve_workflow_task_inputs(workflow, task_id: str) -> dict[str, "ResolvedField"]:
    """
    Resolves Workflow Task Inputs for UI presentation.
    Since workflows are templates, this does not peek at live data or artifacts.
    """
    task = workflow.get_task(task_id)
    if not task:
        raise ValueError(f"Task with ID {task_id} not found in workflow.")

    task_def = task_registry.get_safe(task.registry_key)
    resolved_inputs = {}

    for key, bindings in task.inputs.items():
        _, extra = task_def.field_meta(key) if task_def else (None, {})

        # 1. Defaults
        binding_type = "literal"
        final_value = None
        ui_label = None

        # 2. Evaluate the first binding (since workflows usually only configure the primary binding in UI)
        if bindings:
            binding = bindings[0]
            if isinstance(binding, ReferenceBinding):
                binding_type = "reference"
                ui_label = f"Output '{binding.output_key}' from Task {binding.source_id[:8]}"
            elif isinstance(binding, UserInputBinding):
                binding_type = "user_input"
            elif isinstance(binding, LiteralBinding):
                binding_type = "literal"
                final_value = binding.value

        # 3. Construct the shared UI Model
        resolved_inputs[key] = ResolvedField(
            key=key, 
            label=extra.get("label", key.replace("_", " ").title()), 
            description=extra.get("description", ""), 
            data_type=extra.get("data_type", "unknown"),
            is_primary=extra.get("is_primary", False), 
            accepts_artifact=extra.get("is_artifact", False), 
            binding_type=binding_type,
            value=final_value,
            is_ready=False,  # Workflows are never "ready" to run
            ui_label=ui_label,
        )

    return resolved_inputs
