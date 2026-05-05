"""
Helper to render the input widget to configure tasks.
"""

from lore.core.artifacts.models import Artifact
from lore.core.bindings import LiteralBinding, ReferenceBinding, UserInputBinding
from lore.core.sessions.session import Session
from lore.core.sessions.matcher import find_artifact_candidates, map_artifacts_to_task_inputs
from lore.core.tasks.models import Task, TaskDefinition
from lore.core.tasks.parameters import Cardinality
from lore.core.tasks.registry import task_registry
from lore.core.topology.traversal import find_valid_upstream_tasks
from lore.core.workflows.models import Workflow
from lore.web.utils.forms import format_inputs_for_ui


def build_task_configure_context(
    container: Session | Workflow,
    task_key: str,
    task_id: str | None = None,
    source_artifact_ids: list[str] | None = None,
) -> dict:
    """Builds the universal UI context for configuring a Task."""

    task_def = task_registry.get_safe(task_key)
    if not task_def:
        raise ValueError(f"Task '{task_key}' not found in registry.")

    # 1. Detect Session (live) vs. Workflow (template) container
    is_session = isinstance(container, Session)
    tasks = container.list_tasks() if is_session else container.tasks

    # 2. Handle Edit Mode & Resolution
    prefill_name = task_def.name
    task = None
    if task_id:
        task = container.get_task(task_id)
        if not task:
            raise ValueError(f"Task with ID {task_id} not found in container.")
        prefill_name = task.name

    # 3. Handle Artifacts (Session ONLY)
    raw_inputs = {}
    artifact_candidates = {}
    if is_session:
        if source_artifact_ids:
            raw_inputs.update(
                map_artifacts_to_task_inputs(container, task_def, source_artifact_ids)
            )
        artifact_candidates = find_artifact_candidates(container, task_def)

    prefilled_inputs = format_inputs_for_ui(task_def, raw_inputs)

    # 4. Sort Fields (Required first, then Optional)
    sorted_fields = {}
    optional_fields = {}
    for f_name, f_info in task_def.input_model.model_fields.items():
        if f_info.is_required():
            sorted_fields[f_name] = f_info
        else:
            optional_fields[f_name] = f_info
    sorted_fields.update(optional_fields)

    # 5. Build the UI state for all fields
    fields_ui = {}
    for field_name in sorted_fields.keys():
        _, field_extra = task_def.field_meta(field_name)
        fields_ui[field_name] = _build_field_ui_state(
            container=container,
            task_def=task_def,
            field_name=field_name,
            task=task,
            raw_prefill=prefilled_inputs.get(field_name),
            avail_tasks=find_valid_upstream_tasks(task_id, tasks, field_extra),
            candidates=artifact_candidates.get(field_name, []),
            mode_override=None,
        )

    # 6. Return the payload
    return {
        "task": task,
        "task_def": task_def,
        "prefill_name": prefill_name,
        "sorted_fields": sorted_fields,
        "fields_ui": fields_ui,
        "edit_task_id": task_id,
    }


def build_widget_context(
    container: Session | Workflow,
    task_key: str,
    field_name: str,
    new_mode: str,
    task_id: str | None = None,
) -> dict:
    """Builds the universal UI context for an HTMX widget hot-swap."""
    task_def = task_registry.get_safe(task_key)
    if not task_def:
        raise ValueError(f"Task '{task_key}' not found in registry.")

    field_info, field_extra = task_def.field_meta(field_name)

    tasks = container.list_tasks() if isinstance(container, Session) else container.tasks
    task = container.get_task(task_id) if task_id else None

    avail_tasks = find_valid_upstream_tasks(task_id, tasks, field_extra)

    # 2. Artifact Candidates (Session ONLY)
    artifact_candidates = []
    if isinstance(container, Session) and new_mode in ["literal", "user_input"]:
        # Re-fetch candidates to ensure we have the latest state
        artifact_candidates = find_artifact_candidates(container, task_def).get(field_name, [])

    # 3. Build the field UI state
    ui_state = _build_field_ui_state(
        container=container,
        task_def=task_def,
        field_name=field_name,
        task=task,
        raw_prefill=None,
        avail_tasks=avail_tasks,
        candidates=artifact_candidates,
        mode_override=new_mode,
    )

    # 4. Return payload
    ui_state["field_name"] = field_name
    ui_state["field_info"] = field_info
    return ui_state


def _build_field_ui_state(
    container: Session | Workflow,
    task_def: TaskDefinition,
    field_name: str,
    task: Task | None,
    raw_prefill: str | None,
    avail_tasks: list[dict],
    candidates: list[Artifact],
    mode_override: str | None
) -> dict:
    """The brain for figuring out what the UI should display for a given field."""
    # 1. Widget URL builder
    is_session = isinstance(container, Session)
    if is_session:
        widget_url = (
            f"/sessions/{container.id}/tasks/configure/{task_def.key}/input-widget"
            f"?field_name={field_name}"
        )
        if task:
            widget_url += f"&task_id={task.id}"
    else:
        if not task:
            raise ValueError("Task ID must be provided when configuring a Workflow template.")
        widget_url = (
            f"/workflows/{container.id}/tasks/{task.id}/input-widget"
            f"?field_name={field_name}"
        )

    # 2. Inspect bindings
    raw_bindings = task.inputs.get(field_name) if task else None

    # 3. Determine binding mode (literal vs. reference vs. user_input)
    binding_mode = "literal"
    if mode_override:
        binding_mode = mode_override
    elif raw_bindings:
        if all(isinstance(b, ReferenceBinding) for b in raw_bindings):
            binding_mode = "reference"
        elif all(isinstance(b, UserInputBinding) for b in raw_bindings):
            binding_mode = "user_input"

    # 4. Extract Literals (can handle mixed IDs and manual strings)
    val = None
    if binding_mode in ["literal", "user_input"]:
        if raw_bindings:
            # Extract raw strings/primitives from LiteralBindings
            extracted = [b.value for b in raw_bindings if isinstance(b, LiteralBinding)]

            _, meta = task_def.field_meta(field_name)
            cardinality = meta.get("cardinality", Cardinality.SINGLE)

            if meta.get("is_artifact"):
                # ArtifactInputs need to remain a Python list
                if cardinality.allows_multiple:
                    val = extracted
                else:
                    val = extracted[0] if extracted else None
            else:
                # ValueInputs need to be stiched back into a single comma-separated list
                if len(extracted) > 1:
                    val = ", ".join(str(v) for v in extracted)
                else:
                    val = extracted[0] if extracted else None
        else:
            # Fallback to URL ?source_artifact_ids=... prefill
            val = raw_prefill

    # 5. Extract Reference String
    ref_val = None
    if binding_mode == "reference" and raw_bindings:
        b = raw_bindings[0]
        if isinstance(b, ReferenceBinding):
            ref_val = f"ref:{b.source_id}::{b.output_key}"

    return {
        "widget_url": widget_url,
        "binding_mode": binding_mode,
        "literal_val": val,
        "ref_val": ref_val,
        "avail_tasks": avail_tasks,
        "candidates": candidates,
    }
