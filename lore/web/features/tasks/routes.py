"""
Routes for managing individual Tasks within a Session.
"""

import enum
from typing import Any
from pydantic_core import PydanticUndefined, ValidationError

from fastapi import BackgroundTasks, Depends, Query
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse

from lore.core.tasks import Cardinality, TaskDefinition
from lore.web.deps import RT, TEMPLATES_DIR, ActiveSession, templates, build_breadcrumbs, PageContext
from lore.web.utils import clean, get_form_str, form_to_dict
from lore.core.tasks import Task, task_registry


router = APIRouter(prefix="/sessions/{session_id}/tasks", tags=["tasks"])

# --- Helpers ---

def _apply_and_validate(
    task: "Task", inputs: dict, action: str | None = None, name: str | None = None, update_inputs: bool = True,
) -> bool:
    """
    Mutate the Task in place. Returns should_run bool.
    Must be called from within a Session lock context
    """
    if name:
        task.name = name
    if not update_inputs:
        return action == "RUN" and task.status == "PENDING"
    task.inputs = inputs
    task.error = None
    try:
        task.validate_and_serialize()
        task.status = "PENDING"
    except ValidationError as e:
        task.status = "DRAFT"
        task.error = str(e)
    return action == "RUN" and task.status == "PENDING"


def _resolve_template(registry_key: str) -> str:
    """
    Checks for custom form template for a given Task type, falls back to generic
    """
    custom_template = TEMPLATES_DIR / "features" / "tasks" / f"{registry_key}.html"
    if custom_template.exists():
        return f"/features/tasks/{registry_key}.html"
    return "/features/tasks/configure.html"


def _map_artifacts_to_inputs(
    s: ActiveSession,
    task_def: TaskDefinition,
    source_artifact_ids: list[str] | None,
) -> dict[str, Any]:
    """
    Auto-magically matches source Artifacts to compatible Task inputs
    Find which fields this Artifact could fill
    For single inputs, take the first match. For multi, take all
    """
    if not source_artifact_ids:
        return {}

    prefill = {}
    candidates = s.get_artifact_candidates(task_def)
    model_fields = task_def.input_model.model_fields

    # 1. Build mapping of Artifact ID to compatible fields
    id_to_fields = {}
    for field_name, artifacts in candidates.items():
        for a in artifacts:
            id_to_fields.setdefault(a.id, []).append(field_name)

    # 2. Assign Artifact IDs to fields
    for aid in source_artifact_ids:
        target_fields = id_to_fields.get(aid, [])
        for field_name in target_fields:
            schema = model_fields[field_name].json_schema_extra or {}
            cardinality = Cardinality(schema.get("cardinality", "single"))

            if cardinality.allows_multiple:
                prefill.setdefault(field_name, []).append(aid)
            else:
                # First come, first served for single slots
                if field_name not in prefill:
                    prefill[field_name] = aid

    return prefill


def _format_inputs_for_ui(task_def: TaskDefinition, raw_inputs: dict) -> dict:
    """Helper to transform raw Python types into UI-friendly prefill strings."""
    ui_ready = {}
    model_fields = task_def.input_model.model_fields
    for key, field_info in model_fields.items():
        extra = getattr(field_info, "json_schema_extra", {}) or {}
        widget = extra.get("widget", "text")

        # 1. Priority: User input > Schema default > None
        if key in raw_inputs and raw_inputs[key] is not None:
            val = raw_inputs[key]
        elif field_info.default is not PydanticUndefined:
            val = field_info.default
        else:
            val = None

        # 2. Consider UI widget
        if widget == "artifact_multi_select":
            # Multi-selects must have lists of IDs
            if val is None:
                ui_ready[key] = []
            elif isinstance(val, list):
                ui_ready[key] = [str(v) for v in val]
            else:
                ui_ready[key] = [str(val)]
        elif widget == "artifact_select":
            ui_ready[key] = str(val) if val is not None else ""
        else:
            # Primitive types: Standard inputs
            if val is None:
                ui_ready[key] = ""
            elif isinstance(val, list):
                # Comma-separate lists for text inputs
                ui_ready[key] = ", ".join(str(v) for v in val if v is not None)
            elif isinstance(val, bool):
                ui_ready[key] = val
            elif isinstance(val, enum.Enum):
                ui_ready[key] = val.value
            else:
                ui_ready[key] = str(val)

    return ui_ready

# --- Routes ---

@router.get("/new/{registry_key}", response_class=HTMLResponse)
def configure_new_task(
    registry_key: str,
    s: ActiveSession,
    source_artifact_ids: list[str] = Query(None),
    ctx: PageContext = Depends(),
):
    """
    Render the form for configuring a new Task.
    """
    task_def = task_registry.get(registry_key)
    if not task_def:
        raise HTTPException(404, detail=f"Task type '{registry_key}' not found")

    # 1. Check for candidate source Artifacts
    artifact_prefill = _map_artifacts_to_inputs(s, task_def, source_artifact_ids)
    prefill_inputs = _format_inputs_for_ui(task_def, artifact_prefill)

    artifact_candidates = s.get_artifact_candidates(task_def)

    ctx.breadcrumbs = build_breadcrumbs(s.id, s.name, f"New {task_def.name}")
    return templates.TemplateResponse(
        _resolve_template(registry_key),
        ctx.render(
            session=s,
            task_def=task_def,
            artifact_candidates=artifact_candidates,
            prefill_inputs=prefill_inputs,
            prefill_name=task_def.name,
            is_new=True,
        )
    )


@router.post("/new/{registry_key}", response_class=RedirectResponse)
async def create_task_action(
    registry_key: str,
    session_id: str,
    rt: RT,
    background_tasks: BackgroundTasks,
    ctx: PageContext = Depends(),
):
    """
    Submit new Task configuration, create Task, and run if requested
    """
    task_def = task_registry.get(registry_key)
    if not task_def:
        raise HTTPException(404, detail=f"Task type '{registry_key}' not found")

    form_data = await ctx.request.form()
    task_name = clean(form_data.get("task_name")) or task_def.name
    action = get_form_str(form_data, "action") or "SAVE"  # "SAVE" or "RUN"
    should_run = False
    # FUTURE: Other actions like "COMMIT" to workflow
    raw_inputs = form_to_dict(form_data, task_def.input_model)

    # Create Task
    with rt.get_session(session_id) as s:
        task = s.add_task(
            registry_key=registry_key,
            inputs=raw_inputs,
            task_name=task_name,
        )
        should_run = _apply_and_validate(task, raw_inputs, action, name=task_name)
        task_id = task.id

    # Run or save and redirect
    if should_run:
        background_tasks.add_task(
            rt.execute_task,
            session_id=session_id,
            task_id=task_id,
        )
    elif action == "RUN":
        msg = "Task saved as Draft. Invalid inputs must be fixed before running."
        return ctx.redirect(f"/sessions/{session_id}", message=msg, message_type="warning")

    return ctx.redirect(f"/sessions/{session_id}")


@router.get("/{task_id}", response_class=HTMLResponse)
def view_task(
    task_id: str,
    s: ActiveSession,
    ctx: PageContext = Depends(),
):
    """
    Ready-only view of a Task's details, outputs, and metadata
    """
    try:
        task = s.get_task(task_id)
        resolved_outputs = s.resolve_task_outputs(task_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e

    task_def = task_registry[task.registry_key]

    ctx.breadcrumbs = build_breadcrumbs(s.id, s.name, task.name or task.id)
    return templates.TemplateResponse(
        "/features/tasks/detail.html",
        ctx.render(
            session=s,
            task_def=task_def,
            task=task,
            outputs=resolved_outputs,
        )
    )


@router.get("/{task_id}/edit", response_class=HTMLResponse)
def edit_task(
    task_id: str,
    s: ActiveSession,
    artifacts: list[str] = Query(None),  # ?source=artifacts
    ctx: PageContext = Depends(),
):
    """
    Render the form editing an existing Task. Similar to the "new" form but pre-
    filled with current values.
    """
    try:
        task = s.get_task(task_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e

    task_def = task_registry[task.registry_key]
    raw_inputs = dict(task.inputs)
    if artifacts:
        raw_inputs.update(_map_artifacts_to_inputs(s, task_def, artifacts))
    prefill_inputs = _format_inputs_for_ui(task_def, raw_inputs)

    candidates = s.get_artifact_candidates(task_def)

    ctx.breadcrumbs = build_breadcrumbs(
        s.id, s.name,
        extra=[(task.name or task.id, f"/sessions/{s.id}/tasks/{task_id}")],
        final_item="Edit",
    )
    return templates.TemplateResponse(
        _resolve_template(task.registry_key),
        ctx.render(
            session=s,
            task_def=task_def,
            artifact_candidates=candidates,
            prefill_inputs=prefill_inputs,
            prefill_name=task.name,
            edit_task_id=task.id,
            is_new=False
        )
    )


@router.post("/{task_id}/edit", response_class=RedirectResponse)
async def update_task_inputs_action(
    task_id: str,
    session_id: str,
    rt: RT,
    background_tasks: BackgroundTasks,
    ctx: PageContext = Depends(),
):
    """
    Update existing Task and optionally execute
    """
    form_data = await ctx.request.form()
    action = get_form_str(form_data, "action")
    new_name = get_form_str(form_data, "name")
    is_full_edit = (action != "RENAME")

    with rt.get_session(session_id) as s:
        task = s.get_task(task_id)
        task_def = task_registry[task.registry_key]
        raw_inputs = form_to_dict(form_data, task_def.input_model)
        should_run = _apply_and_validate(task, raw_inputs, action, name=new_name, update_inputs=is_full_edit)

    # Run or save and redirect
    if should_run:
        background_tasks.add_task(
            rt.execute_task,
            session_id=session_id,
            task_id=task_id,
        )
    elif action == "RUN":
        msg = "Task saved as Draft. Invalid inputs must be fixed before running."
        return ctx.redirect(f"/sessions/{session_id}", message=msg, message_type="warning")

    return ctx.redirect(f"/sessions/{session_id}/tasks/{task_id}")

# --- Actions ---

@router.post("/{task_id}/run", response_class=RedirectResponse)
def run_task_only_action(
    session_id: str,
    task_id: str,
    background_tasks: BackgroundTasks,
    rt: RT,
    ctx: PageContext = Depends(),
):
    """Run a pending task"""
    # Use Runtime to get fresh session lock
    with rt.get_session(session_id) as s:
        task = s.get_task(task_id)
        if task.status == "RUNNING":
            # prevent double runs, concurrent writes, etc.
            return ctx.redirect_back(fallback_url=f"/sessions/{session_id}")

        # Force re-validation in case Task was somehow edited since last save
        try:
            # Reset before running; useful for re-running failed
            _ = task.validate_and_serialize()
            task.status = "PENDING"
            task.error = None
        except ValidationError as e:
            task.status = "DRAFT"
            task.error = str(e)
            msg = "Task saved as Draft. Invalid inputs must be fixed before running."
            msg_type = "warning"
            return ctx.redirect_back(fallback_url=f"/sessions/{s.id}", message=msg, message_type=msg_type)

    background_tasks.add_task(rt.execute_task, session_id, task_id)
    return ctx.redirect_back(fallback_url=f"/sessions/{session_id}")


@router.post("/{task_id}/clone", response_class=RedirectResponse)
def clone_task_action(task_id: str, s: ActiveSession, ctx: PageContext = Depends()):
    """Create a deep copy of this Task with a new ID. Does not run the Task."""
    new_task = s.clone_task(task_id)
    return ctx.redirect_back(
        fallback_url=f"/sessions/{s.id}/tasks/{new_task.id}",
        message="Task cloned. Remember to update inputs before running.",
        message_type="info",
    )


@router.post("/{task_id}/delete", response_class=RedirectResponse)
def delete_task_action(task_id: str, s: ActiveSession, ctx: PageContext = Depends()):
    """Delete this Task. NOTE: This is irreversible and will orphan downstream Artifacts."""
    deleted = s.delete_task(task_id)
    if not deleted:
        return ctx.redirect_back(fallback_url=f"/sessions/{s.id}", message="Something went wrong", message_type="danger")
    return ctx.redirect(f"/sessions/{s.id}", message="Task deleted", message_type="success")
