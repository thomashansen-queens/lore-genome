"""
Routes for managing individual Tasks within a Session.
"""

from pydantic import BaseModel, Field

from fastapi import Depends, Query
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse

from lore.core.bindings import LiteralBinding
from lore.core.pipeline import find_artifact_candidates, map_artifacts_to_task_inputs, resolve_task_outputs
from lore.core.tasks.models import TaskConfig
from lore.web.deps import RT, ActiveSession, ReadOnlySession, templates, PageContext
from lore.web.utils.forms import get_form_str, format_inputs_for_ui, form_json_to_dict
from lore.core.tasks import task_registry, TaskStatus


router = APIRouter(prefix="/sessions/{session_id}/tasks", tags=["tasks"])


@router.get("/", response_class=RedirectResponse)
def redirect_to_session(session_id: str, ctx: PageContext = Depends()):
    """Redirect /sessions/{session_id}/tasks to the Session dashboard."""
    return ctx.redirect(f"/sessions/{session_id}")


@router.get("/new", response_class=HTMLResponse)
async def list_available_tasks(
    s: ReadOnlySession,
    ctx: PageContext = Depends(),
):
    """
    Displays all available Tasks that can be configured and committed to the 
    Session.
    """
    # We pass the whole registry values so Jinja can loop through them
    available_tasks = list(task_registry.all.values())

    ctx.generate_breadcrumbs({s.id: s.name, "catalogue": "Task Catalogue"})
    return templates.TemplateResponse(
        request=ctx.request,
        name="features/tasks/catalogue.html",
        context=ctx.render(
            session=s,
            tasks=available_tasks,
        )
    )


@router.get("/new/{task_key}", response_class=HTMLResponse)
async def configure_new_task(
    task_key: str,
    s: ReadOnlySession,
    ctx: PageContext = Depends(),
    load_task: str | None = Query(None),
    source_artifact_ids: list[str] = Query(None),
):
    """
    Universal Workbench UI. Shows Task config and preview panel. Can be pre-
    filled with existing Task (?load_task=ID) and Artifacts (?source_artifact_ids=ID)
    """
    # 1. Load the Task Definition
    task_def = task_registry.get(task_key)
    if not task_def:
        raise HTTPException(status_code=404, detail=f"Task '{task_key}' not found in registry.")

    # 2. Edit if needed
    raw_inputs = {}
    prefill_name = task_def.name
    task_id = None

    if load_task:
        task = s.get_task(load_task)
        if task:
            prefill_name = task.name
            task_id = task.id

            # 3. Unwrap bindings for UI
            for k, bindings in task.inputs.items():
                vals = [b.value for b in bindings if isinstance(b, LiteralBinding)]
                if not vals:
                    continue  # skip non-literal bindings in prefill

                # if multiple values, keep as list
                raw_inputs[k] = vals if len(vals) > 1 else vals[0]

    # 4. Pushed Artifacts
    if source_artifact_ids:
        raw_inputs.update(map_artifacts_to_task_inputs(s, task_def, source_artifact_ids))

    # 5. Format for UI
    prefilled_inputs = format_inputs_for_ui(task_def, raw_inputs)
    artifact_candidates = find_artifact_candidates(s, task_def)

    ctx.generate_breadcrumbs({s.id: s.name, "configure": "Configure Task"})
    return templates.TemplateResponse(
        request=ctx.request,
        name="features/tasks/configure.html",
        context=ctx.render(
            session=s,
            task_def=task_def,
            prefill_name=prefill_name,
            prefilled_inputs=prefilled_inputs,
            artifact_candidates=artifact_candidates,
            edit_task_id=task_id,
            preview_api_url=f"/sessions/{s.id}/tasks/new/{task_key}/preview",
            commit_api_url=f"/sessions/{s.id}/tasks/new/{task_key}",
        )
    )


@router.get("/{task_id}", response_class=HTMLResponse)
def view_task(
    task_id: str,
    s: ReadOnlySession,
    ctx: PageContext = Depends(),
):
    """
    Ready-only view of a Task's details, outputs, and metadata
    """
    task = s.get_task(task_id)
    if task is None:
        raise HTTPException(404, detail=f"Task with ID '{task_id}' not found in Session '{s.id}'.")

    resolved_outputs = resolve_task_outputs(s, task_id)

    task_def = task_registry.get(task.registry_key)
    if not task_def:
        raise HTTPException(404, detail=f"Task definition '{task.registry_key}' not found in registry.")

    task_log = s.get_task_log(task_id)

    ctx.generate_breadcrumbs({s.id: s.name, task_id: task.name or "Task Details"})
    return templates.TemplateResponse(
        request=ctx.request,
        name="/features/tasks/detail.html",
        context=ctx.render(
            session=s,
            task_def=task_def,
            task=task,
            outputs=resolved_outputs,
            task_log=task_log,
        )
    )


@router.get("/{task_id}/edit", response_class=RedirectResponse)
def edit_task(
    task_id: str,
    s: ReadOnlySession,
    ctx: PageContext = Depends(),
    source_artifact_ids: list[str] = Query(None), # Keep this for lineage!
):
    """
    Redirects the user to the configuration page, pre-loading this Task's
    inputs as the starting Draft.
    """
    task = s.get_task(task_id)
    if task is None:
        raise HTTPException(404, detail=f"Task with ID '{task_id}' not found in Session '{s.id}'.")

    target_url = f"/sessions/{s.id}/tasks/new/{task.registry_key}?load_task={task.id}"

    # Forward any source artifacts if they exist
    if source_artifact_ids:
        # FastAPI handles list queries as multiple params (e.g., ?source_artifact_ids=1&source_artifact_ids=2)
        query_string = "&".join([f"source_artifact_ids={aid}" for aid in source_artifact_ids])
        target_url += f"&{query_string}"

    return ctx.redirect(target_url)

# --- Actions ---

class TaskPayload(BaseModel):
    """Contract for the AJAX payload when previewing or committing a Task"""
    inputs: dict
    exec_config: TaskConfig = Field(default_factory=TaskConfig)


@router.post("/{task_id}/rename", response_class=RedirectResponse)
async def rename_task_action(
    task_id: str,
    s: ActiveSession,
    ctx: PageContext = Depends(),
):
    """Dedicated endpoint for in-place renaming of Tasks."""
    form_data = await ctx.request.form()
    new_name = get_form_str(form_data, "name")

    if new_name:
        s.rename_task(task_id, new_name)
    return ctx.redirect_back(fallback_url=f"/sessions/{s.id}/tasks/{task_id}")


@router.post("/{task_id}/run", response_class=RedirectResponse)
def run_task_action(
    session_id: str,
    task_id: str,
    rt: RT,
    ctx: PageContext = Depends(),
):
    """Execute a READY task"""
    # 1. Use Runtime to get fresh session lock
    with rt.open_session(session_id) as s:
        task = s.get_task(task_id)
        if task is None:
            raise HTTPException(404, detail=f"Task with ID '{task_id}' not found in Session '{s.id}'.")

        if task.status in {TaskStatus.RUNNING, TaskStatus.COMPLETED}:
            # prevent double runs, concurrent writes, etc.
            return ctx.redirect_back(fallback_url=f"/sessions/{session_id}")

        # 2. Re-check that Task is runnable
        if not task.status.is_runnable:
            msg = f"Task '{task.name}' is in status '{task.status}' and cannot be run."
            return ctx.redirect_back(fallback_url=f"/sessions/{session_id}", message=msg, message_type="warning")

        task.error = None
        s.mark_dirty()

    # background_tasks.add_task(rt.execute_task, session_id, task_id)
    rt.execute_task(session_id=session_id, task_id=task_id)

    return ctx.redirect_back(
        fallback_url=f"/sessions/{session_id}",
        message="Task execution started.",
        message_type="info",
    )


@router.post("/{task_id}/clone", response_class=RedirectResponse)
def clone_task_action(task_id: str, s: ActiveSession, ctx: PageContext = Depends()):
    """Create a deep copy of this Task with a new ID. Does not run the Task."""
    new_task = s.clone_task(task_id)
    return ctx.redirect(
        url=f"/sessions/{s.id}/tasks/{new_task.id}",
        message="Task cloned. Remember to update inputs before running.",
        message_type="info",
    )


@router.post("/{task_id}/delete", response_class=RedirectResponse)
def delete_task_action(task_id: str, s: ActiveSession, ctx: PageContext = Depends()):
    """Delete this Task. NOTE: This is irreversible and will orphan downstream Artifacts."""
    deleted = s.delete_task(task_id)
    if not deleted:
        return ctx.redirect_back(fallback_url=f"/sessions/{s.id}", message="Failed to delete task.", message_type="danger")
    return ctx.redirect(f"/sessions/{s.id}", message="Task deleted successfully.", message_type="success")


# --- AJAX routes ---

@router.post("/new/{task_key}", response_class=JSONResponse)
async def api_task_commit(
    task_key: str,
    payload: TaskPayload,
    s: ActiveSession,
):
    """
    Commits the current configuration to a Task in the Manifest
    """
    try:
        task_def = task_registry.get(task_key)
        if not task_def:
            raise ValueError(f"Task '{task_key}' not found in registry.")

        # 1. Extract Metadata BEFORE form_to_dict strips it out!
        existing_task_id = payload.inputs.get("task_id")
        task_name = payload.inputs.get("name") or task_def.name

        s.runtime.logger.debug(
            "api_task_commit [%s]: payload keys=%s", task_key, list(payload.inputs.keys())
        )

        # 2. Parse JSON payload
        raw_inputs = form_json_to_dict(payload.inputs, task_def.input_model)

        s.runtime.logger.debug(
            "api_task_commit [%s]: raw_inputs=%r", task_key, raw_inputs
        )

        # 3. Extract parent_artifact_ids dynamically
        parent_ids = []
        for field_name in task_def.input_model.model_fields.keys():
            _, extra = task_def.field_meta(field_name)
            # Check if this field is an ArtifactInput
            if extra.get("accepts_artifact", extra.get("is_artifact", False)):
                val = raw_inputs.get(field_name)
                if isinstance(val, list):
                    parent_ids.extend(val)  # Handle multi-file inputs
                elif val:
                    parent_ids.append(val)  # Handle single-file inputs

        # 4. Upsert logic for Task (allows editing existing tasks)
        task = None
        if existing_task_id:
            try:
                task = s.get_task(existing_task_id)
                if task is None:
                    raise ValueError(f"Task with ID '{existing_task_id}' not found in Session '{s.id}'.")
                if task.status in ["RUNNING", "COMPLETED"]:
                    task = None  # Cannot edit, save as new
            except ValueError:
                pass  # Task ID from form doesn't exist, fallback to new

        if task:
            task = s.update_task(
                task.id,
                inputs=raw_inputs,
                exec_config=payload.exec_config.model_dump(mode="json"),
                name=task_name,
                parent_artifact_ids=list(set(parent_ids)),
            )
        else:
            task = s.add_task(
                registry_key=task_key,
                inputs=raw_inputs,
                name=task_name,
                parent_artifact_ids=list(set(parent_ids)),  # Deduplicate parent IDs
                exec_config=payload.exec_config.model_dump(mode="json"),
            )

        s.runtime.logger.debug(
            "api_task_commit [%s]: task.id=%s  status=%s  error=%r  inputs=%r",
            task_key, task.id, task.status, task.error, task.inputs,
        )

        # 5. Return success and the redirect URL
        return JSONResponse(content={
            "status": "success",
            "redirect_url": f"/sessions/{s.id}/tasks/{task.id}",
        })

    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=400,
        )


# Because this is `def` and not `async def`, FastAPI runs it in a background thread!
@router.post("/new/{task_key}/preview", response_class=JSONResponse)
def api_task_preview(
    session_id: str,
    task_key: str,
    payload: TaskPayload,
    rt: RT,
    ctx: PageContext = Depends(),
):
    """
    AJAX endpoint for the Workbench. Runs the task in memory and returns JSON.
    """
    try:
        # 1. Execute the task, reading inputs from JSON payload
        form_inputs = form_json_to_dict(payload.inputs, task_registry[task_key].input_model)

        results = rt.preview_task(
            session_id,
            task_key,
            form_inputs,
            exec_config=payload.exec_config.model_dump(mode="json"),
        )
        primary_results = results.primary_data
        if not primary_results:
            raise ValueError(
                "Task execution completed, but no primary result. "
                "Does the Task plugin have `is_primary` set?"
            )
        primary_result = primary_results[0]

        # 2. Extract the metadata
        output_data = primary_result.get("data", {})
        view_mode = primary_result.get("view_mode", "raw")
        metadata = primary_result.get("metadata", {})
        adapter_name = primary_result.get("adapter_name", "unknown")

        render_context = {
            "request": ctx.request,
            "data": output_data,
            "view_state": payload.exec_config.adapter.view_state,
            "metadata": metadata,
            "adapter_name": adapter_name,
        }

        if view_mode == "table":
            keys = list(output_data[0].keys()) if output_data else []
            render_context["keys"] = keys

        elif view_mode == "svg":
            pass  # placeholder

        # 3. Render the HTML on the Server
        response = templates.TemplateResponse(
            request=ctx.request,
            name=f"partials/viewers/{view_mode}.html",
            context=render_context,
        )
        html_content = bytes(response.body).decode("utf-8")

        # 5. Send the compiled HTML to frontend for display
        return JSONResponse(content={
            "status": "success",
            "html": html_content,
        })

    except Exception as e:
        rt.logger.error("Preview API Error: %s", str(e))
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=400,
        )
