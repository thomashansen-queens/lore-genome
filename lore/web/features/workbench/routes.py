"""
The Workbench: explore, interact with data, and designing Tasks
"""
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse

from lore.core.tasks import task_registry, ExecConfig
from lore.web.deps import RT, ActiveSession, templates, PageContext
from lore.web.utils.forms import format_inputs_for_ui, form_json_to_dict

router = APIRouter(prefix="/sessions/{session_id}/workbench", tags=["workbench"])

# --- Workbench shell ---

@router.get("/", response_class=HTMLResponse)
async def open_workbench_landing(
    s: ActiveSession,
    ctx: PageContext = Depends(),
):
    """
    Landing page for the Workbench. Displays all available Tasks 
    that can be launched in interactive mode.
    """
    # We pass the whole registry values so Jinja can loop through them
    available_tasks = list(task_registry.all.values())

    ctx.generate_breadcrumbs({s.id: s.name, "workbench": "Workbench"})
    return templates.TemplateResponse(
        "features/workbench/landing.html",
        ctx.render(
            session=s,
            tasks=available_tasks,
        )
    )


@router.get("/tasks/{registry_key}", response_class=HTMLResponse)
async def open_workbench_task(
    registry_key: str,
    s: ActiveSession,
    ctx: PageContext = Depends(),
    load_task: str | None = Query(None),
    source_artifact_ids: list[str] = Query(None),
):
    """
    Universal Workbench UI. Shows Task config and preview panel. Can be pre-
    filled with existing Task (?load_task=ID) and Artifacts (?source_artifact_ids=ID)
    """
    # 1. Load the Task Definition
    task_def = task_registry.get(registry_key)
    if not task_def:
        raise HTTPException(status_code=404, detail=f"Task '{registry_key}' not found in registry.")

    # 2. Edit if needed
    raw_inputs = {}
    prefill_name = task_def.name
    task_id = None

    if load_task:
        task = s.get_task(load_task)
        if task:
            raw_inputs = dict(task.inputs)
            prefill_name = task.name
            task_id = task.id

    # 3. Pushed Artifacts
    if source_artifact_ids:
        raw_inputs.update(s.map_artifacts_to_task_inputs(task_def, source_artifact_ids))

    # 4. Format for UI
    prefilled_inputs = format_inputs_for_ui(task_def, raw_inputs)
    artifact_candidates = s.get_artifact_candidates(task_def)

    # 5. Render the UI
    ctx.generate_breadcrumbs({
        s.id: s.name,
        "workbench": "Workbench",
    })
    return templates.TemplateResponse(
        "features/workbench/tool.html",
        ctx.render(
            session=s,
            task_def=task_def,
            prefill_name=prefill_name,
            prefilled_inputs=prefilled_inputs,
            artifact_candidates=artifact_candidates,
            edit_task_id=task_id,
            # We pass the POST url to the template so JS knows where to fetch
            preview_api_url=f"/sessions/{s.id}/workbench/tasks/{registry_key}/preview",
        )
    )

# --- AJAX for Task preview ---

class PreviewRequest(BaseModel):
    """Contract for the AJAX payload when requesting a Task preview"""
    inputs: dict
    exec_config: ExecConfig = Field(default_factory=ExecConfig)


# Because this is `def` and not `async def`, FastAPI runs it in a background thread!
@router.post("/tasks/{registry_key}/preview", response_class=JSONResponse)
def api_task_preview(
    session_id: str,
    registry_key: str,
    payload: PreviewRequest,
    rt: RT,
    ctx: PageContext = Depends(),
):
    """
    AJAX endpoint for the Workbench. Runs the task in memory and returns JSON.
    """
    try:
        # 1. Execute the task, reading inputs from JSON payload
        form_inputs = form_json_to_dict(payload.inputs, task_registry[registry_key].input_model)

        results = rt.preview_task(
            session_id,
            registry_key,
            form_inputs,
            exec_config=payload.exec_config.model_dump(mode="json"),
        )
        primary_result = results.primary_data
        if not primary_result:
            raise ValueError(
                "Task execution completed, but no primary result. "
                "Does the Task plugin have `is_primary` set?"
            )

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


@router.post("/tasks/{registry_key}/commit", response_class=JSONResponse)
async def api_task_commit(
    registry_key: str,
    payload: PreviewRequest,
    s: ActiveSession,
):
    """
    Commits the current Wokrbench configuration to a Task in the Manifest
    """
    try:
        task_def = task_registry.get(registry_key)
        if not task_def:
            raise ValueError(f"Task '{registry_key}' not found in registry.")

        # 1. Extract Metadata BEFORE form_to_dict strips it out!
        existing_task_id = payload.inputs.get("task_id")
        task_name = payload.inputs.get("name") or task_def.name

        # 2. Parse JSON payload
        raw_inputs = form_json_to_dict(payload.inputs, task_def.input_model)

        # 3. Extract parent_artifact_ids dynamically
        parent_ids = []
        for field_name, field_info in task_def.input_model.model_fields.items():
            # Check if this field is an ArtifactInput
            if "ArtifactInput" in str(field_info.annotation):
                val = raw_inputs.get(field_name)
                if isinstance(val, list):
                    parent_ids.extend(val)  # Handle multi-file inputs
                elif val:
                    parent_ids.append(val)  # Handle single-file inputs

        # 4. Upsert logic for Task
        task = None
        if existing_task_id:
            try:
                task = s.get_task(existing_task_id)
                if task is None:
                    raise ValueError(f"Task with ID '{existing_task_id}' not found in Session '{s.id}'.")
                if task.status in ["RUNNING", "COMPLETED"]:
                    pass  # Cannot edit, save as new
            except ValueError:
                pass # Task ID from form doesn't exist, fallback to new

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
                registry_key=registry_key,
                inputs=raw_inputs,
                name=task_name,
                parent_artifact_ids=list(set(parent_ids)),
                exec_config=payload.exec_config.model_dump(mode="json"),
            )

        # 3. Return success and the redirect URL
        return JSONResponse(content={
            "status": "success",
            "redirect_url": f"/sessions/{s.id}/tasks/{task.id}",
        })

    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=400,
        )
