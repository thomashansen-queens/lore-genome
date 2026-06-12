"""
Routes for managing individual Tasks within a Session.
"""
import asyncio
from collections.abc import AsyncIterable
import html
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from fastapi import APIRouter, HTTPException, Depends, Query, Response
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from lore.core.adapters import AdapterPreview
from lore.core.sessions import resolve_task_outputs
from lore.core.tasks import TaskConfig, TaskStatus, task_registry
from lore.core.topology.matcher import extract_lineage, infer_bindings_from_raw
from lore.web.deps import RT, ActiveSession, ReadOnlySession, templates, PageContext
from lore.web.utils.configure_task import build_task_configure_context, build_widget_context
from lore.web.utils.forms import get_form_str, form_json_to_dict

router = APIRouter(prefix="/sessions/{session_id}/tasks", tags=["tasks"])


@router.get("/", response_class=RedirectResponse)
def redirect_to_session(session_id: str, ctx: PageContext = Depends()):
    """Redirect /sessions/{session_id}/tasks to the Session dashboard."""
    return ctx.redirect(f"/sessions/{session_id}")


@router.get("/catalogue", response_class=HTMLResponse)
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


@router.get("/configure", response_class=RedirectResponse)
def redirect_configure(session_id: str, ctx: PageContext = Depends()):
    """Redirect bare /configure to the Task Catalogue."""
    return ctx.redirect(f"/sessions/{session_id}/tasks/catalogue")


@router.get("/configure/{task_key}", response_class=HTMLResponse)
async def configure_task(
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

    ui_context = build_task_configure_context(
        container=s,
        task_key=task_key,
        task_id=load_task,
        source_artifact_ids=source_artifact_ids,    
    )

    ctx.generate_breadcrumbs({s.id: s.name, "configure": "Configure Task"})
    return templates.TemplateResponse(
        request=ctx.request,
        name="features/tasks/configure.html",
        context=ctx.render(
            session=s,
            context_type="session",
            preview_api_url=f"/sessions/{s.id}/tasks/configure/{task_key}/preview",
            commit_api_url=f"/sessions/{s.id}/tasks/configure/{task_key}",
            **ui_context,  # Unpacks into expected vars for Jinja
        )
    )


@router.get("/{task_id}", response_class=HTMLResponse)
def view_session_task(
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
        name="features/tasks/detail.html",
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

    target_url = f"/sessions/{s.id}/tasks/configure/{task.registry_key}?load_task={task.id}"

    # Forward any source artifacts if they exist
    if source_artifact_ids:
        # FastAPI handles list queries as multiple params (e.g., ?source_artifact_ids=1&source_artifact_ids=2)
        query_string = "&".join([f"source_artifact_ids={aid}" for aid in source_artifact_ids])
        target_url += f"&{query_string}"

    return ctx.redirect(target_url)

# --- HTMX ---

@router.get("/configure/{task_key}/input-widget", response_class=HTMLResponse)
def get_input_widget(
    task_key: str,
    s: ReadOnlySession,
    field_name: str = Query(...),
    task_id: str | None = Query(None),
    ctx: PageContext = Depends(),
):
    """
    HTMX endpoint: Hot-swaps Task input slots
    """
    new_mode = ctx.request.query_params.get(f"__mode__{field_name}", "literal")
    widget_context = build_widget_context(s, task_key, field_name, new_mode, task_id)
    f_name = widget_context.pop("field_name")
    f_info = widget_context.pop("field_info")

    return templates.TemplateResponse(
        request=ctx.request,
        name="components/task_input.html",
        context=ctx.render(
            context_type="session",
            edit_task_id=task_id,
            field_name=f_name,
            field_info=f_info,
            ui_state=widget_context,
        )
    )

# --- Actions ---

class TaskPayload(BaseModel):
    """Contract for the AJAX payload when previewing or committing a Task"""
    name: str | None = None
    task_id: str | None = None  # for edits; ignored on new Tasks
    task_config: TaskConfig = Field(default_factory=TaskConfig)
    run_immediately: str | bool | None = None
    model_config = ConfigDict(extra="allow")  # Allow arbitrary keys for adapter config, etc.

    @property
    def inputs(self) -> dict[str, Any]:
        """Reconstructs inputs dictionary from extra fields in the payload"""
        return self.model_extra or {}


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

    # 3. Send to Runtime to execute
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


# --- HTMX routes ---

@router.post("/configure/{task_key}")
def api_task_commit(
    task_key: str,
    payload: TaskPayload,
    session_id: str,
    rt: RT,
):
    """
    Commits the current configuration to a Task in the Manifest.
    Responds to HTMX.
    """
    try:
        task_def = task_registry.get(task_key)
        if not task_def:
            raise ValueError(f"Task '{task_key}' not found in registry.")

        # 1. Extract Metadata
        existing_task_id = payload.task_id
        task_name = payload.name or task_def.name

        rt.logger.debug(
            "api_task_commit [%s]: payload keys=%s", task_key, list(payload.inputs.keys())
        )

        # 2. Parse payload (pulled from model_extra)
        raw_inputs = form_json_to_dict(payload.inputs, task_def.input_model)

        ui_modes = {
            k.replace("__mode__", ""): v
            for k, v, in payload.inputs.items()
            if k.startswith("__mode__")
        }

        rt.logger.debug(
            "api_task_commit [%s]: raw_inputs=%r", task_key, raw_inputs
        )

        with rt.open_session(session_id, read_only=False) as s:  # Re-open session to ensure we have a lock for writing
            # 3. Topology and lineage
            input_bindings = infer_bindings_from_raw(raw_inputs, s, ui_modes)
            parent_ids = extract_lineage(input_bindings, task_def)

            # 5. Upsert logic for Task (allows editing existing tasks)
            task_id = None
            task = None

            # New tasks will not have an id yet
            if existing_task_id:
                task = s.get_task(existing_task_id)
                if task is None:
                    raise ValueError(f"Task with ID '{existing_task_id}' not found in Session '{s.id}'.")
                if task.status == TaskStatus.RUNNING:
                    task = None  # Cannot edit, save as new

            if task:
                task = s.update_task(
                    task.id,
                    inputs=input_bindings,
                    exec_config=payload.task_config.model_dump(mode="json"),
                    name=task_name,
                    parent_artifact_ids=list(set(parent_ids)),
                )
            else:
                task = s.add_task(
                    registry_key=task_key,
                    inputs=input_bindings,
                    name=task_name,
                    parent_artifact_ids=list(set(parent_ids)),  # Deduplicate parent IDs
                    exec_config=payload.task_config.model_dump(mode="json"),
                )
            task_id = task.id

        if str(payload.run_immediately).lower() == "true":
            rt.execute_task(session_id=session_id, task_id=task_id)

        # 6. Return success and the redirect URL
        return Response(headers={"HX-Redirect": f"/sessions/{session_id}/tasks/{task_id}"})

    except Exception as e:
        rt.logger.error("Commit API Error: %s", str(e), exc_info=True)
        # Return 200 OK so HTMX accepts the payload, but the error string
        err_html = f"""
        <div class="card warning" style="margin-bottom: var(--spacing-md);">
            <strong>Validation Error:</strong> {str(e)}
        </div>
        """
        return HTMLResponse(content=err_html)


# Because this is `def` and not `async def`, FastAPI runs it in a background thread!
@router.post("/configure/{task_key}/preview", response_class=HTMLResponse)
def api_task_preview(
    session_id: str,
    task_key: str,
    payload: TaskPayload,
    rt: RT,
    ctx: PageContext = Depends(),
):
    """
    AJAX endpoint for the Task preview. Runs the task in memory and returns
    rendered HTML of the results.
    """
    from lore.core.adapters import adapter_registry

    try:
        task_def = task_registry.get(task_key)
        if not task_def:
            raise ValueError(f"Task '{task_key}' not found in registry.")

        # 1. Execute the task, reading inputs from JSON payload
        form_inputs = form_json_to_dict(payload.inputs, task_registry[task_key].input_model)
        ui_modes = {
            k.replace("__mode__", ""): v
            for k, v in payload.inputs.items()
            if k.startswith("__mode__")
        }

        # 2. Topology Inference (resolve JIT bindings for preview)
        with rt.open_session(session_id, read_only=True) as s:
            inferred_bindings = infer_bindings_from_raw(form_inputs, s, ui_modes)

        # 3. Ephemeral preview task
        preview_payload = rt.preview_task(
            session_id=session_id,
            task_key=task_key,
            raw_inputs=inferred_bindings,
            exec_config=payload.task_config.model_dump(mode="json"),
        )

        # 4. Adapt data for UI presentation (e.g. table vs. json)
        adapted_previews = {}
        for key, output in preview_payload.output_previews.items():
            # A. Choose which adapter to use (explicit config > type-based)
            adapter_key = payload.task_config.adapter.options.get(f"{key}_adapter")
            adapter = adapter_registry.get(adapter_key) if adapter_key else None

            if not adapter:
                ext = output.data.suffix if isinstance(output.data, Path) else "*"
                adapters = adapter_registry.get_for_type(data_type=output.data_type, extension=ext)
                adapter = adapters[0] if adapters else None

            # B. Mock response if no adapter found
            if not adapter:
                adapted_previews[key] = AdapterPreview(
                    data=None,
                    view_mode="raw",
                    adapter_name="Error",
                    metadata={"error": "No adapter found for type '{output.data_type}'."},
                )
                continue
    
            # C. Adapt the raw output data for frontend preview
            adapted_previews[key] = adapter.preview(
                raw_data=output.data,
                io_metadata={},
                config=payload.task_config.adapter.model_dump(),
            )

        # 5. Render the preview with Jinja
        return templates.TemplateResponse(
            request=ctx.request,
            name=f"features/tasks/fragments/preview.html",
            context=ctx.render(
                payload=preview_payload,
                adapted_previews=adapted_previews,
            )
        )

    except Exception as e:
        rt.logger.error("Preview API Error: %s", str(e), exc_info=True)
        err_html = f"""
        <div class="card" style="background: var(--danger-bg); border-color: var(--danger);">
            <h4 style="color: var(--danger); margin-top: 0;">Preview Failed</h4>
            <p class="small">{str(e)}</p>
        </div>
        """
        return HTMLResponse(content=err_html)


# --- SSE for real-time updates ---

def _drain_log(log_path: Path | None, cursor: int, flush: bool = False) -> tuple[str | None, int]:
    """
    Helper that reads log content written since the last cursor position.
    Returns (escaped_text_or_None, new_cursor).
    """
    if log_path is None or not log_path.exists():
        return None, cursor
    size = log_path.stat().st_size

    # If the log was truncated, reset, or rotated
    if size < cursor:
        cursor = 0
    if size <= cursor:
        return None, cursor

    with open(log_path, "rb") as f:
        f.seek(cursor)
        chunk = f.read()
    if not flush:
        nl = chunk.rfind(b"\n")  # only read up to the last full line
        if nl == -1:
            return None, cursor
        chunk = chunk[:nl+1]

    if not chunk:
        return None, cursor
    return html.escape(chunk.decode("utf-8", errors="replace")), cursor + len(chunk)


@router.get("/{task_id}/stream", response_class=EventSourceResponse)
async def stream_task_updates(
    session_id: str,
    task_id: str,
    rt: RT,
    ctx: PageContext = Depends(),
) -> AsyncIterable[ServerSentEvent]:
    """
    SSE stream that pushes DOM updates for changes in Task state (e.g.
    Task status change, new Artifact emitted, etc.).
    TODO: Add heartbeat to detect disconnects/issues with long-running streams
    """
    # 1. Set initial state (prevents tick 0 update)
    with rt.open_session(session_id, read_only=True) as s:
        task_prev = s.get_task(task_id)
        if task_prev is None:
            raise HTTPException(404, detail=f"Task with ID '{task_id}' not found in Session '{s.id}'.")
        log_path = s.get_task_log_path(task_id)

    # No new data expected
    if task_prev.status.is_terminal:
        yield ServerSentEvent(raw_data="", event="close")
        return

    # 2. If the task exists and could receive updates, open a stream
    log_cursor = log_path.stat().st_size if (log_path and log_path.exists()) else 0

    while True:
        if await ctx.request.is_disconnected():
            break

        # 3. Set live state (re-check on each poll)
        with rt.open_session(session_id, read_only=True) as s:
            task_now = s.get_task(task_id)

        if task_now is None or task_prev is None:
            error_html = f'<div id="task-status-row" hx-swap-oob="true"><div class="badge danger">Vanished</div></div>'
            yield ServerSentEvent(raw_data=error_html, event="task_update")
            break
        terminal = task_now.status.is_terminal

        # 4. Diff the Task state (status change, new outputs, log updates, error)
        status_changed = task_now.status != task_prev.status
        outputs_changed = task_now.outputs != task_prev.outputs
        error_changed = task_now.error != task_prev.error
        new_logs, log_cursor = _drain_log(log_path, log_cursor, flush=terminal)

        # 5. Unpack outputs
        resolved_outputs = {}
        if outputs_changed:
            with rt.open_session(session_id, read_only=True) as s:
                resolved_outputs = resolve_task_outputs(s, task_id)

        # 6. Send data diffs to Jinja
        html_data = ""
        if status_changed or outputs_changed or error_changed or new_logs:
            html_data = templates.get_template(
                "features/tasks/fragments/oob_updates.html"
            ).render(
                task=task_now,
                session_id=session_id,
                status_changed=status_changed,
                outputs=resolved_outputs,
                outputs_changed=outputs_changed,
                task_log=new_logs,
                error_changed=error_changed,
            )
        if html_data.strip():
            yield ServerSentEvent(raw_data=html_data, event="task_update")

        if terminal:
            yield ServerSentEvent(raw_data="", event="close")
            return

        # 7. Wait 1 second before checking again
        task_prev = task_now
        await asyncio.sleep(1)
