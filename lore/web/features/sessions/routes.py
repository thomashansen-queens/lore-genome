"""
LoRe API routes for managing Sessions.

Manages top level Session routes (CRUD)
"""

import asyncio
from collections.abc import AsyncIterable
from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from lore.core.topology.diagram import generate_dag_diagram
from lore.core.topology.traversal import DAGValidationError, sort_tasks_topologically
from lore.core.tasks import task_registry
from lore.core.utils import auto_increment, clean, slugify
from lore.viz.graph import Direction
from lore.web.deps import RT, ActiveSession, ReadOnlySession, templates, PageContext

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _build_push_task_targets(artifacts: list) -> dict[str, list[tuple[str, object]]]:
    """Map artifact IDs to compatible task definitions for push menus."""
    return {
        artifact.id: artifact.get_push_tasks()
        for artifact in artifacts
    }

# --- Dashboards ---

@router.get("/", response_class=HTMLResponse)
def list_sessions(ctx: PageContext = Depends()):
    """Redirect to the dashboard page for now."""
    return ctx.redirect("/dashboard")


@router.post("/new", response_class=RedirectResponse)
def create_session_action(rt: RT, ctx: PageContext = Depends()):
    """Make a new Session and redirect to its detail page."""
    session_obj = rt.create_session()
    return ctx.redirect(f"/sessions/{session_obj.id}")


@router.post("/import")
async def api_import_session(
    rt: RT,
    ctx: PageContext = Depends(),
    file: UploadFile = File(...)
):
    """Uploads and ingests a zipped session archive."""
    if not file.filename:
        return ctx.redirect_back(
            fallback_url="/sessions",
            message="No file provided.",
            message_type="error",
        )
    if not file.filename.endswith('.zip'):
        return ctx.redirect_back(
            fallback_url="/sessions",
            message="File must be a .zip archive.",
            message_type="error",
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            new_session_id = rt.import_session(tmp_path)
            return ctx.redirect(
                url=f"/sessions/{new_session_id}",
                message="Session imported successfully!",
                message_type="success",
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    except Exception as e:
        return ctx.redirect_back(
            fallback_url="/sessions",
            message=f"Error importing session: {str(e)}",
            message_type="error",
        )


@router.get("/{session_id}", response_class=HTMLResponse)
def show_session(s: ReadOnlySession, ctx: PageContext = Depends()):
    """Session dashboard."""
    tasks = s.list_tasks(reverse=True)
    artifacts = s.list_artifacts(reverse=True)
    push_task_targets = _build_push_task_targets(artifacts)

    chronological_tasks = s.list_tasks()
    diagram_lr = generate_dag_diagram(chronological_tasks, task_registry, Direction.LR)

    ctx.generate_breadcrumbs({s.id: s.name})
    return templates.TemplateResponse(
        request=ctx.request,
        name="/features/sessions/detail.html",
        context=ctx.render(
            session=s,
            session_id=s.id,
            tasks=tasks,
            artifacts=artifacts,
            push_task_targets=push_task_targets,
            diagram_lr=diagram_lr,
        ),
    )


@router.get("/{session_id}/view", response_class=HTMLResponse)
def view_session(
    s: ReadOnlySession,
    view_type: str = Query("graph", alias="type"),
    sort_by: str = Query("topo", alias="sort"), 
    order: str = Query("asc"), 
    ctx: PageContext = Depends(),
):
    """HTMX endpoint to swap between Graph and Table views."""
    tasks = s.list_tasks()

    if view_type == "table":
        # 1. Calculate topological order for the "#" column
        topo_map = {}
        try:
            topo_sorted = sort_tasks_topologically(tasks)
            topo_map = {t.id: i+1 for i, t in enumerate(topo_sorted)}
        except DAGValidationError:
            topo_map = {t.id: "?" for t in tasks}

        # 2. Apply the requested sorting
        sort_strategies = {
            "name": lambda t: t.name or "",
            "type": lambda t: t.registry_key,
            "status": lambda t: t.status.value,  # Extract string from Enum
            "modified": lambda t: t.modified_at,
            "topo": lambda t: topo_map.get(t.id, 999) if isinstance(topo_map.get(t.id), int) else 999,
        }

        reverse = (order == "desc")
        if sort_by:
            sort_fn = sort_strategies.get(sort_by, sort_strategies["topo"])
            tasks.sort(key=sort_fn, reverse=reverse)

        return templates.TemplateResponse(
            request=ctx.request,
            name="features/sessions/fragments/task_list.html",
            context=ctx.render(
                session=s, 
                tasks=tasks,
                topo_map=topo_map,
                sort_by=sort_by,
                order=order,
            ),
        )
    else:
        chronological_tasks = s.list_tasks()
        diagram_lr = generate_dag_diagram(chronological_tasks, task_registry, Direction.LR)
        return templates.TemplateResponse(
            request=ctx.request,
            name="components/task_graph.html",
            context=ctx.render(
                session=s,
                diagram_lr=diagram_lr,
            ),
        )


@router.post("/{session_id}/clone", response_class=RedirectResponse)
def clone_session_action(rt: RT, session_id: str, ctx: PageContext = Depends()):
    """Clone a Session for reuse."""
    _ = rt.clone_session(session_id=session_id)
    return ctx.redirect_back(fallback_url="/sessions")


@router.post("/{session_id}/update", response_class=RedirectResponse)
async def update_session_action(
    s: ActiveSession,
    ctx: PageContext = Depends(),
):
    """
    Update an existing Session's basic metadata (e.g., name) via a web form.
    """
    form_data = await ctx.request.form()
    new_name = clean(form_data.get("name"))

    if new_name:
        s.name = new_name
    # Try to send the user back to where they came from
    return ctx.redirect_back(fallback_url=f"/sessions/{s.id}")


@router.post("/{session_id}/delete", response_class=RedirectResponse)
def delete_session(rt: RT, session_id: str, ctx: PageContext = Depends()):
    """Delete a Session and redirect to the Sessions list page."""
    try:
        rt.delete_session(session_id)
        msg = f"Session {session_id} deleted."
        msg_type = "success"
        return ctx.redirect("/sessions", message=msg, message_type=msg_type)

    except FileNotFoundError:
        return ctx.redirect_back(
            fallback_url="/sessions",
            message="Session not found",
            message_type="warning",
        )
    except RuntimeError as e:  # Locked/Running a background task
        return ctx.redirect_back(fallback_url="/sessions", message=str(e), message_type="error")


@router.post("/{session_id}/execute", response_class=RedirectResponse)
def execute_session_action(
    session_id: str,
    rt: RT,
    ctx: PageContext = Depends(),
):
    """Trigger the execution of all Tasks sequentially in a Session."""
    rt.execute_session(session_id)
    ctx.add_msg(
        "Execution Cascade started in background. Monitor progress on the Session page.",
        "success",
    )
    return ctx.redirect_back(fallback_url=f"/sessions/{session_id}")


@router.get("/{session_id}/export")
async def api_export_session(
    session_id: str, 
    background_tasks: BackgroundTasks,
    rt: RT,
    ctx: PageContext = Depends(),
):
    """
    Downloads the session as a zip file.
    TODO: This is rough. Add better logic, take file handling out of the router
    """
    tmp_dir = None
    try:
        # Temp dir to hold the zip file
        temp_dir = Path(tempfile.mkdtemp())

        # Ask the engine to build the zip
        zip_path = rt.export_session(session_id, temp_dir)

        # Clean up the temp file AFTER the user finishes downloading
        background_tasks.add_task(lambda p: shutil.rmtree(p.parent, ignore_errors=True), zip_path)

        return FileResponse(
            path=zip_path,
            filename=zip_path.name,
            media_type="application/zip",
        )
    except Exception as e:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

        return ctx.redirect_back(
            "/sessions",
            message=f"Error exporting session: {str(e)}",
            message_type="error",
        )

# --- AJAX for live updates ---

# @router.get("/{session_id}/tasks/poll", response_class=HTMLResponse)
# def poll_tasks(s: ReadOnlySession, ctx: PageContext = Depends()):
#     """
#     Live polling endpoint to check for Task status updates
#     Returns ONLY the rendered HTML for the Task list (partial)
#     """
#     tasks = s.list_tasks(reverse=True)

#     return templates.TemplateResponse(
#         request=ctx.request,
#         name="/features/sessions/fragments/task_list.html",
#         context=ctx.render(
#             session=s,
#             tasks=tasks,
#         )
#     )


# @router.get("/{session_id}/artifacts/poll", response_class=HTMLResponse)
# def poll_artifacts(s: ReadOnlySession, ctx: PageContext = Depends()):
#     """
#     Live polling endpoint to check for Artifact updates
#     Returns ONLY the rendered HTML for the Artifact list (partial)
#     """
#     artifacts = s.list_artifacts(reverse=True)
#     push_task_targets = _build_push_task_targets(artifacts)

#     return templates.TemplateResponse(
#         request=ctx.request,
#         name="/features/sessions/fragments/artifact_list.html",
#         context=ctx.render(
#             session=s,
#             artifacts=artifacts,
#             push_task_targets=push_task_targets,
#         )
#     )

# --- SSE for real-time updates ---

@router.get("/{session_id}/stream", response_class=EventSourceResponse)
async def stream_session(
    session_id: str,
    rt: RT,
    ctx: PageContext = Depends(),
) -> AsyncIterable[ServerSentEvent]:
    """
    SSE stream that pushes DOM updates for changes in Session state (e.g.
    Task status change, new Artifact emitted, etc.).
    """
    last_task_statuses = {}
    last_artifact_ids = set()

    # Set initial state trackers
    with rt.open_session(session_id, read_only=True) as s:
        initial_tasks = s.list_tasks(reverse=True)
        initial_artifacts = s.list_artifacts(reverse=True)

        last_task_statuses = {t.id: t.status for t in initial_tasks}
        last_artifact_ids = set(a.id for a in initial_artifacts)

    while True:
        if await ctx.request.is_disconnected():
            break

        # 1. Set live state (re-check the Session state on each poll)
        with rt.open_session(session_id, read_only=True) as s:
            current_tasks = s.list_tasks(reverse=True)
            current_artifacts = s.list_artifacts(reverse=True)

        current_task_statuses = {t.id: t.status for t in current_tasks}
        current_artifact_ids = set(a.id for a in current_artifacts)

        # 2. Check for changes
        tasks_changed = current_task_statuses != last_task_statuses
        artifacts_changed = current_artifact_ids != last_artifact_ids

        if tasks_changed or artifacts_changed:
            html_data = ""

            # 3. Render and concatenate out-of-band (OoB) HTML fragments
            if tasks_changed:
                # Diff the tasks to find which ones changed status
                changed_tasks = [
                    t for t in current_tasks 
                    if last_task_statuses.get(t.id) != t.status
                ]

                # topo_map mirrored from the /view endpoint
                topo_map = {}
                try:
                    topo_sorted = sort_tasks_topologically(current_tasks)
                    topo_map = {t.id: i+1 for i, t in enumerate(topo_sorted)}
                except DAGValidationError:
                    topo_map = {t.id: "?" for t in current_tasks}

                diagram = generate_dag_diagram(current_tasks, task_registry, Direction.LR)

                # Graph nodes
                html_data += templates.get_template("features/sessions/fragments/oob_updates.html").render(
                    session_id=session_id,
                    base_url=f"/sessions/{session_id}/tasks/",
                    changed_tasks=changed_tasks,
                    diagram=diagram,
                    topo_map=topo_map,
                )

            if artifacts_changed:
                # Diff the Artifacts to find which ones are new
                new_artifacts = [
                    a for a in current_artifacts 
                    if a.id not in last_artifact_ids
                ]
                push_targets = _build_push_task_targets(current_artifacts)
                html_data += templates.get_template("features/sessions/fragments/artifact_list.html").render(
                    session_id=session_id,
                    artifacts=new_artifacts,
                    push_task_targets=push_targets,
                    oob=True,
                )

            # 4. Push a single SSE message with all the HTML data
            # HTMX listens for 'message' events, which is the default for ServerSentEvent if
            # 'event' is not specified.
            # FastAPI normally yields JSON-encoded dicts, but for out-of-band updates,
            # we need to use raw_data to send plain HTML
            yield ServerSentEvent(raw_data=html_data)

            # 5. Update state trackers
            last_task_statuses = current_task_statuses
            last_artifact_ids = current_artifact_ids

        # Wait 1 second before checking again
        await asyncio.sleep(1)

# --- Session-Workflow endpoints ---

@router.post("/{session_id}/extract-workflow", response_class=RedirectResponse)
def extract_workflow_action(
    s: ReadOnlySession,
    rt: RT,
    workflow_name: str = Form(...),
    ctx: PageContext = Depends(),
):
    """
    Action to extract a Session to a reusable Workflow template.
    Redirects to the new Workflow's detail page.
    """
    workflows = [w["id"] for w in rt.workflows.list_workflows()]
    new_workflow_id = auto_increment(slugify(workflow_name), existing=workflows)

    workflow = rt.workflows.extract_from_session(s, new_workflow_id)
    rt.workflows.save_workflow(workflow)

    ctx.add_msg(f"Session '{s.name}' saved as workflow '{workflow.name}'.", "success")
    return ctx.redirect(f"/workflows/{workflow.id}")
