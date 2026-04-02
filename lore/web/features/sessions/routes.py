"""
LoRe API routes for managing Sessions.

Manages top level Session routes (CRUD)
"""

from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse

from lore.core.utils import auto_increment, clean, slugify
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

    ctx.generate_breadcrumbs({s.id: s.name})
    return templates.TemplateResponse(
        request=ctx.request,
        name="/features/sessions/detail.html",
        context=ctx.render(
            session=s,
            tasks=tasks,
            artifacts=artifacts,
            push_task_targets=push_task_targets,
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

@router.get("/{session_id}/tasks/poll", response_class=HTMLResponse)
def poll_tasks(s: ReadOnlySession, ctx: PageContext = Depends()):
    """
    Live polling endpoint to check for Task status updates
    Returns ONLY the rendered HTML for the Task list (partial)
    """
    tasks = s.list_tasks(reverse=True)

    return templates.TemplateResponse(
        request=ctx.request,
        name="/partials/task_list.html",
        context=ctx.render(
            session=s,
            tasks=tasks,
        )
    )


@router.get("/{session_id}/artifacts/poll", response_class=HTMLResponse)
def poll_artifacts(s: ReadOnlySession, ctx: PageContext = Depends()):
    """
    Live polling endpoint to check for Artifact updates
    Returns ONLY the rendered HTML for the Artifact list (partial)
    """
    artifacts = s.list_artifacts(reverse=True)
    push_task_targets = _build_push_task_targets(artifacts)

    return templates.TemplateResponse(
        request=ctx.request,
        name="/partials/artifact_list.html",
        context=ctx.render(
            session=s,
            artifacts=artifacts,
            push_task_targets=push_task_targets,
        )
    )

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
