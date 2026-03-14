"""
LoRe API routes for managing Sessions.

Manages top level Session routes (CRUD)
"""

from pathlib import Path
import shutil
import tempfile

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse

from lore.core.utils import clean
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
def list_sessions(rt: RT, ctx: PageContext = Depends()):
    """Render the Sessions dashboard."""
    sessions = rt.list_sessions()  # List of sessions as dictionaries

    ctx.generate_breadcrumbs()
    return templates.TemplateResponse(
        "features/sessions/sessions.html",
        ctx.render(runtime=rt, sessions=sessions),
    )

@router.post("/new", response_class=RedirectResponse)
def create_session_action(rt: RT, ctx: PageContext = Depends()):
    """Make a new Session and redirect to its detail page."""
    session_obj = rt.create_session()
    return ctx.redirect(f"/sessions/{session_obj.id}")


@router.get("/{session_id}", response_class=HTMLResponse)
def show_session(s: ActiveSession, ctx: PageContext = Depends()):
    """Session dashboard."""
    tasks = s.list_tasks(reverse=True)
    artifacts = s.list_artifacts(reverse=True)
    push_task_targets = _build_push_task_targets(artifacts)

    ctx.generate_breadcrumbs({s.id: s.name})
    return templates.TemplateResponse(
        "/features/sessions/detail.html",
        ctx.render(
            session=s,
            tasks=tasks,
            artifacts=artifacts,
            push_task_targets=push_task_targets,
        )
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
        return ctx.redirect_back(fallback_url="/sessions", message="Session not found", message_type="warning")
    except RuntimeError as e:  # Locked/Running a background task
        return ctx.redirect_back(fallback_url="/sessions", message=str(e), message_type="error")


@router.get("/{session_id}/export")
async def api_export_session(
    session_id: str, 
    background_tasks: BackgroundTasks,
    rt: RT,
    ctx: PageContext = Depends()
):
    """
    Downloads the session as a zip file.
    TODO: This is rough. Add better logic, take file handling out of the router
    """
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
        return ctx.redirect_back("/sessions", message=f"Error exporting session: {str(e)}", message_type="error")


@router.post("/import")
async def api_import_session(
    rt: RT,
    ctx: PageContext = Depends(),
    file: UploadFile = File(...)
):
    """Uploads and ingests a zipped session archive."""
    if not file.filename:
        return ctx.redirect_back("/sessions", message="No file provided.", message_type="error")
    if not file.filename.endswith('.zip'):
        return ctx.redirect_back("/sessions", message="File must be a .zip archive.", message_type="error")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            new_session_id = rt.import_session(tmp_path)
            return ctx.redirect(f"/sessions/{new_session_id}", message="Session imported successfully!", message_type="success")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    except Exception as e:
        return ctx.redirect_back("/sessions", message=f"Error importing session: {str(e)}", message_type="error")

# --- AJAX for live updates ---

@router.get("/{session_id}/tasks/poll", response_class=HTMLResponse)
def poll_tasks(s: ReadOnlySession, ctx: PageContext = Depends()):
    """
    Live polling endpoint to check for Task status updates
    Returns ONLY the rendered HTML for the Task list (partial)
    """
    tasks = s.list_tasks(reverse=True)

    return templates.TemplateResponse(
        "/partials/task_list.html",
        ctx.render(
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
        "/partials/artifact_list.html",
        ctx.render(
            session=s,
            artifacts=artifacts,
            push_task_targets=push_task_targets,
        )
    )
