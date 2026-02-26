"""
Routes for managing individual artifacts within a Session.
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse

from lore.core.adapters import adapter_registry
from lore.web.deps import ActiveSession, PageContext, build_breadcrumbs, templates
from lore.web.utils import get_form_str

router = APIRouter(prefix="/sessions/{session_id}/artifacts", tags=["artifacts"])


@router.get("/{artifact_id}", response_class=HTMLResponse)
async def view_artifact_overview(
    artifact_id: str,
    s: ActiveSession,
    ctx: PageContext = Depends(),
):
    """
    The home page for an Artifact.
    """
    try:
        artifact = s.get_artifact(artifact_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e

    adapters = artifact.get_adapters()
    push_tasks = artifact.get_push_tasks()
    parent_task = s.get_task(artifact.created_by_task_id) if artifact.created_by_task_id else None
    parent_artifacts = [s.get_artifact(artifact_id) for artifact_id in artifact.parent_artifact_ids]
    preview, is_truncated = s.preview_artifact(artifact_id, mode="bytes")

    ctx.breadcrumbs = build_breadcrumbs(s.id, s.name, artifact.ui_name)
    return templates.TemplateResponse(
        "features/artifacts/detail.html",
        ctx.render(
            session=s,
            artifact=artifact,
            parent_task=parent_task,
            parent_artifacts=parent_artifacts,
            adapters=adapters,
            push_tasks=push_tasks,
            text_preview=preview,
            is_truncated=is_truncated,
        )
    )


@router.get("/{artifact_id}/explore", response_class=HTMLResponse)
async def view_artifact_explore(
    artifact_id: str,
    s: ActiveSession,
    adapter_key: str | None = None,
    ctx: PageContext = Depends(),
):
    """
    View the adapted data for an Artifact
    """
    try:
        artifact = s.get_artifact(artifact_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e

    if adapter_key:
        adapter = adapter_registry[adapter_key]
    else:
        matches = adapter_registry.get_adapters(artifact)
        adapter = matches[0] if matches else None

    if not adapter:
        return ctx.redirect(f"/sessions/{s.id}/artifacts/{artifact_id}", message="No adapter available for this Artifact", message_type="warning")

    raw_data, is_truncated = s.preview_artifact(artifact_id, mode="adapter", limit_lines = 500)
    table_data = adapter.adapt(raw_data)
    keys = adapter.get_keys()
    if not keys and table_data:
        keys = list(table_data[0].keys())  # fallback to all keys in first record

    ctx.breadcrumbs = build_breadcrumbs(
        s.id, s.name,
        extra=[(artifact.ui_name, f"/sessions/{s.id}/artifacts/{artifact_id}")],
        final_item="Explore",
    )
    return templates.TemplateResponse(
        "features/artifacts/explore.html",
        ctx.render(
            session=s,
            artifact=artifact,
            adapter=adapter,
            table_data=table_data,
            keys=keys,
            is_truncated=is_truncated,
        )
    )


@router.post('/{artifact_id}/update', response_class=RedirectResponse)
async def update_artifact_action(
    artifact_id: str,
    s: ActiveSession,
    ctx: PageContext = Depends(),
):
    """
    Update an existing Artifacts's basic metadata (like name) via a web form.
    """
    form_data = await ctx.request.form()
    # 1. Handle renaming (changes file paths)
    new_name = get_form_str(form_data, "name")
    if new_name:
        try:
            s.rename_artifact(artifact_id, new_name)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    # 2. Metadata (updates manifest, no FS changes)
    # FUTURE: This is where various metadata fields can change
    # s.update_artifact_metadata(artifact_id, {"notes": new_notes})

    return ctx.redirect_back(fallback_url=f"/sessions/{s.id}")


@router.get("/{artifact_id}/download")
def download_artifact(artifact_id: str, s: ActiveSession):
    """Triggers a browser download"""
    try:
        path = s.get_artifact_path(artifact_id)
        return FileResponse(path=path, filename=path.name)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/{artifact_id}/delete", response_class=RedirectResponse)
def delete_artifact_action(
    artifact_id: str,
    s: ActiveSession,
    ctx: PageContext = Depends(),
):
    """Delete and Artifact from the Manifest and Disk"""
    s.delete_artifact(artifact_id)
    return ctx.redirect(
        f"/sessions/{s.id}",
        message="Artifact deleted",
        message_type="success",
    )
