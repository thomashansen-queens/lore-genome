"""
Routes for managing individual artifacts within a Session.
"""

import json
from pydantic import BaseModel, Field
import pandas as pd

from fastapi import APIRouter, HTTPException, Depends, Response, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse

from lore.core.tasks import AdapterConfig
from lore.core.io import get_reader_for
from lore.core.utils import filter_and_sort
from lore.web.deps import ActiveSession, PageContext, ReadOnlySession, templates
from lore.web.utils.forms import get_form_str

_DEFAULT_EXPLORE_DISPLAY_LIMIT = 1000


router = APIRouter(prefix="/sessions/{session_id}/artifacts", tags=["artifacts"])

# --- Artifact views ---

@router.get("/", response_class=HTMLResponse)
async def view_artifact_manager(
    s: ReadOnlySession,
    ctx: PageContext = Depends(),
):
    """
    Renders the Artifact Manager for staging uploads and remote references.
    """
    ctx.generate_breadcrumbs({s.id: s.name, "artifacts": "Artifacts"})
    return templates.TemplateResponse(
        request=ctx.request,
        name="features/artifacts/manage.html",
        context=ctx.render(session=s, artifacts=s.artifacts),
    )


@router.post("/ingest", response_class=JSONResponse)
async def api_ingest_artifacts(
    s: ActiveSession,
    metadata: str = Form(...),
    files: list[UploadFile] = File(default_factory=list),
):
    """
    Receives multipart form containing JSON metadata and binary file data.
    TODO: Tidy this up, don't make the router do file IO
    """
    from pathlib import Path
    import shutil
    import tempfile
    items = json.loads(metadata)
    file_idx = 0

    for item in items:
        name = item.get("name")
        data_type = item.get("type", "unknown")

        if item.get("isRemote"):
            uri = item.get("uri")
            s.register_artifact(
                source=uri,
                name=name,
                data_type=data_type,
            )
            continue
        else:
            upload = files[file_idx]
            file_idx += 1

            # materialize the file to a temp file for ingestion
            original_ext = Path(upload.filename).suffix if upload.filename else ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"{original_ext}") as tmp_out:
                shutil.copyfileobj(upload.file, tmp_out)
                tmp_path = Path(tmp_out.name)
            
            try:
                s.register_artifact(
                    source=tmp_path,
                    name=name,
                    data_type=data_type,
                )
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

    return JSONResponse({"status": "success", "message": "Artifacts ingested successfully"})


@router.get("/{artifact_id}", response_class=RedirectResponse)
def explore_artifact_default(
    artifact_id: str,
    s: ReadOnlySession,
    ctx: PageContext = Depends(),
):
    """Redirects to the default view for an Artifact."""
    artifact = s.get_artifact(artifact_id)
    if not artifact:
        return ctx.redirect_back(
            f"/sessions/{s.id}",
            message="Artifact not found",
            message_type="error",
        )
    adapters = artifact.get_adapters()
    if adapters:
        adapter = adapters[0]
        return ctx.redirect(
            f"/sessions/{s.id}/artifacts/{artifact_id}/explore?adapter_key={adapter.name}"
        )
    return ctx.redirect(f"/sessions/{s.id}/artifacts/{artifact_id}/explore")


@router.get("/{artifact_id}/details", response_class=HTMLResponse)
def view_artifact_details(
    artifact_id: str,
    s: ReadOnlySession,
    ctx: PageContext = Depends(),
):
    """
    Presents details about an Artifact's lineage and metadata.
    """
    artifact = s.get_artifact(artifact_id)
    if not artifact:
        return ctx.redirect_back(f"/sessions/{s.id}", message="Artifact not found", message_type="error")

    adapters = artifact.get_adapters()
    push_tasks = artifact.get_push_tasks()
    parent_task = s.get_task(artifact.created_by_task_id) if artifact.created_by_task_id else None
    parent_artifacts = [s.get_artifact(artifact_id) for artifact_id in artifact.parent_artifact_ids]

    path = s.get_artifact_path(artifact_id)
    is_truncated = False

    try:
        reader = get_reader_for(path)
        preview = reader.read_text_chunk(max_chars=5000)
        file_size = reader.get_base_metadata()["file_size_bytes"]
        is_truncated = file_size > len(preview.encode("utf-8"))
    except (ValueError, NotImplementedError):
        preview = "(Preview not available for this file type.)"
        file_size = path.stat().st_size if path.exists() else 0

    ctx.generate_breadcrumbs({s.id: s.name, artifact_id: artifact.ui_name})
    return templates.TemplateResponse(
        request=ctx.request,
        name="features/artifacts/detail.html",
        context=ctx.render(
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
    s: ReadOnlySession,
    adapter_key: str | None = None,
    ctx: PageContext = Depends(),
):
    """
    View the adapted data for an Artifact
    """
    artifact = s.get_artifact(artifact_id)
    if not artifact:
        return ctx.redirect_back(f"/sessions/{s.id}/artifacts/{artifact_id}", message="Artifact not found", message_type="error")

    # 1. Allow user to specify an adapter if multiple are available
    adapters = artifact.get_adapters()
    if adapter_key:
        adapter = next((a for a in adapters if a.name == adapter_key), None)
    else:
        adapter = adapters[0] if adapters else None

    if not adapter:
        return ctx.redirect_back(
            f"/sessions/{s.id}/artifacts/{artifact_id}",
            message="No adapter available for this Artifact",
            message_type="warning",
    )

    # 2. Data loading
    path = s.get_artifact_path(artifact_id)
    reader = get_reader_for(path)
    data, io_metadata = reader.preview(100)
    preview_result = adapter.preview(data, io_metadata)

    ctx.generate_breadcrumbs({
        s.id: s.name,
        artifact_id: artifact.ui_name
    })
    return templates.TemplateResponse(
        request=ctx.request,
        name="features/artifacts/explore.html",
        context=ctx.render(
            session=s,
            artifact=artifact,
            adapter=adapter,
            available_adapters=adapters,
            view_mode=adapter.view_mode,
            data=preview_result.data,
            keys=preview_result.metadata.get("columns", []),
            metadata=preview_result.metadata,
        )
    )

# --- Explore AJAX ---

def _get_explore_df(s: ReadOnlySession, artifact, adapter) -> pd.DataFrame:
    """
    Loads, adapts, and (importantly) caches the full DataFrame for an artifact.
    Uses a private function to leverage LoRe's runtime caching system.
    """
    path = s.get_artifact_path(artifact.id)
    reader = get_reader_for(path)

    def _compute():
        records = adapter.adapt(reader.read_full())
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.apply(lambda col: pd.to_numeric(col, errors="coerce")
                .fillna(col)
                .infer_objects()
                if col.dtype == object else col)
            df = df.convert_dtypes()
        return df

    return s.runtime.cache.get_or_compute(
        session_id=s.id,
        prefix="explore_df",
        compute_fn=_compute,
        cache_kwargs={
            "artifact_id": artifact.id,
            "adapter": adapter.name,
            "created_at": artifact.created_at.isoformat(),
        },
    )


class ExploreDataRequest(BaseModel):
    query: str = ""
    regex: bool = False
    adapter_key: str | None = None
    adapter_config: AdapterConfig = Field(default_factory=AdapterConfig)


@router.post("/{artifact_id}/explore/data", response_class=JSONResponse)
def api_explore_data(
    artifact_id: str,
    payload: ExploreDataRequest,
    s: ReadOnlySession,
    ctx: PageContext = Depends(),
):
    """
    AJAX endpoint for the Artifact Explorer. Applies sort + pandas query filter
    on the full dataset and returns a rendered HTML fragment.
    """
    try:
        artifact = s.get_artifact(artifact_id)
        if not artifact:
            return JSONResponse({"status": "error", "message": "Artifact not found"}, status_code=400)

        adapters = artifact.get_adapters()
        if payload.adapter_key:
            adapter = next((a for a in adapters if a.name == payload.adapter_key), None)
        else:
            adapter = adapters[0] if adapters else None

        if not adapter:
            return JSONResponse({"status": "error", "message": "No adapter available"}, status_code=400)

        path = s.get_artifact_path(artifact_id)
        reader = get_reader_for(path)

        # 1. Load and adapt the full dataset
        df = _get_explore_df(s, artifact, adapter)
        total_rows = len(df)

        # 2. 3-tier filtering
        view_state = payload.adapter_config.view_state
        try:
            df = filter_and_sort(
                df,
                query=payload.query,
                regex=payload.regex,
                sort_by=view_state.get("sort_by"),
                sort_asc=view_state.get("sort_asc", True),
            )
        except ValueError as e:
            return JSONResponse(
                content={"status": "error", "message": f"Invalid filter query: {str(e)}"},
                status_code=400,
            )

        # 3. Data enhancements
        # UI data bars: calculate maximums
        numeric_df = df.select_dtypes(include=["number"])
        numeric_maxes = numeric_df.max().to_dict() if not numeric_df.empty else {}

        filtered_row_count = len(df)

        # 4. Truncate for display
        display_limit = s.runtime.settings.explore_display_limit or _DEFAULT_EXPLORE_DISPLAY_LIMIT
        is_truncated = filtered_row_count > display_limit
        df_display = df.head(display_limit)

        display_records = df_display.to_dict(orient="records")
        columns = list(df.columns) if not df.empty else []

        metadata = {
            "columns": columns,
            "total_rows": total_rows,
            "filtered_rows": filtered_row_count,
            "is_truncated": is_truncated,
            "numeric_maxes": numeric_maxes,
            **reader.get_base_metadata(),
        }

        # 5. Render HTML fragment
        response = templates.TemplateResponse(
            request=ctx.request,
            name="components/viewers/table.html",
            context={
                "request": ctx.request,
                "data": display_records,
                "keys": columns,
                "metadata": metadata,
                "adapter_name": adapter.name,
                "view_state": payload.adapter_config.view_state,
            },
        )
        html = bytes(response.body).decode("utf-8")

        return JSONResponse({
            "status": "success",
            "html": html,
            "row_count": filtered_row_count,
            "total_rows": total_rows,
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@router.post("/{artifact_id}/explore/export")
def api_explore_export(
    artifact_id: str,
    payload: ExploreDataRequest,
    s: ReadOnlySession,
):
    """Downloads the currently filtered and sorted dataset as a CSV."""
    artifact = s.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(404, "Artifact not found")

    adapters = artifact.get_adapters()
    if payload.adapter_key:
        adapter = next((a for a in adapters if a.name == payload.adapter_key), None)
    else:
        adapter = adapters[0] if adapters else None

    if not adapter:
        raise HTTPException(404, "No adapter available for this Artifact")

    view_state = payload.adapter_config.view_state

    # 1. Fetch (probably from cache)
    df = _get_explore_df(s, artifact, adapter)

    # 2. Filter and sort
    try:
        df = filter_and_sort(
            df,
            query=payload.query,
            regex=payload.regex,
            sort_by=view_state.get("sort_by"),
            sort_asc=view_state.get("sort_asc", True),
        )
    except ValueError as e:
        raise HTTPException(400, f"Invalid filter query: {str(e)}") from e

    # 3. Return CSV Response
    csv_data = df.to_csv(index=False)
    filename = f"{artifact.name or 'export'}_filtered.csv"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

# --- Actions ---

@router.post("/{artifact_id}/rename", response_class=RedirectResponse)
async def rename_artifact_action(
    artifact_id: str,
    s: ActiveSession,
    ctx: PageContext = Depends()
):
    """Dedicated endpoint for in-place renaming of Artifacts."""
    form_data = await ctx.request.form()
    new_name = get_form_str(form_data, "name")

    if new_name:
        s.rename_artifact(artifact_id, new_name)
    return ctx.redirect_back(fallback_url=f"/sessions/{s.id}/artifacts/{artifact_id}")


@router.post("/{artifact_id}/update", response_class=RedirectResponse)
async def update_artifact_action(
    artifact_id: str,
    s: ActiveSession,
    ctx: PageContext = Depends(),
):
    """
    Update an existing Artifacts's basic metadata (like name) via a web form.
    """
    form_data = await ctx.request.form()

    # Metadata (updates manifest, no FS changes)
    # FUTURE: This is where various metadata fields can change
    # s.update_artifact_metadata(artifact_id, {"notes": new_notes})

    return ctx.redirect_back(fallback_url=f"/sessions/{s.id}")


@router.get("/{artifact_id}/download")
def download_artifact(artifact_id: str, s: ReadOnlySession):
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
