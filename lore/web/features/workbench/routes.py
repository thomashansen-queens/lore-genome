"""
The Workbench: explore, interact with data, and designing Tasks
"""

import pandas as pd

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import HTMLResponse

from lore.core.adapters import adapter_registry
from lore.web.deps import ActiveSession, templates, build_breadcrumbs, PageContext
from lore.core.utils import normalize_query

router = APIRouter(prefix="/sessions/{session_id}/workbench", tags=["workbench"])

# --- Workbench shell ---

@router.get("/{artifact_id}/{tool_name}", response_class=HTMLResponse)
def open_workbench_tool(
    s: ActiveSession,
    artifact_id: str,
    tool_name: str,
    ctx: PageContext = Depends(),
):
    """
    Opens the workbench layout with a specific Tool loaded (filter, sample, etc.)
    """
    artifact = s.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail=f"Artifact not found ID: {artifact_id}")

    adapters = adapter_registry.get_adapters(artifact)
    adapter = adapters[0] if adapters else None
    suggested_keys = adapter.get_keys() if adapter else []

    # Map tool names to their template components
    tool_templates = {
        "filter": "workbench/tools/filter.html",
        "sample": "workbench/tools/sample.html",
    }

    if tool_name not in tool_templates:
        raise HTTPException(404, f"Tool not found: {tool_name}")

    ctx.breadcrumbs = build_breadcrumbs(s.id, s.name, f"Workbench: {tool_name.title()}")

    return templates.TemplateResponse(
        "workbench/layout.html",
        ctx.render(
            session=s,
            artifact=artifact,
            tool_name=tool_name,
            tool_template=tool_templates[tool_name],
            suggested_keys=suggested_keys,
            initial_preview_url=f"/sessions/{s.id}/workbench/{artifact.id}/preview?partial=true",
        )
    )

# --- Workbench engine ---

@router.get("/{artifact_id}/preview", response_class=HTMLResponse)
def preview_workbench_data(
    s: ActiveSession,
    artifact_id: str,
    ctx: PageContext = Depends(),
    query: str = Query(None),
    partial: bool = False,
    report_type: str = Query("table", description="table | summary"),
    group_by: str = Query(None),
):
    """
    Heavy-lifting data engine. Returns HTML tables (filtered/sampled)
    """
    # 1. Load and adapt
    artifact = s.get_artifact(artifact_id)
    raw_data = s.load_artifact_data(artifact_id)
    adapters = adapter_registry.get_adapters(artifact)
    adapter = adapters[0] if adapters else None

    if adapter:
        data_points = adapter.adapt(raw_data)
    elif isinstance(raw_data, list):
        data_points = raw_data
    else:
        data_points = [raw_data]

    # 2. Data cleaning and normalization
    df = pd.DataFrame(data_points)

    df = df.replace([r'^\s*$', 'na', 'N/A', 'n/a', 'missing'], pd.NA, regex=True)
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce", format="IOS8601")

    df = df.convert_dtypes()
    original_count = len(df)
    note = f"Source: {len(df)} records"

    # 3. Apply filters
    clean_query = normalize_query(query)
    if clean_query:
        try:
            df = df.query(clean_query)
            note = f"Filtered: {len(df)} / {original_count} records"
        except Exception as e:
            note = f"⚠ Query Error: {str(e)}"

    # 4. Render based on report type
    if report_type == "sample_summary" and group_by and group_by in df.columns:
        # FUTURE: add more summary metrics (mean, median, etc.)
        # FUTURE: Sampling logic preview
        summary_df = df.groupby(group_by).size().reset_index(name="count")
        table_html = summary_df.to_html(
            classes="pandas-table table",
            index=True,
            border=0,
            na_rep="",
        )
        note += f" | Grouped by: {group_by}"
    else:
        # Standard: View dataframe
        table_html = df.head(500).to_html(
            classes="pandas-table table",
            index=True,
            border=0,
            na_rep="",
        )

    # 5. Return partial (Iframe content)
    return templates.TemplateResponse(
        "/features/workbench/dataframe_partial.html",
        ctx.render(
            session=s,
            artifact=artifact,
            table_html=table_html,
            total_rows=len(df),
            columns=df.columns.tolist(),
            current_query=clean_query or "",
            note=note,
        )
    )
