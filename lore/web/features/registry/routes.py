"""
Routes for inspecting the Task Registry.
"""
import json
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from lore.core.tasks import task_registry
from lore.web.deps import templates

router = APIRouter(prefix='/registry', tags=['registry'])

@router.get('/', response_class=HTMLResponse)
def library_index(request: Request):
    """List all registered tasks."""
    tasks = []

    for key, task_def in task_registry.all.items():
        # Generate JSON schema if a model exists
        schema = None
        if task_def.input_model:
            schema = json.dumps(task_def.input_model.model_json_schema(), indent=2)

        tasks.append({
            "key": key,
            "description": task_def.description,
            "has_model": task_def.input_model is not None,
            "schema_json": schema
        })

    breadcrumbs = [("Home", "/"), ("Task Registry", None)]

    return templates.TemplateResponse(
        "/task_registry.html",
        {
            "request": request, "breadcrumbs": breadcrumbs,
            "tasks": tasks
        },
    )
