"""
Routers for the Workflows feature.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from lore.core.tasks import task_registry
from lore.core.utils import is_collection_type
from lore.core.topology.diagram import generate_dag_diagram
from lore.core.bindings import LiteralBinding, ReferenceBinding, UserInputBinding
from lore.web.deps import PageContext, RT, templates
from lore.web.utils.configure_task import build_widget_context, build_task_configure_context
from lore.web.utils.forms import get_form_list, get_form_str


router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.get("/", response_class=RedirectResponse)
def list_workflows(ctx: PageContext = Depends()):
    """
    Redirect to the dashboard page for now.
    """
    return ctx.redirect("/dashboard")


@router.post("/import", response_class=RedirectResponse)
async def import_workflow_action(
    rt: RT, 
    ctx: PageContext = Depends(),
    file: UploadFile = File(...)
):
    """Import a Workflow from a JSON file."""
    if not file or not file.filename:
        return ctx.redirect_back(fallback_url="/workflows")
    if not file.filename.endswith(".json"):
        ctx.add_msg("Invalid Workflow JSON file.", "error")
        return ctx.redirect_back(fallback_url="/workflows")

    content = await file.read()

    try:
        workflow = rt.workflows.import_workflow(content)
        ctx.add_msg("Workflow imported successfully!", "success")
        return ctx.redirect(f"/workflows/{workflow.id}")
    except ValueError as e:
        ctx.add_msg(str(e), "error")
        return ctx.redirect_back(fallback_url="/workflows")


@router.get("/{workflow_id}", response_class=HTMLResponse)
def view_workflow(workflow_id: str, rt: RT, ctx: PageContext = Depends()):
    """
    View details of a specific workflow.
    """
    w = rt.workflows.get_workflow(workflow_id)
    if not w:
        raise HTTPException(status_code=404, detail="Workflow not found")

    from lore.core.topology.diagram import Direction
    diagram_tb = generate_dag_diagram(w.tasks, task_registry, Direction.TB)
    diagram_lr = generate_dag_diagram(w.tasks, task_registry, Direction.LR)

    ctx.generate_breadcrumbs({workflow_id: w.name})
    return templates.TemplateResponse(
        request=ctx.request,
        name="features/workflows/detail.html",
        context=ctx.render(
            workflow=w,
            diagram_tb=diagram_tb,
            diagram_lr=diagram_lr,
        ),
    )


@router.post("/{workflow_id}/rename", response_class=RedirectResponse)
def rename_workflow_action(
    workflow_id: str,
    rt: RT,
    name: str = Form(...),
    ctx: PageContext = Depends(),
):
    """
    Rename a workflow's human-readable name. ID (filename) remains unchanged.
    """
    try:
        updated_workflow = rt.workflows.rename_workflow(workflow_id, name)
    except ValueError:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return ctx.redirect(url=f"/workflows/{updated_workflow.id}")


@router.post("/{workflow_id}/update_description", response_class=RedirectResponse)
def update_workflow_description_action(
    workflow_id: str,
    rt: RT,
    description: str = Form(""),
    ctx: PageContext = Depends(),
):
    """
    Update a workflow's description.
    """
    try:
        updated_workflow = rt.workflows.update_workflow_description(workflow_id, description)
    except ValueError:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return ctx.redirect_back(
        fallback_url=f"/workflows/{updated_workflow.id}",
        message="Description updated successfully.",
        message_type="success",
    )


@router.post("/{workflow_id}/clone", response_class=RedirectResponse)
def clone_workflow_action(workflow_id: str, rt: RT, ctx: PageContext = Depends()):
    """Clone an existing Workflow template."""
    try:
        cloned = rt.workflows.clone_workflow(workflow_id)
        return ctx.redirect_back(
            fallback_url=f"/workflows/{cloned.id}",
            message="Workflow cloned successfully.",
            message_type="success",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{workflow_id}/hydrate", response_class=RedirectResponse)
async def hydrate_workflow_action(
    workflow_id: str,
    rt: RT,
    ctx: PageContext = Depends(),
):
    """
    Action to hydrate a workflow to a Session.
    """
    workflow = rt.workflows.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    form_data = await ctx.request.form()

    # 1. Session name (optional)
    session_name = get_form_str(form_data, "session_name")
    if not session_name:
        session_name = f"Run: {workflow.name}"

    # 2. Package the Runtime Inputs
    runtime_inputs = {}
    for field_name in form_data.keys():
        if field_name.startswith("input__"):
            # strip "input__" to pass "step_id__key__idx"
            ref_key = field_name[len("input__"):]
            # Preserve blanks: If an empty value submitted, treat it as None
            binding_vals = get_form_list(form_data, field_name)
            if not binding_vals or (len(binding_vals) == 1 and not binding_vals[0].strip()):
                runtime_inputs[ref_key] = None
            elif len(binding_vals) > 1:
                runtime_inputs[ref_key] = [v.strip() for v in binding_vals if v.strip()]
            else:
                runtime_inputs[ref_key] = binding_vals[0].strip()

    # 3. Create Session and hydrate workflow into it
    new_session = rt.create_session(name=session_name)
    with rt.open_session(new_session.id) as session:
        rt.workflows.hydrate_workflow(workflow, session, runtime_inputs)

    return ctx.redirect(
        f"/sessions/{new_session.id}",
        message=f"Workflow {workflow.id} hydrated to new session successfully.",
        message_type="success",
    )


@router.get("/{workflow_id}/export")
async def export_workflow_action(workflow_id: str, rt: RT):
    """Exports a workflow as a downloadable JSON file."""
    workflow = rt.workflows.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # 1. Serialize workflow to JSON
    json_data = workflow.model_dump_json(indent=2)
    
    # 2. Create a clean default filename
    safe_filename = f"{workflow.id}.json"

    return Response(
        content=json_data,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={safe_filename}"}
    )


@router.post("/{workflow_id}/delete", response_class=RedirectResponse)
def delete_workflow_action(workflow_id: str, rt: RT, ctx: PageContext = Depends()):
    """Deletes a workflow template."""
    rt.workflows.delete_workflow(workflow_id)
    return ctx.redirect_back(
        fallback_url="/workflows",
        message="Workflow deleted successfully.",
        message_type="success",
    )

# --- Workflow task routes ---

@router.get("/{workflow_id}/tasks", response_class=HTMLResponse)
def workflow_steps(workflow_id: str, ctx: PageContext = Depends()):
    return ctx.redirect(f"/workflows/{workflow_id}")


@router.get("/{workflow_id}/tasks/{task_id}", response_class=HTMLResponse)
def view_workflow_task(
    workflow_id: str,
    task_id: str,
    rt: RT,
    ctx: PageContext = Depends(),
):
    workflow = rt.workflows.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    task = workflow.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found in workflow.")

    # Using a helper to adhere to the "thin router" philosophy
    ui_context = build_task_configure_context(workflow, task.registry_key, task_id)

    ctx.generate_breadcrumbs({workflow.id: workflow.name, "configure": "Configure Workflow Task"})

    return templates.TemplateResponse(
        request=ctx.request,
        name="features/workflows/task.html",
        context=ctx.render(
            workflow=workflow,
            context_type="workflow",
            preview_api_url=None,  # No preview for workflows (yet)
            commit_api_url=f"/workflows/{workflow.id}/tasks/{task_id}/update",
            **ui_context,  # Unpacks into expected vars for Jinja
        )
    )


@router.post("/{workflow_id}/tasks/{task_id}/rename", response_class=RedirectResponse)
async def rename_workflow_task_action(
    workflow_id: str,
    task_id: str,
    rt: RT,
    ctx: PageContext = Depends(),
):
    form_data = await ctx.request.form()
    new_name = get_form_str(form_data, "name") or ""
    rt.workflows.rename_task(workflow_id, task_id, new_name)

    return ctx.redirect_back(fallback_url=f"/workflows/{workflow_id}")


@router.post("/{workflow_id}/tasks/{task_id}/update", response_class=RedirectResponse)
async def update_workflow_task_action(
    workflow_id: str,
    task_id: str,
    rt: RT,
    ctx: PageContext = Depends(),
):
    """Update a Workflow Task's configuration."""
    task = rt.workflows.get_task(workflow_id, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found in workflow")
    task_def = task_registry.get(task.registry_key)
    if not task_def:
        raise HTTPException(status_code=404, detail="Task definition not found for task")

    # 1. Get form data
    form_data = await ctx.request.form()
    description = get_form_str(form_data, "description") or ""
    new_inputs = {}  # rebuild config from scratch

    # 2. Parse form data back into bindings
    for field_name, value in form_data.items():
        if not field_name.startswith("binding__"):
            continue

        parts = field_name.split("__")
        if len(parts) != 3:
            continue

        _, key, idx = parts
        binding_type = value

        # Grab that value from the form data
        val_field = f"val__{key}__{idx}"
        binding_vals = get_form_list(form_data, val_field)

        # Ensure the key exists in our new config list
        if key not in new_inputs:
            new_inputs[key] = []

        # 3. Construct the correct Binding object
        if binding_type == "literal":
            # i. Preserve blanks: Empty string means user cleared the value
            if not binding_vals or (len(binding_vals) == 1 and not binding_vals[0].strip()):
                new_inputs[key].append(LiteralBinding(value=None))
            else:
                field_info, extra = task_def.field_meta(key)
                is_multiple = extra.get("multiple", False)
                is_multiple = is_multiple or is_collection_type(field_info.annotation)
                is_multiple = is_multiple or len(binding_vals) > 1  # fallback heuristic

                # ii. Shatter comma-separated UI strings back to lists
                if is_multiple:
                    for b in binding_vals:
                        # Just in case the list came from a text input widget
                        vals = [v.strip() for v in b.split(",") if v.strip()]
                        for v in vals:
                            new_inputs[key].append(LiteralBinding(value=v))

                # iii. Allow commas in values
                else:
                    new_inputs[key].append(LiteralBinding(value=binding_vals[0].strip()))

        elif binding_type == "user_input":
            new_inputs[key].append(UserInputBinding(input_key=key))

        elif binding_type == "reference":
            # HTML input uses format "step_id.output_key"
            ref_parts = binding_vals[0].split(".", 1) if binding_vals else []
            if len(ref_parts) == 2:
                new_inputs[key].append(ReferenceBinding(source_id=ref_parts[0], output_key=ref_parts[1]))
            else:
                # This will break the workflow until fixed
                new_inputs[key].append(task.inputs[key][int(idx)])  # fallback to original binding if parsing fails

    # 4. Update the task and save the workflow
    rt.workflows.update_task(workflow_id, task_id, description, new_inputs)

    return ctx.redirect_back(
        fallback_url=f"/workflows/{workflow_id}",
        message="Task updated successfully.",
        message_type="success",
    )

# --- HTMX endpoints ---

@router.get("/{workflow_id}/tasks/{task_id}/input-widget", response_class=HTMLResponse)
def get_workflow_input_widget(
    workflow_id: str,
    task_id: str,
    field_name: str,
    rt: RT,
    ctx: PageContext = Depends()
):
    """HTMX endpoint to hot-swap input widgets when the user changes the dropdown type."""
    workflow = rt.workflows.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    task = workflow.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found in workflow")

    new_mode = ctx.request.query_params.get(f"__mode__{field_name}", "literal")

    widget_context = build_widget_context(workflow, task.registry_key, field_name, new_mode, task_id)
    f_name = widget_context.pop("field_name")
    f_info = widget_context.pop("field_info")

    return templates.TemplateResponse(
        request=ctx.request,
        name="components/task_input.html",
        context=ctx.render(
            context_type="workflow",
            edit_task_id=task_id,
            field_name=f_name,
            field_info=f_info,
            ui_state=widget_context,
        )
    )
