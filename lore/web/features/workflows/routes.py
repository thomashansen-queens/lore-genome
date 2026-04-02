"""
Routers for the Workflows feature.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic_core import PydanticUndefined

from lore.core.tasks import task_registry
from lore.core.utils import is_collection_type
from lore.core.workflows.diagram import generate_workflow_diagram
from lore.core.bindings import LiteralBinding, ReferenceBinding, UserInputBinding
from lore.web.deps import PageContext, RT, templates
from lore.web.utils.forms import get_form_list, get_form_str, format_binding_for_ui


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
    workflow = rt.workflows.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    diagram_tb = generate_workflow_diagram(workflow, task_registry, "TB")
    diagram_lr = generate_workflow_diagram(workflow, task_registry, "LR")
    
    ctx.generate_breadcrumbs({workflow_id: workflow.name})
    return templates.TemplateResponse(
        "features/workflows/detail.html",
        ctx.render(
            workflow=workflow,
            task_registry=task_registry,
            diagram_tb=diagram_tb,
            diagram_lr=diagram_lr,
        )
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

    return ctx.redirect_back(fallback_url=f"/workflows/{updated_workflow.id}")


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
    for field_name, value in form_data.items():
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

# --- Step routes ---

@router.get("/{workflow_id}/steps", response_class=HTMLResponse)
def workflow_steps(workflow_id: str, ctx: PageContext = Depends()):
    return ctx.redirect(f"/workflows/{workflow_id}")


@router.get("/{workflow_id}/steps/{step_id}", response_class=HTMLResponse)
def view_workflow_step(workflow_id: str, step_id: str, rt: RT, ctx: PageContext = Depends()):
    workflow = rt.workflows.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    step = next((s for s in workflow.steps if s.id == step_id), None)
    if not step:
        raise HTTPException(status_code=404, detail="Step not found in workflow")

    # 1. Get Task schema (allowing for missing TaskDefinition in shared workflows)
    task_def = task_registry.get_safe(step.task_key)
    all_keys = list(set(task_def.input_model.model_fields.keys()) | set(step.inputs.keys()))

    # 2. UI config with values
    ui_config = {}
    for key in all_keys:
        bindings = step.inputs.get(key, [])
        ui_config[key] = []
        literals = []

        for b in bindings:
            # i. Type could be Pydantic model or dict; allow both
            b_type = getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)

            # ii. Combine multiple literals to single comma-separated string
            if b.type == "literal":
                literals.append(format_binding_for_ui(b.value))
            else:
                ui_config[key].append(b)

        if literals:
            # Filter out empty strings to avoid weird commas
            clean_literals = [l for l in literals if l.strip()]
            ui_config[key].insert(0, {"type": "literal", "value": ", ".join(clean_literals)})

        # iii. Preserve blanks: If no literals, show default
        if not ui_config[key]:
            field_info, _ = task_def.field_meta(key)
            default_val = ""
            if field_info.default is not PydanticUndefined:
                default_val = format_binding_for_ui(field_info.default)

            ui_config[key].append({"type": "literal", "value": default_val})

    # 3. UI schemas for rendering input fields
    ui_schemas = {}
    for key in all_keys:
            field_info, extra = task_def.field_meta(key)
            mutable_extra = dict(extra) if extra else {}
            mutable_extra["is_required"] = (
                field_info.is_required()
                if field_info and hasattr(field_info, "is_required")
                else False
            )
            ui_schemas[key] = mutable_extra

    ctx.generate_breadcrumbs({
        workflow_id: workflow.name,
        f"{workflow_id}/steps/{step_id}": task_def.name,
    })
    return templates.TemplateResponse(
        "features/workflows/step.html",
        ctx.render(
            workflow=workflow,
            step=step,
            task_def=task_def,
            ui_config=ui_config,
            ui_schemas=ui_schemas,
        )
    )


@router.post("/{workflow_id}/steps/{step_id}/rename", response_class=RedirectResponse)
async def rename_workflow_step_action(
    workflow_id: str,
    step_id: str,
    rt: RT,
    ctx: PageContext = Depends(),
):
    form_data = await ctx.request.form()
    new_name = get_form_str(form_data, "name") or ""
    rt.workflows.rename_step(workflow_id, step_id, new_name)

    return ctx.redirect_back(fallback_url=f"/workflows/{workflow_id}")


@router.post("/{workflow_id}/steps/{step_id}/update", response_class=RedirectResponse)
async def update_workflow_step_action(
    workflow_id: str,
    step_id: str,
    rt: RT,
    ctx: PageContext = Depends(),
):
    """Update a Workflow Step's configuration."""
    step = rt.workflows.get_step(workflow_id, step_id)
    if not step:
        raise HTTPException(status_code=404, detail="Step not found in workflow")
    task_def = task_registry.get(step.task_key)
    if not task_def:
        raise HTTPException(status_code=404, detail="Task definition not found for step")

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
                new_inputs[key].append(step.inputs[key][int(idx)])  # fallback to original binding if parsing fails

    # 4. Update the step and save the workflow
    rt.workflows.update_step(workflow_id, step_id, description, new_inputs)

    return ctx.redirect_back(
        fallback_url=f"/workflows/{workflow_id}",
        message="Step updated successfully.",
        message_type="success",
    )
