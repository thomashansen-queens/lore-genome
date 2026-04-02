"""
Routes for the global settings page.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from lore.core.settings import config_registry, save_settings
from lore.web.deps import RT, templates, PageContext
from lore.web.utils.forms import form_html_to_dict, get_form_str

router = APIRouter(prefix="/settings", tags=["settings"])

# Populate sidebar for each route
def get_sidebar_plugins():
    return [
        {"key": key, "title": schema.model_config.get("title", key.title())}
        for key, schema in config_registry.all.items()
    ]


@router.get("/", response_class=RedirectResponse)
def redirect_to_core():
    """Default Settings landing page."""
    return RedirectResponse(url="/settings/core")


@router.get("/core")
def view_core_settings(
    rt: RT, 
    ctx: PageContext = Depends(),
    plugins_list = Depends(get_sidebar_plugins),
):
    """Render the Core engine settings form."""
    # All settings live in one place, but we only want to show a subset
    hidden_fields = {"plugins", "api_host", "api_port"}
    core_fields = {
        name: meta for name, meta in type(rt.settings).model_fields.items()
        if name not in hidden_fields
    }

    ctx.generate_breadcrumbs({"settings": "Settings", "core": "Core"})
    return templates.TemplateResponse(
        request=ctx.request,
        name="features/settings/core.html",
        context=ctx.render(
            settings=rt.settings,
            registered_plugins=plugins_list,
            core_fields=core_fields,
        )
    )


@router.post("/core")
async def save_core_settings(
    rt: RT, 
    ctx: PageContext = Depends()
):
    """Parse the form and save Core engine settings."""
    form_data = await ctx.request.form()

    try:
        # 1. Parse HTML form
        update_data = form_html_to_dict(
            form_data,
            type(rt.settings),
            bad_chars=None,
            blank_to_default=True,
        )

        # 2. Merge the form updates into the current settings state
        current_state = rt.settings.model_dump()
        current_state.update(update_data)

        # 3. Let Pydantic enforce your constraints (e.g. max/min limits)
        rt.settings = type(rt.settings).model_validate(current_state)

    except ValueError as e:
        return ctx.redirect_back(
            fallback_url="/settings/core", 
            message=f"Invalid setting: {e}", 
            message_type="error"
        )

    # 4. Persist to disk
    save_settings(rt.settings_dir, rt.settings)

    return ctx.redirect_back(
        fallback_url="/settings/core", 
        message="Core settings updated successfully.", 
        message_type="success"
    )


# --- Plugin-specific settings routes ---

@router.get("/plugins/{plugin_key}")
def view_plugin_settings(
    plugin_key: str, 
    rt: RT, 
    ctx: PageContext = Depends(),
    plugins_list = Depends(get_sidebar_plugins),
):
    """Dynamically render the form for a specific plugin."""
    if plugin_key not in config_registry.all:
        raise HTTPException(404, "Plugin not found")

    schema_cls = config_registry.all[plugin_key]
    instance = rt.settings.get_plugin_config(plugin_key)
    if instance is None:
        instance = schema_cls()  # Initialize with defaults if not set yet

    active_plugin = {
        "key": plugin_key,
        "title": schema_cls.model_config.get("title", plugin_key.title()),
        "description": schema_cls.__doc__,
        "instance": instance,
        "fields": schema_cls.model_fields,
    }

    return templates.TemplateResponse(
        request=ctx.request,
        name="features/settings/plugin.html",
        context=ctx.render(
            registered_plugins=plugins_list,
            active_plugin=active_plugin,
        )
    )


@router.post("/plugins/{plugin_key}")
async def save_plugin_settings(
    plugin_key: str, 
    rt: RT, 
    ctx: PageContext = Depends()
):
    """Save only the settings for this specific plugin."""
    if plugin_key not in config_registry.all:
        raise HTTPException(404, "Plugin not found")

    schema_cls = config_registry.all[plugin_key]
    form_data = await ctx.request.form()

    try:
        # 1. Parse HTML form
        update_data = form_html_to_dict(form_data, schema_cls, bad_chars=None)

        # 2. Merge the form updates into the current settings state
        current_plugin_obj = rt.settings.get_plugin_config(plugin_key)
        current_state = current_plugin_obj.model_dump() if current_plugin_obj else {}
        current_state.update(update_data)
        plugin_data = schema_cls.model_validate(current_state)
        rt.settings.plugins[plugin_key] = plugin_data.model_dump()

    except ValueError as e:
        return ctx.redirect_back(
            fallback_url=f"/settings/plugins/{plugin_key}", 
            message=f"Invalid setting: {e}", 
            message_type="error",
        )

    # 3. Persist to disk
    save_settings(rt.settings_dir, rt.settings)

    return ctx.redirect_back(
        fallback_url=f"/settings/plugins/{plugin_key}", 
        message=f"{schema_cls.model_config.get('title', schema_cls.__name__)} settings saved.", 
        message_type="success",
    )
