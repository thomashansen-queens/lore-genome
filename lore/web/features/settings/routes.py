"""
Routes for the global settings page.
"""
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from lore.web.deps import RT, templates, PageContext
from lore.web.utils.forms import form_html_to_dict

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/", response_class=HTMLResponse)
def settings_page(rt: RT, ctx: PageContext = Depends()):
    """Render the settings page."""
    ctx.generate_breadcrumbs()
    return templates.TemplateResponse(
        "core/settings.html",
        ctx.render(
            settings=rt.settings,
            secrets=rt.secrets,
            active_root=str(rt.data_root),
        )
    )


@router.post("/", response_class=HTMLResponse)
async def save_settings_action(rt: RT, ctx: PageContext = Depends()):
    """Save all settings and secrets from the settings forms."""
    form_data = await ctx.request.form()

    requires_restart, errors = rt.update_settings(
        settings_dict=form_html_to_dict(form_data, rt.settings.__class__, bad_chars=None),
        secrets_dict=form_html_to_dict(form_data, rt.secrets.__class__, bad_chars=None),
    )

    # 3. Send back on error
    if errors:
        return ctx.redirect_back(
            message="; ".join(errors),
            message_type="danger",
            fallback_url="/settings",
        )

    # 4. Save and success
    msg = "Settings updated successfully."
    if requires_restart:
        msg += " Please restart the server to apply changes."
    return ctx.redirect_back(
        message=msg,
        message_type="success",
        fallback_url="/settings",
    )
