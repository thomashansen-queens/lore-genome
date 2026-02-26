"""
Routes for the global settings page.
"""
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from lore.core.settings import save_settings, save_secrets
from lore.web.deps import RT, templates, PageContext
from lore.web.utils import get_form_str

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/", response_class=HTMLResponse)
def settings_page(rt: RT, ctx: PageContext = Depends()):
    """Render the settings page."""
    ctx.breadcrumbs = [("Home", "/"), ("Settings", None)]
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
    errors = []
    requires_restart = False

    # 1. Update data root if provided
    raw_data_root = get_form_str(form_data, "data_root", reject=None)
    if isinstance(raw_data_root, str):
        try:
            data_root = Path(raw_data_root.strip()).expanduser().resolve()
        except Exception:
            errors.append(f"Invalid data root path: {raw_data_root}")
            data_root = None

        if data_root and data_root != rt.settings.data_root:
            # check that path is viable
            if data_root.parent == data_root:
                errors.append("Data root should not literally be root!")
            elif data_root.exists() and not data_root.is_dir():
                errors.append("Data root must be a directory!")
            else:
                rt.settings.data_root = data_root
                requires_restart = True

    # 2. Sanitize and update NCBI key
    raw_key = get_form_str(form_data, "ncbi_api_key")
    if isinstance(raw_key, str):
        if raw_key and " " in raw_key:
            errors.append("NCBI API key cannot contain spaces.")
        else:
            rt.secrets.ncbi_api_key = raw_key or ""

    ctx.breadcrumbs = [("Home", "/"), ("Settings", None)]

    # 3. Sanitize and validate MMseqs2 path
    raw_mmseqs = get_form_str(form_data, "mmseqs_path")
    if isinstance(raw_mmseqs, str):
        clean_mmseqs = raw_mmseqs.strip()
        if clean_mmseqs:
            rt.settings.mmseqs_path = clean_mmseqs
        else:
            rt.settings.mmseqs_path = "mmseqs"

    # 4. Send back on error
    if errors:
        return templates.TemplateResponse(
            "/core/settings.html",
            ctx.render(
                request=ctx.request,
                settings=rt.settings,
                secrets=rt.secrets,
                message="; ".join(errors),
                message_type="danger",
                active_root=str(rt.data_root),
            )
        )

    # 4. Save and success
    save_settings(rt.cache_dir, rt.settings)
    save_secrets(rt.cache_dir, rt.secrets)

    if requires_restart:
        message = "Settings saved successfully! Please restart the server to apply changes."
    else:
        message = "Settings saved successfully!"

    return templates.TemplateResponse(
        "/core/settings.html",
        ctx.render(
            settings=rt.settings,
            secrets=rt.secrets,
            message=message,
            message_type="success",
            active_root=str(rt.data_root),
        )
    )
