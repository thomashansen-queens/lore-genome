"""
Routes for the main home page of LoRe Genome web app.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from lore.web.deps import RT, templates, PageContext

router = APIRouter(tags=["home"])

@router.get("/", response_class=HTMLResponse)
def home_page(
    rt: RT,
    ctx: PageContext = Depends(),
):
    """
    Splash page/home screen.
    """
    recent_session = None
    try:
        sessions = rt.list_sessions(sort_by="updated_at")
        recent_session = sessions[0] if sessions else None
    except Exception:
        pass
    return templates.TemplateResponse(
        "/core/home.html",
        ctx.render(
            breadcrumbs=[("Home", None)],
            rt=rt,
            recent_session=recent_session,
        )
    )

@router.get("/help", response_class=HTMLResponse)
def help_page(
    rt: RT,
    ctx: PageContext = Depends(),
):
    """
    Help page.
    """
    return templates.TemplateResponse(
        "/core/help.html",
        ctx.render(
            breadcrumbs=[("Home", "/"), ("Help", None)],
            rt=rt,
        )
    )
