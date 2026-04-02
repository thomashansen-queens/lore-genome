"""
Routes for the main home page of LoRe Genome web app.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, RedirectResponse

from lore.web.deps import RT, templates, PageContext

router = APIRouter(tags=["home"])

# @router.get("/", response_class=HTMLResponse)
# def home_page(
#     rt: RT,
#     ctx: PageContext = Depends(),
# ):
#     """
#     Splash page/home screen.
#     """
#     recent_session = None
#     try:
#         sessions = rt.list_sessions(sort_by="updated_at")
#         recent_session = sessions[0] if sessions else None
#     except Exception as e:
#         rt.logger.error("Failed to load recent sessions for home page: %s", e)
#     return templates.TemplateResponse(
#         "/home/index.html",
#         ctx.render(
#             rt=rt,
#             recent_session=recent_session,
#         )
#     )

@router.get("/", response_class=RedirectResponse)
def home_page(ctx: PageContext = Depends()):
    """
    Redirect to the main dashboard.
    """
    ctx.generate_breadcrumbs()
    return ctx.redirect("/dashboard")

@router.get("/help", response_class=HTMLResponse)
def help_page(
    rt: RT,
    ctx: PageContext = Depends(),
):
    """
    Help page.
    """
    return templates.TemplateResponse(
        "/home/help.html",
        ctx.render(
            rt=rt,
        )
    )

@router.get("/dashboard", response_class=HTMLResponse)
def list_runtime_contents(rt: RT, ctx: PageContext = Depends()):
    """
    Workflows and Sessions listed on a dashboard.
    """
    sessions = rt.list_sessions()  # List of sessions as dictionaries
    workflows = rt.workflows.list_workflows()

    ctx.generate_breadcrumbs()
    return templates.TemplateResponse(
        "/home/dashboard.html",
        ctx.render(runtime=rt, sessions=sessions, workflows=workflows),
    )
