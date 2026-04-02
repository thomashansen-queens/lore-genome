"""
Routes for the main home page of LoRe Genome web app.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, RedirectResponse

from lore.web.deps import RT, templates, PageContext

router = APIRouter(tags=["home"])


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
        request=ctx.request,
        name="/home/help.html",
        context=ctx.render(rt=rt),
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
        request=ctx.request,
        name="/home/dashboard.html",
        context=ctx.render(runtime=rt, sessions=sessions, workflows=workflows),
    )
