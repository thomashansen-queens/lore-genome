"""
Allows dependency injection of the Runtime into FastAPI routes.
By importing RT from this module, route handlers can declare a parameter
of type RT to automatically receive the Runtime instance.

Example:
from lore.web.deps import RT

def my_route(rt: RT):
    # 'rt' is automatically provided and typed as a Runtime instance
"""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Generator, Tuple, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from fastapi import HTTPException, Request, Depends
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from lore.core.runtime import Runtime
from lore.core.session import Session
from lore.core.tasks import task_registry
from lore.core.adapters import adapter_registry

# --- Templates setup ---
TEMPLATES_DIR = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["task_registry"] = task_registry.all
templates.env.globals["adapter_registry"] = adapter_registry
templates.env.globals["version"] = version("lore-genome")

# --- Runtime Dependency injection ---
def get_runtime(request: Request) -> Runtime:
    """Dependency to retrieve the Runtime from the FastAPI request state."""
    return request.app.state.rt

# type alias for dependency injection
RT = Annotated[Runtime, Depends(get_runtime)]

# --- Session Dependency injection ---
def get_active_session(session_id: str, rt: RT) -> Generator['Session', None, None]:
    """
    Dependency that finds a session, opens it (Context Manager), yields it,
    and handles 404s if it doesn't exist.
    """
    try:
        # The 'yield' keeps the context manager open while the route runs
        with rt.get_session(session_id) as s:
            yield s
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found") from e

# Type alias
ActiveSession = Annotated['Session', Depends(get_active_session)]

def get_read_only_session(session_id: str, rt: RT) -> Generator["Session", None, None]:
    """
    Dependency for read-only session access (e.g., polling endpoints).
    Safely opens and closes the session without triggering a disk write.
    """
    try:
        with rt.get_session(session_id, read_only=True) as s:
            yield s
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found") from e

# Type alias for easy injection in your routers
ReadOnlySession = Annotated["Session", Depends(get_read_only_session)]

# --- Breadcrumbs ---

Crumb = tuple[str, Optional[str]]

def build_breadcrumbs(
    session_id: str | None = None,
    session_name: str | None = None,
    final_item: str | None = None,
    extra: list[Crumb] | None = None,
) -> List[Crumb]:
    """
    Helper to build a standard breadcrumb trail for session-related pages.
    """
    crumbs: List[Crumb] = [("Home", "/"), ("Sessions", "/sessions/")]
    # Session context
    if session_id:
        label = session_name if session_name else session_id
        # Truncate long IDs for display
        if not session_name and len(label) > 12:
            label = f"{label[:8]}..."

        crumbs.append((label, f"/sessions/{session_id}/"))
    # Intermediate items
    if extra:
        crumbs.extend(extra)
    # Current page (leaf)
    if final_item:
        crumbs.append((final_item, None))
    return crumbs

# --- Page context collector ---

@dataclass
class PageContext:
    """
    Automatically collects Request and standard arguments required for almost
    every HTML page.
    FastAPI will automatically populates these from the Request/Query Params.
    """
    request: Request
    message: str | None = None  # query param: ?message=...
    message_type: str = "info"  # query param: ?message_type=...

    def __post_init__(self):
        """
        Runs automatically after __init__. Sets up internal state that doesn't
        come from HTTP parameters. Otherwise, FastAPI will try to use breadcrumbs
        as a Request Body paramenter.
        """
        self.breadcrumbs = build_breadcrumbs()
        self.messages: list[tuple[str, str]] = []
        if self.message:
            self.messages.append((self.message_type, self.message))

    def render(self, **kwargs):
        """Combine standard UI context with page-specific data passed as kwargs."""
        base_context = {
            "request": self.request,
            "messages": self.messages,
            "breadcrumbs": self.breadcrumbs,
        }
        return base_context | kwargs

    def add_msg(self, message: str = "", message_type: str = "info"):
        """
        Helper to add a message to the queue
        Types: success, danger, warning, info
        """
        self.messages.append((message_type, message))

    def get_safe_referer(self, *, required: bool = False) -> str | None:
        """
        Return a same-origin Referer URL, or None when missing/unsafe.
        If required=True, raises ValueError when a safe Referer is unavailable.
        """
        referer = self.request.headers.get("referer")
        if not referer:
            if required:
                raise ValueError("Missing Referer header")
            return None

        ref = urlparse(referer)
        req = self.request.url

        is_relative = not ref.scheme and not ref.netloc and referer.startswith("/")
        is_same_origin = ref.scheme == req.scheme and ref.netloc == req.netloc

        if is_relative or is_same_origin:
            return referer

        if required:
            raise ValueError("Unsafe Referer header")
        return None

    def with_query_params(self, url: str, **params: str | None) -> str:
        """Return URL with non-empty query params merged into existing params."""
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))

        for key, val in params.items():
            if val is None:
                continue
            query[key] = val

        new_query = urlencode(query, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    def redirect(
        self,
        url: str,
        *,
        status_code: int = 303,
        message: str | None = None,
        message_type: str = "info",
    ) -> RedirectResponse:
        """Create RedirectResponse with optional query params (e.g. flash message)"""
        target = self.with_query_params(
            url,
            message=message,
            message_type=message_type if message else None,
        )
        return RedirectResponse(url=target, status_code=status_code)

    def redirect_back(
        self,
        fallback_url: str,
        *,
        status_code: int = 303,
        message: str | None = None,
        message_type: str = "info",
        required_referer: bool = False,
    ) -> RedirectResponse:
        """
        Redirect to safe referer when available, otherwise fallback URL.
        """
        target = self.get_safe_referer(required=required_referer) or fallback_url
        return self.redirect(
            target,
            status_code=status_code,
            message=message,
            message_type=message_type,
        )
