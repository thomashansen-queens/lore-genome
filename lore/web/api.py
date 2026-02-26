"""
FastAPI app instructions and entry point for LoRe Genome web UI.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from lore.core.runtime import Runtime
from lore.web.features.sessions.routes import router as sessions_router  # includes tasks router
from lore.web.features.settings.routes import router as settings_router
from lore.web.features.home.routes import router as home_router
from lore.web.features.registry.routes import router as registry_router
from lore.web.features.artifacts.routes import router as artifacts_router
from lore.web.features.tasks.routes import router as tasks_router
from lore.web.features.workbench.routes import router as workbench_router

def create_app(rt: Runtime) -> FastAPI:
    """Create and configure the FastAPI app for a LoRe Genome Runtime."""
    app = FastAPI()
    app.state.rt = rt
    app.mount("/static", StaticFiles(
        directory=Path(__file__).resolve().parent / "static"),
        name="static",
    )
    # Routers for different features
    app.include_router(home_router)
    app.include_router(settings_router)
    app.include_router(sessions_router)
    app.include_router(registry_router)
    app.include_router(artifacts_router)
    app.include_router(tasks_router)
    app.include_router(workbench_router)
    # Health check endpoint
    @app.get("/health")
    def health():
        return {"ok": True}
    return app
