"""
FastAPI app instructions and entry point for LoRe Genome web UI.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from lore.core.runtime import Runtime
from lore.web.features.sessions.routes import router as sessions_router
from lore.web.features.settings.routes import router as settings_router
from lore.web.features.home.routes import router as home_router
from lore.web.features.artifacts.routes import router as artifacts_router
from lore.web.features.tasks.routes import router as tasks_router
from lore.web.features.runtime.routes import router as runtime_router
from lore.web.features.workflows.routes import router as workflows_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for FastAPI app. Ensures proper initialization and cleanup."""
    # --- Startup ---
    # (no special instructions for now)
    yield

    # --- Shutdown ---
    if hasattr(app.state, "rt"):
        app.state.rt.logger.info("Shutting down Runtime and cleaning up background jobs...")
        app.state.rt.executor.shutdown()


def create_app(rt: Runtime) -> FastAPI:
    """Create and configure the FastAPI app for a LoRe Genome Runtime."""
    app = FastAPI(title="LoRe Genome", lifespan=lifespan)
    app.state.rt = rt
    app.mount("/static", StaticFiles(
        directory=Path(__file__).resolve().parent / "static"),
        name="static",
    )
    # Routers for different features
    app.include_router(home_router)
    app.include_router(settings_router)
    app.include_router(sessions_router)
    app.include_router(artifacts_router)
    app.include_router(tasks_router)
    app.include_router(runtime_router)
    app.include_router(workflows_router)
    # Health check endpoint
    @app.get("/health")
    def health():
        return {"ok": True}
    return app
