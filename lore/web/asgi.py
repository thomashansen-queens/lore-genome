"""
LoRe ASGI app factory for development with uvicorn set to auto-reload.
"""

from fastapi import FastAPI

from lore.core.runtime import build_runtime
from lore.web.api import create_app

def create_app_factory() -> FastAPI:
    """Create a FastAPI app factory for development with auto-reload."""
    rt = build_runtime(verbose=True)
    return create_app(rt)
