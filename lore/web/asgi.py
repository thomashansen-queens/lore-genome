"""
LoRe ASGI app factory for development with uvicorn set to auto-reload.
"""
import logging
from fastapi import FastAPI

from lore.core.runtime import build_runtime
from lore.web.api import create_app

def create_app_factory() -> FastAPI:
    """Create a FastAPI app factory for development with auto-reload."""
    logger = logging.getLogger("lore")
    logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    logger.setLevel(logging.DEBUG)
    rt = build_runtime(verbose=True)
    return create_app(rt)
