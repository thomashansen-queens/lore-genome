"""
LoRe ASGI app factory for development with uvicorn set to auto-reload.
"""
import os
from pathlib import Path
import logging
from fastapi import FastAPI

from lore.core.runtime import Runtime
from lore.core.settings import Settings, Secrets
from lore.web.api import create_app

def create_app_factory() -> FastAPI:
    """Create a FastAPI app factory for development with auto-reload."""
    data_root = Path(os.environ.get("LORE_DATA_ROOT", ".")).resolve()
    # build logger
    logger = logging.getLogger("lore")
    logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    logger.setLevel(logging.DEBUG)
    rt = Runtime(
        cache_dir=data_root / "cache",
        data_root=data_root,
        sessions_dir=data_root / "sessions",
        settings_dir=data_root / "settings",
        settings=Settings(),
        secrets=Secrets(),
        logger=logger,
    )
    return create_app(rt)
