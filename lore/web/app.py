"""
Utility functions for serving the LoRe Genome web application.
"""
import os
import socket

import uvicorn

from lore.core.runtime import Runtime
from lore.web.api import create_app
from lore.web.asgi import create_app_factory  # dev tool

def pick_free_port(host: str, port: int | None) -> int:
    """Find a free port on localhost."""
    if port is not None:
        return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))  # 0 = let OS choose
        return int(s.getsockname()[1])

def make_url(host: str, port: int) -> str:
    """Construct a URL string from host and port."""
    return f"http://{host}:{port}"

def run_ui(rt: Runtime, host: str = "127.0.0.1", port: int | None = None) -> None:
    """Launch the LoRe Genome web service."""
    app = create_app(rt)
    port = pick_free_port(host, port)
    url = make_url(host, port)
    rt.logger.info("Starting LoRe UI at %s", url)
    uvicorn.run(app, host=host, port=port, log_level="info")

def run_ui_reload(rt: Runtime, host: str = "127.0.0.1", port=8080) -> None:
    """Launch the LoRe Genome web service with auto-reload for development."""
    os.environ["LORE_DATA_ROOT"] = str(rt.data_root)
    url = make_url(host, port)
    rt.logger.info("Starting LoRe UI (with reload) at %s", url)
    uvicorn.run("lore.web.asgi:create_app_factory", factory=True, host=host, port=port, log_level="info", reload=True)
