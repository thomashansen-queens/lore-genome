"""
Helpers for composing adapter configuration in the web layer.
"""
from typing import Any


def build_adapter_config(
    source_metadata: dict[str, Any] | None = None,
    ui_config: dict[str, Any] | None = None,
    ext: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """
    Compose the config dict handed to an adapter's adapt()/preview().
    Priority of overrides (lowest to highest):
    1. source_metadata: data-source declarations
    2. ui_config: live overrides from UI/manual input
    3. ext / extra: call-site feedback (e.g. RAM ceiling, ext arg)
    """
    config = {**(source_metadata or {}), **(ui_config or {})}
    if ext is not None:
        config["ext"] = ext
    config.update(extra)
    return config
