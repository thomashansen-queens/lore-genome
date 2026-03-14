"""
Plugin management for LoRe Genome.
"""
import importlib.util
from pathlib import Path
import sys
import logging


logger = logging.getLogger("lore.plugins")


def discover_plugins(directory: Path | str) -> None:
    """
    Scan for plugin modules and import and runtime. Looks for decorators like
    @task_registry.register
    """
    target_dir = Path(directory).expanduser().resolve()

    if not target_dir.exists() or not target_dir.is_dir():
        logger.warning(f"Plugin directory not found: '{target_dir}'")
        return

    logger.info("Scanning for plugins in %s", target_dir)

    for py_file in target_dir.rglob("*.py"):
        # 1. Skip dunder and hidden
        if py_file.name.startswith("__") or py_file.name.startswith("."):
            continue

        # 2. Create a module name based on relative path to avoid collisions
        rel_path = py_file.relative_to(target_dir)
        module_name = "lore_plugin_" + str(rel_path.with_suffix("")).replace("/", "_").replace("\\", "_")

        if module_name in sys.modules:
            continue  # already loaded

        try:
            # --- importlib boilerplate ---
            # A. Make "spec" from the file location
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                # B. Create empty module from spec and register it in sys.modules
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module

                # C. Execute the module to populate it (triggers decorators)
                spec.loader.exec_module(module)

                logger.debug("Loaded plugin module: %s", module_name)

        except Exception as e:
            logger.error(f"Failed to load plugin '{py_file}': {e}", exc_info=True)
