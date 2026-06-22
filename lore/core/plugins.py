"""
Plugin management for LoRē Genome.
"""
import importlib
import os
from pathlib import Path
import sys
import logging


logger = logging.getLogger(__name__)


def _import_plugin(module_name: str) -> None:
    """
    Dynamically import a plugin module from the given path.
    """
    try:
        importlib.import_module(module_name)
        logger.debug(f"Successfully loaded plugin: {module_name}")
    except ImportError as e:
        # Dependency management: For now, just log the missing lib and let the
        # user sort it out.
        missing_lib = e.name or str(e)
        logger.warning(
            f"Skipping plugin '{module_name}'. Missing dependency: '{missing_lib}'. "
            f"Try `pip install {missing_lib.split('.')[0]}` to resolve."
        )
    except Exception as e:
        logger.error(f"Failed to load plugin '{module_name}': {e}", exc_info=True)


def discover_plugins(directory: Path | str, namespace: str | None = None) -> None:
    """
    Scan for plugin modules and import them, evaluating code in __init__.py and
    in doing so, running decorators to register readers, adapters, and tasks.
    Recurses through subdirs, stopping at packages (__init__.py is present).
    """
    #1. Resolve target directory and namespace
    target_dir = Path(directory).expanduser().resolve()

    if not target_dir.exists() or not target_dir.is_dir():
        logger.warning(f"Plugin directory not found: '{target_dir}'")
        return

    logger.info("Scanning for plugins in %s", target_dir)

    if not namespace and str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))
        logger.debug(f"Added '{target_dir}' to sys.path for plugin discovery.")

    # 2. Walk the directory tree, stopping at package roots
    for root_str, dirs, files in os.walk(target_dir):
        root = Path(root_str)

        # 3. Skip hidden/dunder dirs
        dirs[:] = [d for d in dirs if not d.startswith(".") and not d.startswith("__")]

        # 4. Start digging
        if root != target_dir and (root / "__init__.py").exists():
            rel_path = root.relative_to(target_dir)
            module_name = ".".join(rel_path.parts)
            if namespace:
                module_name = f"{namespace}.{module_name}"
            if module_name not in sys.modules:
                _import_plugin(module_name)
            dirs.clear()  # Don't recurse into subdirs of a package
            continue

        # 5. If not a package, look for standalone .py scripts
        for file in files:
            if file.startswith("." or file.startswith("__")) or not file.endswith(".py"):
                continue
            rel_path = (root / file).relative_to(target_dir)
            module_name = ".".join(rel_path.with_suffix("").parts)
            if namespace:
                module_name = f"{namespace}.{module_name}"
            if module_name not in sys.modules:
                _import_plugin(module_name)
