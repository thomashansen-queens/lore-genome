"""
Helpers to navigate and manipulate nested dicts via "dot paths".
"""
from typing import Any

def get_by_path(d: dict, path: str, *, default = None) -> Any:
    """Get a value from a nested dict via a dot-separated path."""
    if not path or path == ".":
        return d
    keys = path.split('.')
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def set_by_path(d: dict, path: str, value) -> None:
    """Set a value in a nested dict via a dot-separated path."""
    if not path or path == ".":
        # Assume overwriting the root is a logic error in the caller
        raise ValueError("Cannot set_by_path on root ('.')")
    keys = path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value

def delete_by_path(d: dict, path: str) -> None:
    """Delete a key in a nested dict via a dot-separated path."""
    keys = path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            return  # Key path does not exist; nothing to delete
        current = current[key]
    current.pop(keys[-1], None)  # Remove the key if it exists
