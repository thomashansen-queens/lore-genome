"""
Simple utility functions for use across the codebase.
"""

import json
import re
from typing import Any
try:
    from fastapi import UploadFile
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    UploadFile = type("UploadFile", (), {}) # Dummy type for isinstance checks

# [^\w\-.]: Anything that isn't a word character [a-zA-Z0-9_] or - .
# [\\/?@]: No slashes, question marks, at signs (more permissive)
_DEFAULT_BAD_CHARS = re.compile(r'[\\/?@]')

def clean(v: Any, *,
          empty_to_none: bool = True,
          reject: re.Pattern | None = _DEFAULT_BAD_CHARS,
          replace_with: str | None = None,
) -> str | None:
    """Helper cleans strings. Guards against objects to prevent type errors."""
    if v is None:
        return None
    if _HAS_FASTAPI and isinstance(v, UploadFile):
        raise ValueError(f"Expected string-like value, got UploadFile: {getattr(v, 'filename', 'unknown')}")

    if not isinstance(v, str):
        v = str(v)
    s = v.strip()

    if empty_to_none and s == "":
        return None

    if reject and reject.search(s):
        if replace_with is not None:
            s = reject.sub(replace_with, s)
        else:
            raise ValueError(f"Invalid characters in string: {s}")

    return s

ARTIFACT_ID_HEURISTIC = re.compile(r"^[0-9a-f]{12}$")

def is_artifact_id(val: Any) -> bool:
    """Heuristic check to see if a string looks like a LoRe Artifact ID."""
    if not isinstance(val, str):
        return False
    return bool(ARTIFACT_ID_HEURISTIC.match(val))


def serialize_to(records: list[dict], data_type: str, extension: str) -> str:
    """
    Serialize a list of records to a string in the given data_type and extension.
    Work-in-progress, currently only JSON and JSONL
    """
    ext = extension.lower().strip(".")

    # JSON lines
    if ext in {"jsonl", "ndjson"} or "jsonl" in data_type:
        return "\n".join(json.dumps(rec) for rec in records)

    # Standard JSON
    return json.dumps(records, indent=2)


def normalize_query(query_string: str) -> str:
    """
    Translates user-friendly or SQL-style syntax into valid Pandas query syntax.
    Useful for Explore Artifact view.
    """
    if not query_string:
        return ""

    # 1. SQL-style Logical Operators -> Python style
    # We use regex word boundaries (\b) to ensure we don't replace 'AND' inside a country name
    query_string = re.sub(r'\bAND\b', 'and', query_string, flags=re.IGNORECASE)
    query_string = re.sub(r'\bOR\b', 'or', query_string, flags=re.IGNORECASE)

    # 2. Cleanup whitespace
    query_string = " ".join(query_string.split())

    return query_string


def slugify(value: str, max_length: int = 32) -> str:
    """
    Simple slugify function to create filesystem-friendly names.
    Converts to lowercase, replaces spaces with underscores, and removes bad chars.
    """
    value = value.lower()
    value = re.sub(r'\s+', '_', value)  # Replace spaces with underscores
    value = re.sub(r'[^\w\-.]', '', value)  # Remove non-word chars except - and .
    return value[:max_length]
