"""
Simple utility functions for use across the codebase.
"""

import re
from typing import Any, Collection
try:
    from fastapi import UploadFile
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    UploadFile = type("UploadFile", (), {}) # Dummy type for isinstance checks

# [^\w\-.]: Anything that isn't a word character [a-zA-Z0-9_] or - .
# [\\/?@]: No slashes, question marks, at signs (more permissive)
_DEFAULT_BAD_CHARS = re.compile(r'[\\/?@]')
BadChars = re.Pattern[str] | str | set[str] | None

def _compile_bad_chars(bad_chars: BadChars) -> re.Pattern[str] | None:
    """Compile bad characters into a regex pattern for efficient searching and replacing."""
    if bad_chars is None:
        return None
    if isinstance(bad_chars, re.Pattern):
        return bad_chars
    if isinstance(bad_chars, set):
        chars = "".join(sorted(bad_chars))
    else:  # str
        chars = bad_chars

    return re.compile(f"[{re.escape(chars)}]") if chars else None


def clean(v: Any, *,
          empty_to_none: bool = True,
          bad_chars: BadChars = _DEFAULT_BAD_CHARS,
          replace_with: str | None = None,
) -> str | None:
    """Helper cleans strings. Guards against objects to prevent type errors."""
    if v is None:
        return None
    if _HAS_FASTAPI and isinstance(v, UploadFile):
        raise ValueError(f"Expected string-like value, got UploadFile: {getattr(v, 'filename', 'unknown')}")

    s = v.strip() if isinstance(v, str) else str(v).strip()
    if empty_to_none and s == "":
        return None

    pattern = _compile_bad_chars(bad_chars)
    if pattern and pattern.search(s):
        if replace_with is not None:
            s = pattern.sub(replace_with, s)
        else:
            raise ValueError(f"Invalid characters in string: {s}")

    return s

ARTIFACT_ID_HEURISTIC = re.compile(r"^[0-9a-f]{12}$")

def is_artifact_id(val: Any) -> bool:
    """Heuristic check to see if a string looks like a LoRe Artifact ID."""
    if not isinstance(val, str):
        return False
    return bool(ARTIFACT_ID_HEURISTIC.match(val))


# def serialize_to(records: list[dict], data_type: str, extension: str) -> str:
#     """
#     Serialize a list of records to a string in the given data_type and extension.
#     Work-in-progress, currently only JSON and JSONL
#     """
#     ext = extension.lower().strip(".")

#     # JSON lines
#     if ext in {"jsonl", "ndjson"} or "jsonl" in data_type:
#         return "\n".join(json.dumps(rec) for rec in records)

#     # Standard JSON
#     return json.dumps(records, indent=2)


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

# --- Object naming ---

def slugify(value: str, max_length: int = 32) -> str:
    """
    Simple slugify function to create filesystem-friendly names.
    Converts to lowercase, replaces spaces with underscores, and removes bad chars.
    """
    value = value.strip().lower()
    if not value:
        return "unnamed"
    value = re.sub(r'\s+', '_', value)  # Replace spaces with underscores
    value = re.sub(r'[^\w\-.]', '', value)  # Remove non-word chars except - and .
    return value[:max_length]

_BAD_DISPLAY_CHARS = re.compile(r"[\x00-\x1f\x7f\n\r]")

def normalize_display_name(name: str | None, default: str = "unnamed") -> str:
    """Normalize a user-provided name for objects. If name is None or empty, uses default."""
    name = name.strip() if name and name.strip() else default
    cleaned = clean(name, bad_chars=_BAD_DISPLAY_CHARS, replace_with="", empty_to_none=False)
    assert cleaned is not None
    return cleaned

_INCREMENT_RE = re.compile(r"^(?P<base>.*?)(?: \((?P<num>\d+)\))?$")

def auto_increment(name: str, existing: Collection[str]) -> str:
    """
    If name already exists in existing, append (2), (3), etc. until it's unique.
    """
    if name not in existing:
        return name

    match = _INCREMENT_RE.match(name)
    base = match.group("base") if match else None

    i = 2  # start with (2) for the first duplicate rather than (1)
    candidate = f"{base} ({i})"
    while candidate in existing:
        i += 1
        candidate = f"{base} ({i})"
    return candidate
