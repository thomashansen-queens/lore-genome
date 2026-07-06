"""
Methods for generating display-ready text strings.
"""
from collections.abc import Iterable


def _char_width(font_size: int, is_monospace: bool = True) -> float:
    """Estimate the px width of a single character"""
    return font_size * (0.6 if is_monospace else 0.5)


def estimate_width(text: str, font_size: int, is_monospace: bool = True) -> float:
    """Estimate the px width of a string. Assume monospace is ~0.6 * size"""
    return len(text) * _char_width(font_size, is_monospace)


def widest(
    strings: Iterable[str],
    font_size: int,
    is_monospace: bool = True,
    default: float = 0.0,
) -> float:
    """Estimated width in px of the widest string (0.0 if empty)."""
    return max(
        (estimate_width(s, font_size, is_monospace) for s in strings),
        default=default,
    )


def truncate_to_fit(text: str, max_width: float, font_size: int, is_monospace: bool = True) -> str:
    """Truncate text to fit within max_width, adding ellipsis if needed"""
    max_chars = int(max_width / _char_width(font_size, is_monospace))
    if max_chars < 3:
        # Not enough space for any text
        return ""
    if len(text) > max_chars:
        return text[:max_chars - 1] + "…"
    return text
