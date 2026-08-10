"""
Universal length resolution.
    12, 12.0        -> 12 px (absolute)
    "12", "12px"    -> 12 px (absolute)
    "50%"           -> 50% of a supplied reference total
"""

Length = float | int | str


def resolve_px(value: Length, total: float = 0.0) -> float:
    """
    Resolve a length to absolute pixels.

    `total` is the reference extent a percentage is measured against (e.g. the
    available track width); it is ignored for absolute values.
    """
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    if text.endswith("%"):
        return total * float(text[:-1].strip()) / 100.0
    if text.endswith("px"):
        return float(text[:-2].strip())
    return float(text.strip())  # bare numeric string
