"""
Tools for web-based interfaces with Arifacts in LoRe Genome.
"""

import json
from pathlib import Path

def format_preview_text(raw_content: str, filename: str) -> str:
    """
    Formats raw text for display. 
    - Pretty-prints JSON if possible.
    - Falls back to 'JSON-ish' formatter if invalid.
    - Returns raw text otherwise.
    """
    ext = Path(filename).suffix.lower()
    
    if ext in {".json", ".jsonl", ".ndjson"}:
        try:
            # Valid JSON? Pretty print it.
            json_obj = json.loads(raw_content)
            return json.dumps(json_obj, indent=2)
        except json.JSONDecodeError:
            # Invalid/Truncated? Best-effort pretty-print
            return jsonish_pretty_preview(raw_content)
            
    return raw_content


def jsonish_pretty_preview(text: str, indent_step: int = 2, max_len: int | None = None) -> str:
    """
    Best-effort pretty formatting for (possibly truncated) JSON text.

    - Indents/brackets/commas/colons only when NOT inside a JSON string.
    - Tracks escapes so quotes inside strings don't flip state.
    - NOTE: Does NOT guarantee valid JSON.
    """
    if max_len is not None:
        text = text[:max_len]

    out: list[str] = []
    indent = 0
    in_str = False
    escape = False
    at_line_start = True

    def emit(s: str) -> None:
        nonlocal at_line_start
        out.append(s)
        if s:
            at_line_start = s.endswith("\n")

    def newline_and_indent() -> None:
        emit("\n")
        emit(" " * indent)

    # Helper: look ahead for next non-whitespace char
    def next_non_ws(start: int) -> str | None:
        j = start
        while j < len(text) and text[j] in " \t\r\n":
            j += 1
        return text[j] if j < len(text) else None

    # Main loop
    i = 0
    while i < len(text):
        ch = text[i]

        if in_str:
            emit(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        # not in string
        if ch in " \t\r\n":
            # collapse horizontal whitespace outside strings
            i += 1
            continue

        if ch == '"':
            in_str = True
            emit(ch)
            i += 1
            continue

        if ch in "{[":
            emit(ch)
            # Lookahead for empty object/array; don't add a newline/indent.
            nxt = next_non_ws(i + 1)
            if (ch == "{" and nxt == "}") or (ch == "[" and nxt == "]"):
                i += 1
                continue
            indent += indent_step
            newline_and_indent()
            i += 1
            continue

        if ch in "}]":
            indent = max(0, indent - indent_step)
            if not at_line_start:
                newline_and_indent()
            emit(ch)
            i += 1
            continue

        if ch == ",":
            emit(",")
            newline_and_indent()
            i += 1
            continue

        if ch == ":":
            emit(": ")
            i += 1
            continue

        # numbers, true/false/null, etc. could be here
        emit(ch)
        i += 1

    return "".join(out)
