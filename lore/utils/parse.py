"""
Utility functions to parse and coerce values from various sources.
"""
from typing import Any
import re

# [^\w\-.]: Anything that isn't a word character [a-zA-Z0-9_] or - .
# [\\/?@]: No slashes, question marks, at signs (more permissive)
_DEFAULT_BAD_CHARS = re.compile(r'[\\/?@]')

def clean(v: Any, *, empty_to_none: bool = True,
          reject: re.Pattern | None = _DEFAULT_BAD_CHARS) -> str | None:
    """Helper cleans strings."""
    if v is None:
        return None
    if not isinstance(v, str):
        v = str(v)
    s = v.strip()
    if empty_to_none and s == "":
        return None
    if reject and reject.search(s):
        raise ValueError(f"Invalid characters: {s}")
    return s

def as_list_str(v: Any) -> list[str] | None:
    """Helper converts comma-separated strings or lists to cleaned list[str]."""
    if v is None:
        return None
    if isinstance(v, list):
        out = []
        for x in v:
            s = clean(x)
            if s is not None:
                out.append(s)
        return out or None
    # allow comma-separated strings from manual edits / simple forms
    if isinstance(v, str):
        out = [clean(s) for s in v.split(",")]
        out = [s for s in out if s is not None]
        return out or None
    raise ValueError(f"Expected list[str] or comma-string, got {type(v).__name__}")

def as_bool(v: Any) -> bool | None:
    """Helper converts various values to bool."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {'1', 'true', 'yes', 'on'}:
            return True
        if s in {'0', 'false', 'no', 'off', 'null'}:
            return False
    if isinstance(v, (int, float)) and v in {0, 1}:
        return v != 0
    if isinstance(v, (list, dict)):
        return bool(v)
    raise ValueError(f"Expected bool-like value, got {type(v).__name__}")

def as_int(v: Any) -> int | None:
    """Helper converts various values to int."""
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return int(s)
    if isinstance(v, float) or isinstance(v, bool):
        return int(v)
    raise ValueError(f"Expected int-like value, got {type(v).__name__}")

# --- FASTA ---

def fasta_lookup(
    targets: list[str] | set[str],
    fasta_path: str,
    substring_matches: bool = False,
) -> dict[str, str]:
    """
    Memory safe streaming of FASTA file to pluck out dict of {header: sequence} 
    for the input list of accessions.
    """
    extracted_seqs = {}

    # Faster function if strict matching
    if not substring_matches:
        targets = set(targets)

    with open(fasta_path, "r", encoding="utf-8") as f:
        seq_key = None
        seq_parts = []
        capture_current = False

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save the previous sequence if it was one we wanted
                if capture_current and seq_key is not None:
                    extracted_seqs[seq_key] = "".join(seq_parts)

                # Extract the new accession
                seq_parts = []
                header_content = line[1:]
                if substring_matches:
                    capture_current = any(t in header_content for t in targets)
                    seq_key = header_content
                else:
                    accession = header_content.split()[0]
                    capture_current = accession in targets
                    seq_key = accession

            elif capture_current:
                seq_parts.append(line)

        # Catch the tail end
        if capture_current and seq_key is not None:
            extracted_seqs[seq_key] = "".join(seq_parts)

    return extracted_seqs
