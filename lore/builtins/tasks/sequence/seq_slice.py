"""
Utility to slice, dice, trim, and anchor FASTA sequences.
"""
import re
import lore


class SeqSliceInputs:
    """Inputs for Sequence Slicer Task"""
    sequences = lore.ArtifactInput(
        accepted_data=["fasta", "protein_fasta", "nucleotide_fasta"],
        select="multiple",
        load_as="adapted",
        label="Sequences",
    )
    start = lore.ValueInput(
        str | None, 
        default=None,
        label="Start Position",
        description="1-based index (e.g., 50), Pythonic negative index (e.g., -800), or a sequence motif (e.g., LPXTG).",
    )
    end = lore.ValueInput(
        str | None, 
        default=None,
        label="End Position",
        description="1-based index, negative index, or a sequence motif. Leave blank to slice to the end.",
    )
    keep_remainder = lore.ValueInput(
        bool,
        default=False,
        label="Keep Remainder",
        description="If checked, the discarded portion of each sequence will be returned as a separate output.",
    )


class SeqSliceOutputs:
    """Outputs for Sequence Slicer Task"""
    sliced_fasta = lore.TaskOutput(
        data_type=lore.Passthrough("sequences"),
        label="Sliced Sequences",
        is_primary=True,
    )
    remainder_fasta = lore.TaskOutput(
        data_type=lore.Passthrough("sequences"),
        label="Remainder (Discarded) Sequences",
        is_primary=False,
    )


def _resolve_index(val: str | None, seq: str, is_start: bool) -> int:
    """Parses the input as an integer or a motif string to find the correct index."""
    if not val:
        return 0 if is_start else len(seq)

    val_str = str(val).strip()

    # 1. Try parsing as an integer
    try:
        idx = int(val_str)
        if idx > 0:
            return idx - 1  # Convert biological 1-based to Python 0-based
        elif idx < 0:
            return len(seq) + idx  # Pythonic negative indexing
        else:
            return 0
    except ValueError:
        pass  # It's a string/motif

    # 2. Parse as a Motif/Regex
    # Convert biological "X" to regex "."
    pattern_str = val_str.upper().replace("X", ".")

    try:
        match = re.search(pattern_str, seq.upper())
        if match:
            # If it's the start, slice begins at the start of the motif.
            # If it's the end, slice includes the motif by ending at the end of the match.
            return match.start() if is_start else match.end()
        else:
            return -1 # Motif not found
    except re.error as e:
        raise ValueError(f"Invalid regex/motif pattern '{val_str}': {e}")


def _extract_fasta_data(record: dict | str) -> tuple[str, str]:
    """
    Extracts the header and sequence from an adapted FASTA dict or raw string.
    Returns a tuple of (header, sequence).
    """
    # Fallback for raw text blocks
    if isinstance(record, str):
        record_str = record.strip()
        if not record_str.startswith(">"):
            return "", ""
        lines = record_str.splitlines()
        header = lines[0]
        seq = "".join(l.strip() for l in lines[1:])
        return header, seq

    # Duck-typing for Adapted Dictionaries
    if isinstance(record, dict):
        # 1. Sniff the Accession/ID
        acc = next((str(v) for k, v in record.items() if k.lower().endswith(("accession", "id", "acc", "name"))), "Unknown")

        # 2. Sniff the Description (optional)
        desc = next((str(v) for k, v in record.items() if k.lower().endswith(("description", "desc"))), "")

        # 3. Sniff the Sequence (catches "seq", "sequence", "protein_sequence", etc.)
        seq = next((str(v) for k, v in record.items() if k.lower().endswith(("seq", "sequence"))), "")

        if not seq:
            raise ValueError(f"Could not locate sequence data in record keys: {list(record.keys())}")

        header = acc if acc.startswith(">") else f">{acc}"
        header += f" {desc}" if desc else ""

        return header, seq

    return "", ""


@lore.task(
    "sequence.slice",
    inputs=SeqSliceInputs,
    outputs=SeqSliceOutputs,
    name="Slice & Trim Sequences",
    category="Sequence Utilities",
    preview_mode="full",
    icon="AT✂️CG",
)
def sequence_slice(
    ctx: lore.ExecutionContext,
    sequences: list[str],
    start: str | None = None,
    end: str | None = None,
    keep_remainder: bool = False,
):
    """
    Slices a batch of FASTA sequences using coordinates or sequence motifs.
    """
    if not sequences:
        raise ValueError("No sequences provided to slice.")

    sliced_records = []
    remainder_records = []

    for record in sequences:
        # 1. Handle Adapted dictionaries vs raw strings
        header, seq = _extract_fasta_data(record)
        if not seq:
            ctx.logger.warning(f"No sequence data found in record: {record}. Skipping.")
            continue

        # 2. Resolve indices
        start_idx = _resolve_index(start, seq, is_start=True)
        end_idx = _resolve_index(end, seq, is_start=False)

        # Handle unfound motifs
        if start_idx == -1 or end_idx == -1:
            ctx.logger.warning(f"Motif not found in {header.split()[0]}. Skipping sequence.")
            continue

        # Bounds safety
        start_idx = max(0, start_idx)
        end_idx = min(len(seq), max(start_idx, end_idx))

        # Perform the slice
        ctx.logger.debug(f"Slicing {header.split()[0]}: start={start_idx}, end={end_idx}")
        target_seq = seq[start_idx:end_idx]
        remainder_seq = seq[:start_idx] + seq[end_idx:]

        def _format_fasta(head: str, s: str, suffix: str) -> str:
            new_head = f"{head} | {suffix}"
            formatted_seq = "\n".join(s[i:i+80] for i in range(0, len(s), 80))
            return f"{new_head}\n{formatted_seq}"

        if target_seq:
            sliced_records.append(_format_fasta(header, target_seq, f"sliced:{start_idx}-{end_idx}"))
        if keep_remainder and remainder_seq:
            remainder_records.append(_format_fasta(header, remainder_seq, f"remainder_of:{start_idx}-{end_idx}"))

    if not sliced_records:
        raise RuntimeError("Slicing resulted in an empty dataset. Check your coordinates/motifs.")

    # Materialize
    ctx.logger.info(f"Successfully sliced {len(sliced_records)} sequences.")

    ctx.materialize_content(
        content="\n".join(sliced_records) + "\n",
        output_key="sliced_fasta",
        name="sliced",
        extension="fasta",
    )

    if keep_remainder and remainder_records:
        ctx.materialize_content(
            content="\n".join(remainder_records) + "\n",
            output_key="remainder_fasta",
            name="remainder",
            extension="fasta",
        )
