"""
Adapter for handling FASTA files in LoRē Genome.
"""
from typing import Any, ClassVar, Iterator

import lore
import numpy as np


class BaseFastaAdapter(lore.TabularAdapter):
    """
    Adapter for NCBI-style FASTA files with three part headers:
    >accession description sequence
    For example:
    ...

    kwargs:
    - line_length (int | None): Wraps sequences to a specified line length

    TODO: Support other FASTA header formats (e.g. UniProt's pipe-separated)
    """
    accepted_formats = {"fasta", "faa", "fa", "fna", "ffn"}
    version = "1.0.0"

    @property
    def schema(self):
        return {
            "accession": "accession",
            "description": "description",
            "sequence": "sequence",
        }

    def parse(self, raw_data: Any, config: dict | None = None, **kwargs) -> list[dict]:
        """
        Parses raw FASTA data into a list of records with 'accession', 'description',
        and 'sequence' fields.
        """
        if not raw_data:
            return []
        kwconfig = self._prepare_config(config, **kwargs)

        # 1. Decode and read as lines
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode("utf-8-sig")
        if isinstance(raw_data, str):
            return list(self.parse_stream(iter(raw_data.splitlines()), kwconfig))

        # 2. Preview mode: Peeks at first n lines
        if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], str):
            return list(self.parse_stream(iter(raw_data), kwconfig))

        # 3. Already parsed? Return as-is
        if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict):
            return raw_data

        return []

    # --- Lossless parser ---

    def parse_stream(
        self,
        raw_stream: Iterator[str],
        config: dict | None = None,
        **kwargs,
    ) -> Iterator[dict]:
        """
        Memory-efficient FASTA parser. Yields one dict record at a time, with
        'accession', 'description', and 'sequence' fields. Suitable for large
        files.
        """
        kwconfig = self._prepare_config(config, **kwargs)

        # 1. Initialize state for schema
        # TODO: Accept other schemas and dynamically adapt?
        current_accession = None
        current_desc = None
        seq_buffer = []

        # 2. Stream through lines, concatenating sequence until next header or EOF
        for line in raw_stream:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # write then clear buffer
                if current_accession is not None:
                    yield {
                        "accession": current_accession,
                        "description": current_desc,
                        "sequence": "".join(seq_buffer),
                    }

                # Parse new header
                parts = line[1:].split(None, 1)
                current_accession = parts[0] if parts else "unknown_entry"
                current_desc = parts[1] if len(parts) > 1 else None
                seq_buffer = []
            else:
                seq_buffer.append(line)

        # Yield the final buffer on EOF
        if current_accession is not None:
            yield {
                "accession": current_accession,
                "description": current_desc,
                "sequence": "".join(seq_buffer),
            }

    # --- Lossless serialization ---

    def serialize(self, records: list[dict], config: dict | None = None, **kwargs) -> str:
        """Converts parsed records back into a FASTA string."""
        if not records:
            return ""

        kwconfig = self._prepare_config(config, **kwargs)
        line_length = kwconfig.get("line_length", 80)

        lines = []
        for r in records:
            # 1. Rebuild header
            header = f">{r.get('accession', 'unknown_entry')}"
            description = r.get("description")
            if description:
                header += f" {description}"
            lines.append(header)

            # 2. Rebuild sequence with optional line wrapping
            seq = r.get("sequence", "")
            if line_length and line_length > 0:
                lines.extend(seq[i:i+line_length] for i in range(0, len(seq), line_length))
            else:
                lines.append(seq)

        return "\n".join(lines)

    def serialize_stream(
        self,
        records_stream: Iterator[dict],
        config: dict | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Memory-efficient serializer. Yields one record at a time. Suitable for
        very large files.
        """
        kwconfig = self._prepare_config(config, **kwargs)
        line_length = kwconfig.get("line_length", 80)

        for r in records_stream:
            # 1. Rebuild header
            header = f">{r.get('accession', 'unknown_entry')}"
            description = r.get("description")
            if description:
                header += f" {description}"

            # 2. Rebuild sequence with optional line wrapping
            seq = r.get("sequence", "")
            if line_length and line_length > 0:
                wrapped_seq = "\n".join(
                    seq[i:i+line_length] for i in range(0, len(seq), line_length)
                )
            else:
                wrapped_seq = seq

            # 3. Yield the entire block at once
            yield f"{header}\n{wrapped_seq}\n"

# --- Fasta adapters by domain (protein, DNA, RNA) ---
# These are just thin wrappers for types and schemas

@lore.adapter()
class ProteinFastaAdapter(BaseFastaAdapter):
    """Semantic wrapper for protein sequences"""
    accepted_types: ClassVar[set[str]] = {"protein_fasta"}

    @property
    def schema(self):
        return {
            "protein_accession": "accession",
            "protein_description": "description",
            "protein_sequence": "sequence",
        }


@lore.adapter()
class NucleotideFastaAdapter(BaseFastaAdapter):
    """Semantic wrapper for nucleotide sequences"""
    accepted_types: ClassVar[set[str]] = {"nucleotide_fasta"}

    @property
    def schema(self):
        return {
            "nucleotide_accession": "accession",
            "nucleotide_description": "description",
            "nucleotide_sequence": "sequence",
        }


# --- Specialized FASTA adapter for computations ---
# This essentially applies ExPASy's ProtParam calculations on the fly to a 
# FASTA file. Their tool is avilable at https://web.expasy.org/protparam/

# Average molecular masses (Da)
# These values are for hydrated amino acids; for peptide MW calculation,
# the mass of water is substracted for each bond formed.
# NOTE: These are average masses, not monoisotopic, in case that matters!
_AA_MW = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "E": 147.13, "Q": 146.15, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}
_WATER = 18.02


def _molecular_weight(seq: str) -> float | None:
    """Calculate peptide MW, subtracting one H2O per peptide bond."""
    try:
        hydrated_mass = sum(seq.count(aa) * mass for aa, mass in _AA_MW.items())
        return round(hydrated_mass - _WATER * (len(seq) - 1), 2)
    except Exception:
        return None


# Isoelectric point (pI) calculations
_PKA_TOSELAND = {
    "D": 3.60, "E": 4.29, "H": 6.33, "C": 6.87, "Y": 9.61, "K": 10.45, "R": 12.0,
    "n": 8.71, "c": 3.19,
}
_PKA_BJELLQVIST = {
    "nA": 7.59, "nM": 7.00, "nS": 6.93, "nP": 8.36, "nT": 6.82, "nV": 7.44, "nE": 7.70,
    "D": 4.05, "E": 4.45, "H": 5.98, "C": 9.0, "Y": 10.0, "K": 10.0, "R": 12.0,
    "cD": 4.55, "cE": 4.75,
    "n": 7.50, "c": 3.55,
}
_ACIDIC = {"D", "E", "C", "Y"}
_BASIC  = {"H", "K", "R"}


def _isoelectric_point(seq: str, pka: dict = _PKA_TOSELAND) -> float | None:
    """
    Binary search for the pH at which net charge is zero (isoelectric point).
    Uses Henderson-Hasselbalch equation to calculate net charge of a peptide
    at a given pH. Returns early on convergence.

    Uses values from:
    Toseland et al. 2006 (https://doi.org/10.1093/nar/gkj035) which is the most accurate pKa
    dataset, per Kozlowski 2021 (https://doi.org/10.1093/nar/gkab295).

    NOTE: Audain et al. 2016 (https://doi.org/10.1093/bioinformatics/btv674) prefers numbers from 
    Bjellqvist et al. 1993 (https://doi.org/10.1002/elps.11501401163) and Bjellqvist et al. 1994 
    (https://doi.org/10.1002/elps.1150150171), which is also what ExPaSy uses.
    """
    counts = {aa: seq.count(aa) for aa in _ACIDIC | _BASIC}

    lo, hi = 0.0, 14.0

    # Reaches 14 / 2^i precision after i iterations
    # Using 14 iterations reaches 0.0008 pH precision, which is more than
    # enough for practical purposes. That this is 14 iterations is a 
    # coincidence and is not related to the pH range of 0-14.
    prev = -1
    for _ in range(14):
        ph = (lo + hi) / 2.0

        charge = 0.0
        charge += 1.0 / (1.0 + 10 ** (ph - pka["n"]))  # N-terminus
        charge -= 1.0 / (1.0 + 10 ** (pka["c"] - ph))  # C-terminus

        for aa in _ACIDIC:
            if counts[aa]:
                charge -= counts[aa] / (1.0 + 10 ** (pka[aa] - ph))
        for aa in _BASIC:
            if counts[aa]:
                charge += counts[aa] / (1.0 + 10 ** (ph - pka[aa]))

        # Update binary search bounds
        if charge > 0:
            lo = ph
        else:
            hi = ph

        # Early stop on convergence
        if abs(ph - prev) < 0.001:
            break
        prev = ph

    return round(ph, 4)


# Extinction coefficient calculations

def _extinction_coefficient(seq: str, reduced: bool = True) -> float | None:
    """
    Extinction coefficient at 280 nm per Pace et al. 1995.
    https://doi.org/10.1002/pro.5560041120
    ε280 M-1cm-1 = (W x 5500) + (Y x 1490) + (C x 125)
    """
    return round(
        seq.count("W") * 5500 + 
        seq.count("Y") * 1490 + 
        (0 if reduced else seq.count("C") * 125),
        2,
    )


# Kyte-Doolittle hydropathy index (hydrophobicity)
_KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, 
    "H": -3.2, "D": -3.5, "E": -3.5, "N": -3.5, "Q": -3.5, "K": -3.9, "R": -4.5,
}


def _gravy(seq: str) -> float | None:
    """
    GRand AVerage of hydropathY (GRAVY) from Kyte-Doolittle scale. Uses values from
    Kyte & Doolittle 1982 (https://doi.org/10.1016/0022-2836(82)90515-0). Higher is hydrophobic, 
    lower is hydrophilic.
    """
    try:
        score = sum(seq.count(aa) * val for aa, val in _KD.items())
        valid_aas = sum(seq.count(aa) for aa in _KD)  # Excludes unknown AA codes

        return round(score / valid_aas, 4) if valid_aas > 0 else None
    except Exception:
        return None


_AROMATIC = {"F", "W", "Y"}


def _aromaticity(seq: str) -> float | None:
    """Calculate the proportion of aromatic residues in a peptide sequence."""
    aromatics = sum(seq.count(aa) for aa in _AROMATIC)
    try:
        return round(aromatics / len(seq), 3)
    except Exception:
        return None


@lore.adapter()
class ProtParamAdapter(ProteinFastaAdapter):
    """
    Protein physicochemical properties from FASTA sequences. Obviously does
    not work for nucleotide FASTA files.

    This essentially applies ExPASy's ProtParam calculations on the fly to a 
    FASTA file. Their tool is avilable at https://web.expasy.org/protparam/
    
    Please see comments in the code for details on specific calculations and 
    subtle differences between this and ExPASy's implementation (e.g. choice 
    of pKa values for isoelectric point).
    """
    accepted_types = {"protein_fasta"}
    view_mode = "table"
    version = "1.0.0"

    @property
    def schema(self):
        return {
            "accession": "accession",
            "description": "description",
            "length": lambda x: len(x["sequence"]),
            "mw_da": lambda x: _molecular_weight(x["sequence"].upper()),
            "isoelectric_point": lambda x: _isoelectric_point(x["sequence"].upper()),
            "extinction_coefficient": lambda x: _extinction_coefficient(x["sequence"].upper()),
            "gravy": lambda x: _gravy(x["sequence"].upper()),
            "cysteines": lambda x: x["sequence"].count("C"),
            "aromaticity": lambda x: _aromaticity(x["sequence"].upper()),
        }
