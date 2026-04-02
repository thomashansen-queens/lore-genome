"""
Adapter for handling FASTA files in LoRe Genome. Currently only works for proteins.
"""
from typing import Any, Iterator

import lore.dsl as lore


@lore.adapter()
class FastaAdapter(lore.TableAdapter):
    """Adapter for FASTA files"""
    accepted_formats = {"fasta", "faa", "fa"}
    accepted_types = {"protein_fasta", "nucleotide_fasta", "fasta"}
    version = "1.0.0"

    @property
    def schema(self):
        return {
            "accession": "accession",
            "name": "name",
            "sequence": "sequence",
        }

    def _apply_schema(self, raw_record: dict) -> dict:
        """Dynamically applies the schema (including lambdas) to a parsed record."""
        adapted = {}
        for target_key, mapping in self.schema.items():
            if callable(mapping):
                # Execute the lambda function (e.g., mw_da, length)
                try:
                    adapted[target_key] = mapping(raw_record)
                except Exception:
                    adapted[target_key] = None
            elif isinstance(mapping, str):
                # Simple string key lookup
                adapted[target_key] = raw_record.get(mapping)
            else:
                adapted[target_key] = mapping
        return adapted

    def adapt(self, raw_data: Any, config: dict | None = None) -> list[dict]:
        """
        Handles blocks of text i.e. from a preview or a full read
        """
        if isinstance(raw_data, list) and len(raw_data) > 0 and isinstance(raw_data[0], str):
            # Use built-in stream adapter method for DRYness
            return list(self.adapt_stream(iter(raw_data), config))

        if isinstance(raw_data, str):
            return list(self.adapt_stream(iter(raw_data.splitlines()), config))

        return super().adapt(raw_data, config)

    def adapt_stream(self, raw_stream: Iterator[str], config: dict | None = None) -> Iterator[dict]:
        """
        Memory-efficient FASTA parser. Yields one record dict at a time so
        callers can handle files too large to load at once.
        """
        current: dict = {"accession": None, "name": None, "sequence": []}

        for line in raw_stream:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # write then clear buffer
                if current["accession"] is not None:
                    parsed_rec = {**current, "sequence": "".join(current["sequence"])}
                    yield self._apply_schema(parsed_rec)

                parts = line[1:].split(None, 1)
                if not parts:
                    parts = ["unknown_entry"]
                current = {
                    "accession": parts[0],
                    "name": parts[1] if len(parts) > 1 else None,
                    "sequence": [],
                }
            else:
                current["sequence"].append(line)

        # dump final buffer
        if current["accession"] is not None:
            parsed_rec = {**current, "sequence": "".join(current["sequence"])}
            yield self._apply_schema(parsed_rec)

    def serialize(self, records: list[dict], **kwargs) -> str:
        """Converts parsed records back into a FASTA string."""
        lines = []
        for r in records:
            header = f">{r['accession']}"
            if r.get("name"):
                header += f" {r['name']}"
            lines.append(header)
            lines.append(r["sequence"])
        return "\n".join(lines) + "\n"

# --- Specialized FASTA adapter for computations ---

# Average molecular masses (Da)
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
        return round(
            sum(_AA_MW[aa] for aa in seq.upper() if aa in _AA_MW) - _WATER * (len(seq) - 1),
            2,
        )
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


def _net_charge(seq: str, pH: float, pka: dict) -> float:
    """Uses Henderson-Hasselbalch equation to calculate net charge of a peptide at a given pH."""
    charge = 0.0
    # N-terminus
    charge += 1.0 / (1.0 + 10 ** (pH - pka["n"]))
    # C-terminus
    charge -= 1.0 / (1.0 + 10 ** (pka["c"] - pH))
    # Internal residues
    for aa in seq:
        if aa in _ACIDIC and aa in pka:
            charge -= 1.0 / (1.0 + 10 ** (pka[aa] - pH))
        elif aa in _BASIC and aa in pka:
            charge += 1.0 / (1.0 + 10 ** (pH - pka[aa]))
    return charge


def _isoelectric_point(seq: str, pka: dict = _PKA_TOSELAND) -> float | None:
    """
    Toseland et al. 2006 (https://doi.org/10.1093/nar/gkj035) is the most accurate pKa dataset, 
    per Kozlowski 2021 (https://doi.org/10.1093/nar/gkab295).

    NOTE: Audain et al. 2016 (https://doi.org/10.1093/bioinformatics/btv674) prefers numbers from 
    Bjellqvist et al. 1993 (https://doi.org/10.1002/elps.11501401163) and Bjellqvist et al. 1994 
    (https://doi.org/10.1002/elps.1150150171), which is also what ExPaSy uses.
    """
    seq = seq.upper()
    lo, hi = 0.0, 14.0
    for _ in range(20):  # binary search for pI
        mid = (lo + hi) / 2
        if _net_charge(seq, mid, pka) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


# Extinction coefficient calculations

def _extinction_coefficient(seq: str, reduced: bool = True) -> float | None:
    """
    Extinction coefficient at 280 nm per Pace et al. 1995.
    https://doi.org/10.1002/pro.5560041120
    ε280 M-1cm-1 = (W x 5500) + (Y x 1490) + (C x 125)
    """
    seq = seq.upper()
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
        scores = [_KD[aa] for aa in seq.upper() if aa in _KD]
        return round(sum(scores) / len(scores), 4) if scores else None
    except Exception:
        return None


_AROMATIC = {"F", "W", "Y"}


def _aromaticity(seq: str) -> float | None:
    """Calculate the proportion of aromatic residues in a peptide sequence."""
    try:
        s = seq.upper()
        return round(sum(1 for aa in s if aa in _AROMATIC) / len(s), 3)
    except Exception:
        return None


@lore.adapter()
class ProtParamAdapter(FastaAdapter):
    """Protein physicochemical properties from FASTA sequences"""
    accepted_types = {"protein_fasta"}
    view_mode = "table"
    version = "1.0.0"

    @property
    def schema(self):
        return {
            "accession": "accession",
            "name": "name",
            "length": lambda x: len(x["sequence"]),
            "mw_da": lambda x: _molecular_weight(x["sequence"]),
            "isoelectric_point": lambda x: _isoelectric_point(x["sequence"]),
            "extinction_coefficient": lambda x: _extinction_coefficient(x["sequence"]),
            "gravy": lambda x: _gravy(x["sequence"]),
            "cysteines": lambda x: x["sequence"].count("C"),
            "aromaticity": lambda x: _aromaticity(x["sequence"]),
        }
