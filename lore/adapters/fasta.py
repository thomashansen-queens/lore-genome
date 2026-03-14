"""
Adapter for handling FASTA files in LoRe Genome. Currently only works for proteins.
"""
from typing import Any, Iterator

from lore.core.adapters import adapter_registry, TableAdapter


class FastaAdapter(TableAdapter):
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

# Monoisotopic masses (Da)
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
        return round(sum(_AA_MW[aa] for aa in seq.upper() if aa in _AA_MW) - _WATER * (len(seq) - 1), 2)
    except Exception:
        return None

# Kyte-Doolittle hydrophobicity scale
# Citation: https://doi.org/10.1016/0022-2836(82)90515-0 from 1982!
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

def _gravy(seq: str) -> float | None:
    """GRand AVerage of hydropathY (GRAVY) from Kyte-Doolittle scale."""
    try:
        scores = [_KD[aa] for aa in seq.upper() if aa in _KD]
        return round(sum(scores) / len(scores), 4) if scores else None
    except Exception:
        return None

_AROMATIC = {"F", "W", "Y"}

def _aromaticity(seq: str) -> float | None:
    """Calculate the aromaticity of a peptide sequence."""
    try:
        s = seq.upper()
        return round(sum(1 for aa in s if aa in _AROMATIC) / len(s), 4)
    except Exception:
        return None

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
            # "isoelectric_point": lambda x: _isoelectric_point(x["sequence"]),
            "gravy": lambda x: _gravy(x["sequence"]),
            "cysteines": lambda x: x["sequence"].count("C"),
            "aromaticity": lambda x: _aromaticity(x["sequence"]),
        }


adapter_registry.register(FastaAdapter())
adapter_registry.register(ProtParamAdapter())
