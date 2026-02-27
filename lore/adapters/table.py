"""
Simple adapter for parsing TSV files.
"""
import csv
import io
from typing import ClassVar
from lore.core.adapters import adapter_registry, TableAdapter


class CsvAdapter(TableAdapter):
    """
    Adapter for handling TSV, CSV files in LoRe Genome.
    """
    accepted_formats: ClassVar[set[str]] = {"tsv", "tab", "txt", "csv"}

    def parse(self, raw_data) -> list[dict]:
        """
        Parse TSV/CSV data into a list of dictionaries, keys take from header row
        """
        if not raw_data.strip():
            return []

        # 1. Mem-safe peek at first line to determine delimiter (tab or comma)
        first_line = raw_data[:2048].split("\n", 1)[0]
        delimiter = "," if "," in first_line and "\t" not in first_line else "\t"

        # 2. Built-in csv reader handles edge-cases
        f = io.StringIO(raw_data)
        reader = csv.DictReader(f, delimiter=delimiter, restval=None)

        return list(reader)


adapter_registry.register(CsvAdapter())
