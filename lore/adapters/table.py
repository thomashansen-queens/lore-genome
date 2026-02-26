"""
Simple adapter for parsing TSV files.
"""
import csv
import io
from lore.core.adapters import adapter_registry, BaseAdapter


class TableAdapter(BaseAdapter):
    """
    Adapter for handling TSV, CSV files in LoRe Genome.
    """
    accepted_formats = {"tsv", "tab", "txt", "csv"}

    def parse(self, raw_data) -> list[dict]:
        """
        Parse TSV data into a list of dictionaries, where each dictionary 
        represents a row with column headers as keys.
        """
        if not raw_data.strip():
            return []

        # 1. Peek at first line to determine delimiter (tab or comma)
        first_line = raw_data.splitlines()[0]
        delimiter = "," if "," in first_line and "\t" not in first_line else "\t"

        # 2. Built-in csv reader handles edge-cases
        reader = list(csv.reader(io.StringIO(raw_data), delimiter=delimiter))
        if not reader:
            return []

        first_row = reader[0]

        # 3. Build dicts
        records = []
        headers = first_row
        data_rows = reader[1:]

        for row in data_rows:
            if len(row) != len(headers):
                # Rows with missing or extra columns: fill None or ignore extra
                row = row + [None] * (len(headers) - len(row)) if len(row) < len(headers) else row[:len(headers)]
            record = dict(zip(headers, row))
            records.append(record)

        return records


adapter_registry.register(TableAdapter())
