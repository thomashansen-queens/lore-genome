"""
Generic CSV/TSV adapter.
"""
import csv
import io
from typing import Any, ClassVar, Iterator

from ..tabular import TabularAdapter


class CsvAdapter(TabularAdapter):
    """
    Adapter for CSV/TSV files. Uses Python's built-in csv library for parsing.
    """
    accepted_formats: ClassVar[set[str]] = {"csv", "tsv"}
    accepted_types: ClassVar[set[str]] = {"*"}
    view_mode: ClassVar[str] = "table"
    version: ClassVar[str] = "1.0.0"

    # Args for csv.DictReader and csv.DictWriter can be passed via config
    CSV_KWARGS = frozenset({
        "fieldnames", "delimiter", "quotechar", "escapechar", "doublequote",
        "skipinitialspace", "lineterminator", "quoting", "strict",
    })

    def _prepare_csv_kwargs(self, config: dict, extension: str) -> dict:
        """Helper to extract valid csv formatting kwargs from a config dict."""
        kwargs = {}
        # Default delimiter fallback based on extension
        kwargs["delimiter"] = "\t" if extension in ["tsv"] else ","

        # Alias for columns -> fieldnames (will be overridden by explicit fieldnames if both)
        if "columns" in config:
            kwargs["fieldnames"] = config.get("columns")

        # Override with explicit config kwargs
        for key in self.CSV_KWARGS:
            if key in config:
                kwargs[key] = config[key]
        return kwargs

    def parse(self, raw_data: Any, config: dict | None = None, **kwargs) -> list[dict]:
        """
        Losslessly parses a CSV/TSV into a list of dicts.
        """
        if not raw_data:
            return []

        kwconfig = self._prepare_config(config, **kwargs)

        # 1a. Monolithic file reads
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode("utf-8-sig")
        if isinstance(raw_data, str):
            raw_data = raw_data.strip().splitlines()

        # 1b. Already parsed? Return as-is
        if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict):
            return raw_data

        # 2. Instructions from config to DictReader
        ext = kwconfig.get("ext", "")
        csv_kwargs = self._prepare_csv_kwargs(kwconfig, ext)

        # 3. Header handling: If headerless and no name supplied
        has_header = kwconfig.get("header", True)
        if not has_header and "fieldnames" not in csv_kwargs:
            # Peek at first row to count columns
            first_row = next(csv.reader([raw_data[0]], **csv_kwargs))
            csv_kwargs["fieldnames"] = [f"column_{i}" for i in range(len(first_row))]

        # 4. Skip the file's header row only when explicit fieldnames override it.
        # Use iterator rather than slicing (raw_data[1:]) to avoid copying large data
        data_iterator = iter(raw_data)
        if has_header and "fieldnames" in csv_kwargs:
            try:
                next(data_iterator)
            except StopIteration:
                pass

        dict_reader = csv.DictReader(data_iterator, **csv_kwargs)
        return list(dict_reader)

    def parse_stream(
        self,
        raw_stream: Iterator[str],
        config: dict | None = None,
        **kwargs,
    ) -> Iterator[dict]:
        """
        Yields parsed CSV records from an input text stream
        """
        kwconfig = self._prepare_config(config, **kwargs)
        ext = kwconfig.get("ext", "")
        csv_kwargs = self._prepare_csv_kwargs(kwconfig, ext)

        # 1. Header handling
        has_header = kwconfig.get("header", True)
        if has_header in (False, None) and "fieldnames" not in csv_kwargs:
            import itertools
            stream_peek, raw_stream = itertools.tee(raw_stream)
            try:
                first_line = next(stream_peek)
                first_row = next(csv.reader([first_line], **csv_kwargs))
                csv_kwargs["fieldnames"] = [f"column_{i}" for i in range(len(first_row))]
            except StopIteration:
                # Empty stream, return without yielding
                return

        # 2. Skip header line if needed
        if has_header and "fieldnames" in csv_kwargs:
            try:
                next(raw_stream)  # Skip header line
            except StopIteration:
                # Stream had only header, return without yielding
                return

        dict_reader = csv.DictReader(raw_stream, **csv_kwargs)
        yield from dict_reader

    # --- Output methods ---

    def serialize(self, records: list[dict], config: dict | None = None, **kwargs) -> str:
        """
        Reverses the adapt process: takes a list of dicts and outputs a CSV/TSV string.
        """
        if not records:
            return ""

        kwconfig = self._prepare_config(config, **kwargs)
        ext = kwconfig.get("ext", "")
        csv_kwargs = self._prepare_csv_kwargs(kwconfig, ext)

        # Remap column names if passed via config, otherwise use first record
        if "fieldnames" not in csv_kwargs:
            csv_kwargs["fieldnames"] = list(records[0].keys())

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            extrasaction="ignore",
            **csv_kwargs,
        )

        # Optionally allow config to disable header writing (default: write header)
        if kwconfig.get("write_header", True):
            writer.writeheader()

        writer.writerows(records)

        return output.getvalue()
