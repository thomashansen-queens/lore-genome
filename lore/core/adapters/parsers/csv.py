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
        kwargs["delimiter"] = "\t" if extension in ("tsv") else ","

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

        # 3. Header handling
        if kwconfig.get("header", True) in (False, None) and "fieldnames" not in csv_kwargs:
            # Peek at first row to count columns
            first_row = next(csv.reader([raw_data[0]], **csv_kwargs))
            csv_kwargs["fieldnames"] = [f"column_{i}" for i in range(len(first_row))]

        dict_reader = csv.DictReader(raw_data, **csv_kwargs)
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
        # TODO: Should I decide with utf-8-sig here as well to handle BOM in streaming cases?
        # It would add complexity to already-expensive streaming handling
        kwconfig = self._prepare_config(config, **kwargs)
        ext = kwconfig.get("ext", "")
        csv_kwargs = self._prepare_csv_kwargs(kwconfig, ext)

        # Header handling
        if kwconfig.get("header", True) in (False, None) and "fieldnames" not in csv_kwargs:
            import itertools
            stream_peek, raw_stream = itertools.tee(raw_stream)
            try:
                first_line = next(stream_peek)
                first_row = next(csv.reader([first_line], **csv_kwargs))
                csv_kwargs["fieldnames"] = [f"column_{i}" for i in range(len(first_row))]
            except StopIteration:
                # Empty stream, return without yielding
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
