"""
Generic JSON / list[dict] adapter.
This adapter will attempt to parse JSON into a list of records. For specific
schemas, subclass this and override schema as needed.
"""
import json
from typing import Any, ClassVar, Iterator

from ..tabular import TabularAdapter


class JsonAdapter(TabularAdapter):
    """
    Native support for JSON/JSONL files that do not require special parsing.
    """
    accepted_formats: ClassVar[set[str]] = {"json", "ndjson", "jsonl"}
    accepted_types: ClassVar[set[str]] = {"*"}  # e.g. {"ncbi_genome_report", "protein_sequence"}
    view_mode: ClassVar[str] = "table"
    version: ClassVar[str] = "1.0.0"

    def parse(self, raw_data: Any, config: dict | None = None, **kwargs) -> list[dict]:
        """
        Intercepts monolithic data and converts it to a list of dicts
        """
        if not raw_data:
            return []
        kwconfig = self._prepare_config(config, **kwargs)

        # 1. Decode bytes if needed
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode("utf-8")

        # 2. Parse monolithic JSON strings
        if isinstance(raw_data, str):
            if kwconfig.get("ext") in ("jsonl", "ndjson"):
                raw_data = [json.loads(line) for line in raw_data.splitlines() if line.strip()]
            else:
                raw_data = json.loads(raw_data)

        # 3. Already a list of dicts, no special handling
        return super().adapt(raw_data, kwconfig)

    def parse_stream(
        self,
        raw_stream: Iterator[str],
        config: dict | None = None,
        **kwargs,
    ) -> Iterator[dict]:
        """
        Yields parsed JSON records from a stream. Works especially well for large
        JSONL/NDJSON files.
        """
        kwconfig = self._prepare_config(config, **kwargs)
        ext = kwconfig.get("ext", "json")

        if ext in ("jsonl", "ndjson"):
            for line in raw_stream:
                if line.strip():
                    yield json.loads(line)
        else:
            # For monolithic JSON, we need to buffer the entire stream
            # TODO: For true streaming, implement the ijson library
            buffer = "".join(raw_stream)
            yield from self.parse(buffer, kwconfig)

    # --- Output methods ---

    def serialize(self, records: list[dict], config: dict | None = None, **kwargs) -> str:
        """
        Use the built-in json library to serialize adapted records back into
        a string format.
        """
        if not records:
            return ""
        kwconfig = self._prepare_config(config, **kwargs)
        ext = kwconfig.get("ext", "json")

        if ext in ("jsonl", "ndjson"):
            return "\n".join(json.dumps(r) for r in records) + "\n"

        return json.dumps(records, indent=2)

    def serialize_stream(
        self,
        records_stream: Iterator[dict],
        config: dict | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Currently only serializes JSONL/NDJSON, as they can be streamed line by line.
        For monolithic JSON, the built-in JSON library does not support streaming.
        """
        kwconfig = self._prepare_config(config, **kwargs)
        ext = kwconfig.get("ext", "json")

        if ext in ("jsonl", "ndjson"):
            for record in records_stream:
                yield json.dumps(record) + "\n"
        else:
            # Stream monolithic JSON by faking array
            yield "[\n"
            first = True
            for record in records_stream:
                if not first:
                    yield ",\n"
                yield json.dumps(record, indent=2)
                first = False
            yield "\n]\n"
