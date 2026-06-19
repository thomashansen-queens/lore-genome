"""
Implementation of a Reader for tabular data formats.
"""
import json
from typing import Iterator
from .base import BaseReader


class TableReader(BaseReader):
    """
    Delegates the loading of tabular data
    (CSV, Parquet, JSONL, JSON array)
    """

    def get_metadata(self) -> dict:
        """
        Deep metadata. For tables, can't do a full row count (would require full
        file scan)
        """
        base_meta = self.get_base_metadata()
        ext = base_meta.get("extension")

        base_meta["can_stream"] = ext in {"csv", "tsv", "txt", "jsonl", "ndjson"}
        return base_meta

    def stream(self, config: dict | None = None, **kwargs) -> Iterator[str | dict]:
        """
        Memory-safe generator for large tabular files
        """
        io_config = {**(config or {}), **kwargs}
        ext = self.path.suffix.lower().lstrip(".")

        if ext in ("csv", "tsv", "txt"):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line

        elif ext in ("jsonl", "ndjson"):
            with open(self.path, "r", encoding="utf-8") as f:
                first_line = next((line for line in f if line.strip()), None)

            if not first_line:
                return

            try:
                parsed = json.loads(first_line)
                if not isinstance(parsed, dict):
                    raise ValueError("Valid JSON, but not a dictionary object")
            except (json.JSONDecodeError, ValueError):
                raise NotImplementedError(f"File '{self.path}' is not valid JSONL")

            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Corrupted JSONL data in '{self.path.name}': {str(e)}\nLine "
                                f"content: {line[:100]}..."
                            )

        else:
            raise NotImplementedError(f"Streaming not implemented for '{ext}'")

    def read_full(self, config: dict | None = None, **kwargs) -> list[str | dict]:
        """
        Loads the entire file into memory (or raises MemoryError if too big)
        """
        io_config = {**(config or {}), **kwargs}
        try:
            return list(self.stream(io_config))
        except NotImplementedError:
            pass  # not streamable, proceed to monolithic JSON

        text = self.path.read_text(encoding="utf-8").strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return [data]  # wrap single JSON object in a list
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]  # filter to dicts only
        except json.JSONDecodeError:
            pass  # not JSON, give up

        return []

    def preview(
        self,
        peek_limit: int = 100,
        config: dict | None = None,
        **kwargs,
    ) -> tuple[list[str | dict], dict]:
        """
        Smart preview with graceful monolithic fallback.
        Returns: tuple[preview_data, preview_metadata]
        """
        io_config = {**(config or {}), **kwargs}
        strategy = io_config.get("strategy", "peek")
        max_ram_rows = io_config.get("max_ram_rows", 10000)
        max_ram_bytes = io_config.get("max_ram_bytes", 50 * 1024 * 1024)  # 50 MB default

        data = []
        eof_hit = True
        rows_read = 0
        current_ram_bytes = 0
        ram_limit_hit = False

        try:
            for record in self.stream(io_config):
                rows_read += 1

                if strategy == "peek" and rows_read > peek_limit:
                    eof_hit = False
                    break

                if len(data) < max_ram_rows and not ram_limit_hit:
                    # Some efficiency could be gained here by sampling size, but probably not worth the milliseconds
                    record_size = len(str(record))
                    if current_ram_bytes + record_size > max_ram_bytes:
                        ram_limit_hit = True
                    else:
                        data.append(record)
                        current_ram_bytes += record_size

            io_strategy = f"Streamed ({strategy})"
            io_total_rows = rows_read if (strategy in ("full", "eager") or eof_hit) else None

        except NotImplementedError:
            # fallback to monolithic load
            all_records = self.read_full(io_config)
            io_total_rows = len(all_records)
            data = all_records[:max_ram_rows]
            eof_hit = io_total_rows <= max_ram_rows
            ram_limit_hit = not eof_hit
            io_strategy = f"Monolithic fallback ({strategy})"

        metadata = self.get_metadata()

        # CSV/TSV can be headerless, so only assume columns if we have a dict
        columns = []
        if data and isinstance(data[0], dict):
            columns = list(data[0].keys())

        metadata.update(
            {
                "io_strategy": io_strategy,
                "file_eof_hit": eof_hit,
                "preview_limit": peek_limit if strategy == "peek" else max_ram_rows,
                "total_rows": io_total_rows,  # Will be None if streamed, which is correct!
                "columns": columns,
                "ram_limit_hit": ram_limit_hit,
                "preview_ram_bytes": current_ram_bytes,
            }
        )

        return data, metadata
