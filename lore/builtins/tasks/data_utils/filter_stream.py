"""
Task for filtering a list of records based on a pandas query string (streaming).

Interprets queries in a three-tiered approach:
1. If regex mode is enabled, treat the query as a regular expression pattern
2. Otherwise, attempt to apply the query as a pandas query string
3. If pandas cannot parse the query, treat it as a simple case-insensitive sub-
string search across all columns
"""
import logging
import re
from typing import Iterator
import pandas as pd

import lore


class QueryInputs:
    """Input model for the filter by query task."""
    source = lore.ArtifactInput(
        label="Artifact(s) to filter",
        accepted_data="tabular",  # LoRe 'trait' system will interpret this
        select="multiple",
        load_as="raw_stream",
    )
    regex = lore.ValueInput(
        bool,
        label="Regex mode",
        description="Treat the query string as a raw Regular Expression across all columns (bypasses Pandas query).",
        default=False,
    )
    query_string = lore.ValueInput(
        str | None,
        label="Query string",
        description="Pandas query string."
            "Copy and paste for NCBI Strict PostHoc Filter: ➜ "
            "~(genome_accession.str.startswith('GCA_', na=False) & paired_accession.str.startswith('GCF_', na=False)) "
            "and assembly_level in ['Complete Genome', 'Chromosome'] "
            "and genome_notes.isnull() "
            "and best_ani_match.str.contains('YOUR ORGANISM HERE', na=False)",
        default=None,
        examples=["assembly_level == 'Complete Genome' and year > 2020"],
    )


class QueryOutputs:
    """Output model for the filter by query task."""
    filtered_data = lore.TaskOutput(
        data_type=lore.Passthrough("source"),
        label="Filtered Data",
        description="A new Artifact containing only the records that match the query.",
        is_primary=True,
    )


def _make_query_pattern(query_string: str) -> str:
    """
    Convert a simple query string into a case-insensitive substring search pattern.
    TODO: This currently makes some assumptions about user intent. I could/should
    document this better and/or use the logger to explain assumptions
    """
    cleaned = query_string.strip().replace('"', '').replace("'", "")

    # Turn list into | separated for regex search
    if "," in cleaned:
        parts = [re.escape(p.strip()) for p in cleaned.split(",") if p.strip()]
        return "|".join(parts)
    return re.escape(cleaned)


def _process_and_serialize_chunk(
    parsed_chunk: list[dict],
    adapter: lore.TabularAdapter,
    query_string: str,
    regex: bool,
    ext: str,
    header_written: bool,
    logger: logging.Logger,
) -> str:
    """
    Helper to use Pandas to efficiently filter a chunk of adapted records, then 
    serialize the surviving records back in their original, unadapted format.
    """
    adapted_records = [adapter.adapt_record(r) for r in parsed_chunk]
    df = pd.DataFrame(adapted_records)

    # 1. Default behaviour for empty data or query string
    if df.empty:
        raise ValueError(
            "The adapted DataFrame is empty. Check the input data and adapter schema."
        )

    if not query_string or not str(query_string).strip():
        return adapter.serialize(
            parsed_chunk,
            extension=ext,
            header=not header_written,
        )

    # 2. Explicit Regex mode (skip escaping, skip pandas query)
    if regex:
        try:
            mask = df.astype(str).apply(
                lambda c: c.str.contains(query_string, regex=True)
            ).any(axis=1)
            surviving_indices = df[mask].index.tolist()
        except re.error as regex_err:
            raise ValueError(
                f"Invalid regular expression: {query_string}"
            ) from regex_err

    else:
        # 3. Pandas Query string
        try:
            surviving_indices = df.query(query_string).index.tolist()
        except Exception as query_err:
            if any(op in query_string for op in ["==", "!=", ">", "<", "&", "|", "~"]):
                raise ValueError(
                    f"Invalid Pandas Query Syntax: {query_err}\n"
                    f"Query: {query_string}"
                ) from query_err
            else:
                logger.info(
                    "Invalid pandas query (%s). Falling back to substring search for: %s", 
                    query_err, query_string
                )

            # 4. Fallback to case-insensitive substring search
            try:
                search_pattern = _make_query_pattern(query_string)
                mask = df.astype(str).apply(
                    lambda col: col.str.contains(search_pattern, case=False, na=False, regex=True)
                ).any(axis=1)
                surviving_indices = df[mask].index.tolist()
            except Exception as e:
                raise ValueError(f"Invalid query string '{query_string}': {e}") from e

    surviving_records = [parsed_chunk[i] for i in surviving_indices]

    return adapter.serialize(
        surviving_records,
        extension=ext,
        header=not header_written,
    )


def _filter_stream(
    raw_stream: Iterator[list],
    adapter: lore.TabularAdapter,
    query_string: str,
    regex: bool,
    ext: str,
    logger: logging.Logger,
    chunk_size: int = 10000,
) -> Iterator[str]:
    """
    Filter a streamed input by processing it in chunks. Rather than filtering
    record-at-a-time, make a DataFrame for each chunk, then apply the filter to
    take advantage of vectorized operations.
    """
    header_written = False
    current_chunk = []

    # 1. Iterator through the stream to build chunks
    for parsed_record in adapter.parse_stream(raw_stream, extension=ext):
        current_chunk.append(parsed_record)

        # 2. When the chunk is full, process it
        if len(current_chunk) >= chunk_size:
            yield _process_and_serialize_chunk(
                current_chunk, adapter, query_string, regex, ext, header_written, logger,
            )
            header_written = True  # Only write header for the first chunk
            # then reset the chunk
            current_chunk = []

    # 3. Clean up the final, partially-filled chunk
    if current_chunk:
        yield _process_and_serialize_chunk(
            current_chunk, adapter, query_string, regex, ext, header_written, logger,
        )


@lore.task(
    "filter.query_streamed",
    inputs=QueryInputs,
    outputs=QueryOutputs,
    name="Filter by query (streamed)",
    category="Data processing",
    icon="🔍︎",
    preview_mode="dry_run",
)
def filter_query_handler(
    ctx: lore.ExecutionContext,
    raw_stream: Iterator[list],
    regex: bool = False,
    query_string: str | None = None,
):
    """
    Non-desctructively filter a list of records using a pandas query string on
    the adapted DataFrame. This is a streaming version of the task for very
    large files and/or low-memory systems.
    """
    # 1. Get adapter and input artifact metadata
    adapter = ctx.get_input_adapter("source")
    if adapter is None:
        raise ValueError("No adapter found for the input Artifact(s).")
    if not isinstance(adapter, lore.TabularAdapter):
        raise ValueError(
            f"The adapter for the input Artifact(s) must be a TabularAdapter, "
            f"but got {type(adapter)}."
        )

    source_artifacts = ctx.input_artifacts.get("source", [])
    inherited_type = source_artifacts[0].data_type if source_artifacts else "unknown"
    ext = source_artifacts[0].extension if source_artifacts else "json"

    # 2. Create the chunked generator
    query_string = query_string or ""
    filtered_stream = _filter_stream(raw_stream, adapter, query_string, regex, ext, ctx.logger)

    ctx.materialize_stream(
        stream=filtered_stream,
        output_key="filtered_data",
        extension=ext,
    )
