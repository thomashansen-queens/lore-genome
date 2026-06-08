"""
Tests for I/O operations.
"""
from typing import cast

from lore.core.io import TableReader, get_reader_for


def test_table_reader_json_array(dummy_json_file):
    """
    Tests that TableReader correctly yields individual dictionaries 
    from a physical JSON (JSON array) file.
    """
    reader = TableReader(dummy_json_file)

    streamed_data = cast(list[dict], reader.read_full())

    assert len(streamed_data) == 3
    assert streamed_data[0]["id"] == 1
    assert streamed_data[1]["genes"] == ["BRCA1", "TP53"]
    assert streamed_data[1]["nest_list"][0][0]["genus"] == "Canis"
    assert streamed_data[2]["nested"]["val"] == "C"


def test_table_reader_streams_jsonl(dummy_jsonl_file):
    """
    Tests that TableReader correctly yields individual dictionaries 
    from a physical JSONL (JSON lines) file.
    """
    reader = TableReader(dummy_jsonl_file)

    # Turn stream generator into a list to check its contents
    streamed_data = cast(list[dict], list(reader.stream()))

    assert len(streamed_data) == 3
    assert streamed_data[0]["id"] == 1
    assert streamed_data[1]["genes"] == ["BRCA1", "TP53"]
    assert streamed_data[1]["nest_list"][1][0]["genus"] == "Mus"
    assert streamed_data[2]["nested"]["val"] == "C"


def test_table_reader_preview_jsonl(dummy_jsonl_file):
    """
    Tests that TableReader correctly yields a preview (first N lines) from a 
    physical JSONL (JSON lines) file.
    """
    reader = TableReader(dummy_jsonl_file)

    preview, metadata = reader.preview(limit=2)
    preview = cast(list[dict], preview)

    assert len(preview) == 2
    assert metadata["file_eof_reached"] is False
    assert metadata["strategy_used"] == "Streamed preview"
    assert metadata["total_rows"] is None

    assert preview[0]["id"] == 1
    assert preview[1]["genes"] == ["BRCA1", "TP53"]
    assert preview[1]["nest_list"][1][0]["genus"] == "Mus"


def test_table_reader_metadata(dummy_jsonl_file):
    """
    Tests that the reader correctly identifies the file properties 
    without reading the whole thing.
    """
    reader = TableReader(dummy_jsonl_file)
    metadata = reader.get_metadata()

    assert metadata["exists"] is True
    assert metadata["extension"] == "jsonl"
    assert metadata["can_stream"] is True
    assert metadata["file_size_bytes"] > 0


def test_get_reader_factory_routing(dummy_jsonl_file):
    """Proves the factory correctly routes based on file extensions."""
    reader = get_reader_for(dummy_jsonl_file)
    assert isinstance(reader, TableReader)
