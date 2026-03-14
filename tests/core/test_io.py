"""
Tests for I/O operations.
"""
from lore.core.io import TableReader

def test_table_reader_json_array(dummy_json_file):
    """
    Tests that TableReader correctly yields individual dictionaries 
    from a physical JSON (JSON array) file.
    """
    # 1. ARRANGE
    reader = TableReader(dummy_json_file)

    # 2. ACT
    streamed_data = list(reader.read_full())

    # 3. ASSERT
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
    # 1. ARRANGE
    reader = TableReader(dummy_jsonl_file)

    # 2. ACT
    # We call stream() and turn the generator into a list to check its contents
    streamed_data = list(reader.stream())

    # 3. ASSERT
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
    # 1. ARRANGE
    reader = TableReader(dummy_jsonl_file)

    # 2. ACT
    preview, metadata = reader.preview(limit=2)

    # 3. ASSERT
    assert len(preview) == 2
    assert metadata["is_truncated"] is True
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
