"""
Tests for the Reader layer (lore.core.readers).
"""
from typing import cast

from lore.core.readers import ImageReader, TableReader, TextReader, get_reader_for


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

    preview, metadata = reader.preview(peek_limit=2)
    preview = cast(list[dict], preview)

    assert len(preview) == 2
    assert metadata["file_eof_hit"] is False
    assert metadata["io_strategy"] == "Streamed (peek)"
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


def test_get_reader_factory_routing(dummy_jsonl_file, tmp_path):
    """Proves the factory correctly routes based on file extension."""
    assert isinstance(get_reader_for(dummy_jsonl_file), TableReader)
    assert isinstance(get_reader_for(tmp_path / "seq.fasta"), TextReader)
    assert isinstance(get_reader_for(tmp_path / "plot.svg"), ImageReader)
    assert isinstance(get_reader_for(tmp_path / "plot.png"), ImageReader)


def test_get_reader_factory_unsupported():
    """An unregistered extension raises rather than returning None."""
    import pytest
    with pytest.raises(ValueError):
        get_reader_for("mystery.zzz")


def test_image_reader_svg_returns_text(tmp_path):
    """SVG is vector markup, so it reads back as a string and is a complete read."""
    svg = tmp_path / "icon.svg"
    svg.write_text("<svg viewBox='0 0 10 10'></svg>", encoding="utf-8")

    data, metadata = ImageReader(svg).preview()

    assert isinstance(data, str)
    assert data.startswith("<svg")
    assert metadata["is_vector"] is True
    assert metadata["file_eof_hit"] is True  # whole file loaded


def test_image_reader_raster_returns_bytes(tmp_path):
    """Raster formats (e.g. PNG) read back as bytes."""
    png = tmp_path / "pixel.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n fake bytes")

    data, metadata = ImageReader(png).preview()

    assert isinstance(data, bytes)
    assert metadata["is_vector"] is False
