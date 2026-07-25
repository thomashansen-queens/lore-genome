"""
Tests for the CSV/TSV adapter, focused on header handling: historically a spot
where two competing strategies (slice vs. iterator skip) collided and silently
corrupted the default "has header, no explicit fieldnames" case.
"""
import pytest

from lore.core.adapters.parsers.csv import CsvAdapter


@pytest.fixture
def adapter() -> CsvAdapter:
    return CsvAdapter()


def test_parse_with_header_no_fieldnames(adapter):
    """Default case: row 0 is the header and becomes the dict keys."""
    rows = adapter.parse("name,score\nalice,10\nbob,20", config={"ext": "csv"})
    assert rows == [
        {"name": "alice", "score": "10"},
        {"name": "bob", "score": "20"},
    ]


def test_parse_with_header_and_explicit_fieldnames(adapter):
    """Explicit fieldnames override the file's header row, which is skipped."""
    rows = adapter.parse(
        "name,score\nalice,10\nbob,20",
        config={"ext": "csv", "fieldnames": ["n", "s"]},
    )
    assert rows == [
        {"n": "alice", "s": "10"},
        {"n": "bob", "s": "20"},
    ]


def test_parse_headerless_synthesizes_columns(adapter):
    """Headerless data with no fieldnames gets column_N names; no rows lost."""
    rows = adapter.parse("alice,10\nbob,20", config={"ext": "csv", "header": False})
    assert rows == [
        {"column_0": "alice", "column_1": "10"},
        {"column_0": "bob", "column_1": "20"},
    ]


def test_parse_headerless_with_fieldnames(adapter):
    """Headerless data with explicit fieldnames keeps every row."""
    rows = adapter.parse(
        "alice,10\nbob,20",
        config={"ext": "csv", "header": False, "fieldnames": ["n", "s"]},
    )
    assert rows == [
        {"n": "alice", "s": "10"},
        {"n": "bob", "s": "20"},
    ]


def test_parse_tsv_delimiter(adapter):
    """The .tsv extension drives a tab delimiter."""
    rows = adapter.parse("a\tb\n1\t2", config={"ext": "tsv"})
    assert rows == [{"a": "1", "b": "2"}]


def test_parse_already_parsed_passthrough(adapter):
    """A list of dicts is already parsed and returned untouched."""
    records = [{"name": "alice"}, {"name": "bob"}]
    assert adapter.parse(records) == records


def test_parse_empty_returns_empty(adapter):
    assert adapter.parse("") == []
    assert adapter.parse(None) == []


def test_parse_strips_utf8_bom(adapter):
    """A UTF-8 BOM from Excel on the first header cell must not pollute the column name."""
    rows = adapter.parse(b"\xef\xbb\xbfname,score\nalice,10", config={"ext": "csv"})
    assert rows == [{"name": "alice", "score": "10"}]


def test_serialize_round_trip(adapter):
    """parse(serialize(records)) recovers the original records (string values)."""
    records = [{"name": "alice", "score": "10"}, {"name": "bob", "score": "20"}]
    text = adapter.serialize(records, config={"ext": "csv"})
    assert adapter.parse(text, config={"ext": "csv"}) == records
