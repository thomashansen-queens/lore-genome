"""
Fixtures for tests.
"""
import json
from pathlib import Path
from unittest.mock import patch
from pydantic import BaseModel
import pytest

from lore.core.adapters import TableAdapter
from lore.core.runtime import build_runtime, Runtime
from lore.core.sessions import Session
from lore.core.tasks import task_registry

# --- Data fixtures ---

@pytest.fixture
def spaghetti_records() -> list[dict]:
    """The shared raw data for all TableReader tests."""
    return [
        {
            "id": 1,
            "nested": {"val": "A"},
            "genes": ["lacZ"],
            "nest_list": [{"genus": "Escherichia", "species": "coli"}],
        },
        {
            "id": 2,
            "nested": {"val": "B"},
            "genes": ["BRCA1", "TP53"],
            "nest_list": [[{"genus": "Canis", "species": "lupus"}], [{"genus": "Mus", "species": "musculus"}]],
        },
        {
            "id": 3,
            "nested": {"val": "C"},
            "genes": [],
            "nest_list": [[]],
        }
    ]


@pytest.fixture
def dummy_json_file(tmp_path: Path, spaghetti_records: list[dict]) -> Path:
    """Creates a temporary JSON array file."""
    json_path = tmp_path / "test_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(spaghetti_records, indent=2))
    return json_path


@pytest.fixture
def dummy_jsonl_file(tmp_path: Path, spaghetti_records: list[dict]) -> Path:
    """
    Creates a temporary JSONL file on the hard drive for IO testing.
    """
    jsonl_path = tmp_path / "test_data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in spaghetti_records:
            f.write(json.dumps(record) + "\n")
    return jsonl_path

# --- Runtime/Session fixtures ---

@pytest.fixture
def temp_runtime(tmp_path: Path) -> Runtime:
    """
    Bootstraps the master Runtime engine inside a temporary Pytest directory.
    """
    rt = build_runtime(data_root=tmp_path, verbose=False)
    return rt


@pytest.fixture
def temp_session(temp_runtime: Runtime):
    """
    Creates and yields an active Session tied to our temporary runtime.
    """
    session = temp_runtime.create_session(name="Test Session")

    # Enter the ContextManager for setup and teardown of the Session
    with session:
        yield session


@pytest.fixture
def closed_session(temp_runtime: Runtime) -> Session:
    """
    Creates and yields an active Session tied to our temporary runtime.
    """
    session = temp_runtime.create_session(name="Test Session")

    # Enter the ContextManager for setup and teardown of the Session
    with session:
        pass  # Immediately exit to close the session
    return session

# --- Task registry fixtures ---

@pytest.fixture
def isolated_task_registry(monkeypatch: pytest.MonkeyPatch):
    """
    Creates a sandboxed TaskRegistry.
    """
    # shallow copy of built-in tasks
    pristine_tasks = task_registry.all.copy()

    # force global registry to become this temporary dict for the duration of the test
    monkeypatch.setattr(task_registry, "_tasks", pristine_tasks)
    return task_registry

# --- Semantic Matching Adapter Fixtures ---

@pytest.fixture
def semantic_registry():
    """Provides a pristine AdapterRegistry for testing matching logic."""
    from lore.core.adapters import AdapterRegistry
    return AdapterRegistry()


@pytest.fixture
def populated_registry(semantic_registry):
    """A registry populated with highly specific dummy adapters to test resolution."""

    class MockNcbiAdapter(TableAdapter):
        # Semantic type *and* format-specific adapter
        accepted_types = {"ncbi_genome_reports"}
        accepted_formats = {"json"}

        @property
        def schema(self):
            return {"genome_accession": "accession"}

        def to_dataframe(self, raw_data, metadata, config):
            pass

    class GenericJsonAdapter(TableAdapter):
        # File format-specific adapter
        # accepted_types = {"*"}
        accepted_formats = {"json", "jsonl"}

    class ProteinFastaAdapter(TableAdapter):
        # Semantic type-specific adapter (any format e.g. fa, faa, fasta, txt)
        accepted_types = {"protein_fasta"}
        accepted_formats = {"*"}

    semantic_registry.register(MockNcbiAdapter())
    semantic_registry.register(GenericJsonAdapter())
    semantic_registry.register(ProteinFastaAdapter())

    return semantic_registry
