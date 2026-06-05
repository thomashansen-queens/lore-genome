"""
Global fixtures for LoRē unit tests.
"""
import json
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel
import pytest

from lore.core.adapters import BaseAdapter
from lore.core.runtime import build_runtime, Runtime
from lore.core.sessions import Session
from lore.core.tasks import task_registry, Task, TaskDefinition, TaskStatus
from lore.core.execution.context import PreviewContext

# --- DATA FACTORIES & WORKSPACE FIXTURES ---

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
    json_path.write_text(json.dumps(spaghetti_records), encoding="utf-8")
    return json_path


@pytest.fixture
def dummy_jsonl_file(tmp_path: Path, spaghetti_records: list[dict]) -> Path:
    """
    Creates a temporary JSONL file on the hard drive for IO testing.
    """
    jsonl_path = tmp_path / "test_data.jsonl"
    lines = [json.dumps(record) for record in spaghetti_records]
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path

# --- ENGINE RUNTIME & RENVIRONMENT FIXTURES ---

@pytest.fixture(scope="session")
def temp_runtime(tmp_path_factory: pytest.TempPathFactory) -> Runtime:
    """
    Bootstraps the master Runtime engine inside a temporary Pytest directory.
    Scoped to "session" so all tests share the same Runtime instance, but each
    test gets a fresh Session via the temp_session fixture.
    """
    session_data_root = tmp_path_factory.mktemp("lore_master_runtime")

    rt = build_runtime(data_root=session_data_root, verbose=False)
    return rt


@pytest.fixture
def temp_session(temp_runtime: Runtime):
    """
    Creates and yields an active, open Session context manager tied to
    our temporary runtime.
    """
    session = temp_runtime.create_session(name="Test Active Session")

    # Enter the ContextManager for setup and teardown of the Session
    with session:
        yield session


@pytest.fixture
def closed_session(temp_runtime: Runtime) -> Session:
    """
    Creates and yields a closed Session tied to our temporary runtime.
    """
    session = temp_runtime.create_session(name="Test Closed Session")

    # Enter the ContextManager for setup and teardown of the Session
    with session:
        pass  # Immediately exit to close the session
    return session

# --- ADAPTER/SEMANTIC MATCHING FIXTURES ---

@pytest.fixture
def semantic_registry():
    """A clean, test-specific AdapterRegistry for testing matching logic."""
    from lore.core.adapters import AdapterRegistry
    return AdapterRegistry()


@pytest.fixture
def populated_adapter_registry(semantic_registry):
    """A registry populated with highly specific dummy adapters to test resolution."""
    class MockNcbiAdapter(BaseAdapter):
        accepted_types = {"ncbi_genome_reports"}
        accepted_formats = {"json"}
        view_mode = "table"

    class GenericJsonAdapter(BaseAdapter):
        accepted_types = {"*"}
        accepted_formats = {"json", "jsonl"}
        view_mode = "table"

    class ProteinFastaAdapter(BaseAdapter):
        accepted_types = {"protein_fasta"}
        accepted_formats = {"*"}
        view_mode = "table"

    semantic_registry.register(MockNcbiAdapter())
    semantic_registry.register(GenericJsonAdapter())
    semantic_registry.register(ProteinFastaAdapter())

    return semantic_registry

# --- LIVE EXECUTION FIXTURES ---

@pytest.fixture
def ephemeral_task() -> Task:
    """An initialized, valid Ephemeral Task model for Context testing."""
    return Task(
        id=f"test_preview_{uuid4().hex[:8]}",
        registry_key="test.diagnostic_task",
        status=TaskStatus.RUNNING,
        inputs={},
        exec_config={"adapter": {"strategy": "peek", "view_state": {}}},
    )


@pytest.fixture
def real_preview_context(temp_runtime, closed_session, ephemeral_task) -> PreviewContext:
    """
    A fully integrated PreviewContext with a live Task and Session.
    """
    class EmptySchema(BaseModel):
        pass

    diagnostic_def = TaskDefinition(
        key="test.diagnostic_task",
        name="Diagnostic Task",
        handler=lambda ctx, **kwargs: None,  # No-op handler for testing
        input_model=EmptySchema,
        output_model=EmptySchema,
    )

    return PreviewContext(
        runtime=temp_runtime,
        session_id=closed_session.id,
        task=ephemeral_task,
        task_def=diagnostic_def,
        input_artifacts={},
    )
