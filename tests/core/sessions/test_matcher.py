import pytest
from unittest.mock import MagicMock

from lore.core.sessions.matcher import find_artifacts_for_field
from lore.core.topology.traversal import find_valid_upstream_tasks

# --- Setup ---

def create_mock_artifact(id: str, types: set, columns: list | None = None):
    """Make an Artifact with specified resolvable types to test TaskInput matching."""
    mock = MagicMock()
    mock.id = id
    mock.resolvable_types.return_value = types
    mock.metadata = {"columns": columns or []}
    return mock


def create_mock_task_def(outputs_extra: dict[str, dict]) -> MagicMock:
    """Creates a mock TaskDefinition with specific output field metadata."""
    mock_def = MagicMock()
    mock_model = MagicMock()

    # Real mock fields that have ,json_schema_extra attached
    real_fields_dict = {}
    for k, extra_dict in outputs_extra.items():
        mock_field = MagicMock()
        mock_field.json_schema_extra = extra_dict
        real_fields_dict[k] = mock_field

    mock_model.model_fields = real_fields_dict
    mock_def.output_model = mock_model

    # Mirrors TaskDefinition's field_meta method
    def mock_field_meta(key, is_output=False):
        if is_output and key in outputs_extra:
            return MagicMock(), outputs_extra[key]
        return MagicMock()

    mock_def.field_meta.side_effect = mock_field_meta

    return mock_def


# --- Tests ---

def test_find_artifacts_for_field_semantic():
    """Test standard semantic type matching."""
    mock_session = MagicMock()
    art_fasta = create_mock_artifact("1", {"fasta"})
    art_table = create_mock_artifact("2", {"table"})
    mock_session.list_artifacts.return_value = [art_fasta, art_table]

    # Test exact match
    field_extra = {"is_artifact": True, "accepted_data": ["fasta"]}
    result = find_artifacts_for_field(mock_session, field_extra)
    assert len(result) == 1
    assert result[0].id == "1"

    # Test wildcard
    field_extra_wild = {"is_artifact": True, "accepted_data": ["*"]}
    assert len(find_artifacts_for_field(mock_session, field_extra_wild)) == 2


def test_find_artifacts_for_field_structural():
    """Test duck-typing for required columns."""
    from lore.core.adapters import TableAdapter

    mock_session = MagicMock()
    art_table_good = create_mock_artifact("1", {"table"}, columns=["gene", "p_value"])
    art_table_bad = create_mock_artifact("2", {"table"}, columns=["gene", "fold_change"])
    
    art_table_good.get_adapters.return_value = [MagicMock(spec=TableAdapter)]
    art_table_bad.get_adapters.return_value = [MagicMock(spec=TableAdapter)]

    mock_session.list_artifacts.return_value = [art_table_good, art_table_bad]

    field_extra = {
        "is_artifact": True, 
        "accepted_data": ["p_value"],
    }

    result = find_artifacts_for_field(mock_session, field_extra)
    assert len(result) == 1
    assert result[0].id == "1"


def test_find_valid_upstream_outputs(monkeypatch):
    """Test JIT reference matching across the DAG."""
    mock_session = MagicMock()
    
    # Setup an upstream Task
    upstream_task = MagicMock()
    upstream_task.id = "task_A"
    upstream_task.registry_key = "tool.A"

    def mock_resolve(out_key, container):
        if out_key == "report":
            return {"data_type": "table"}
        elif out_key == "sequence":
            return {"data_type": "fasta"}
        return {"data_tye": "*"}

    upstream_task.resolve_output_type.side_effect = mock_resolve
    mock_session.list_tasks.return_value = [upstream_task]

    # Patch the global task_registry so the matcher can find the upstream definition
    mock_registry = MagicMock()
    mock_def = create_mock_task_def({
        "report": {"data_type": "table"},
        "sequence": {"data_type": "fasta"}
    })
    mock_registry.get_safe.return_value = mock_def
    monkeypatch.setattr("lore.core.tasks.registry.task_registry.get_safe", mock_registry.get_safe)

    # Current field needs a fasta
    field_extra = {"is_artifact": True, "accepted_data": ["fasta"]}
    
    # Run matcher
    result = find_valid_upstream_tasks(
        container=mock_session,
        current_task_id="task_B",
        tasks=[upstream_task],
        field_extra=field_extra,
    )

    # Assertions
    assert len(result) == 1
    assert result[0]["task"].id == "task_A"
    assert result[0]["valid_outputs"] == ["sequence"] # It correctly ignored "report"
