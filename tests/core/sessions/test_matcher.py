from unittest.mock import MagicMock

from lore.core.sessions.matcher import (
    find_artifact_candidates,
    find_artifacts_for_field,
    map_artifacts_to_task_inputs,
)
from lore.core.topology.traversal import find_valid_upstream_tasks

# --- Setup ---

def create_mock_artifact(id: str, types: set, data_type: str | None = None):
    """
    Make an Artifact with specified resolvable types to test TaskInput matching.

    `resolvable_types()` is the single source of truth for semantic-string
    matching: it already folds in the artifact's data_type, adapter-provided
    types, static schema keys, and dynamic column/key metadata (see
    Artifact.resolvable_types). The matcher only intersects against it, so tests
    inject the final resolved set directly rather than re-deriving it from columns.
    `data_type` is set separately because the trait path matches on it.
    """
    mock = MagicMock()
    mock.id = id
    mock.resolvable_types.return_value = types
    mock.data_type = data_type if data_type is not None else next(iter(types), "unknown")
    mock.get_adapters.return_value = []
    return mock


def create_mock_task_def_inputs(inputs_extra: dict[str, dict]) -> MagicMock:
    """Creates a mock TaskDefinition with specific *input* field metadata."""
    mock_def = MagicMock()
    mock_model = MagicMock()
    mock_model.model_fields = {k: MagicMock() for k in inputs_extra}
    mock_def.input_model = mock_model

    # Mirrors TaskDefinition.field_meta -> (FieldInfo, json_schema_extra)
    def mock_field_meta(key, is_output=False):
        return MagicMock(), inputs_extra[key]

    mock_def.field_meta.side_effect = mock_field_meta
    return mock_def


def create_mock_task_def(outputs_extra: dict[str, dict]) -> MagicMock:
    """Creates a mock TaskDefinition with specific output field metadata."""
    mock_def = MagicMock()
    mock_model = MagicMock()

    # Real mock fields that have json_schema_extra attached
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


# --- find_artifacts_for_field ---

def test_find_artifacts_for_field_semantic():
    """Standard semantic-type matching plus the wildcard shortcut."""
    mock_session = MagicMock()
    art_fasta = create_mock_artifact("1", {"fasta"})
    art_table = create_mock_artifact("2", {"table"})
    mock_session.list_artifacts.return_value = [art_fasta, art_table]

    # Exact match
    field_extra = {"is_artifact": True, "accepted_data": ["fasta"]}
    result = find_artifacts_for_field(mock_session, field_extra)
    assert [a.id for a in result] == ["1"]

    # Wildcard returns everything (and short-circuits list_artifacts)
    field_extra_wild = {"is_artifact": True, "accepted_data": ["*"]}
    assert len(find_artifacts_for_field(mock_session, field_extra_wild)) == 2


def test_find_artifacts_for_field_structural():
    """
    Duck-typing: a field requiring a column name (e.g. 'p_value') matches any
    artifact whose resolvable_types expose that column. The column -> resolvable
    type promotion happens inside Artifact.resolvable_types(); the matcher just
    intersects the requirement against that set.
    """
    mock_session = MagicMock()
    art_good = create_mock_artifact("1", {"table", "gene", "p_value"})
    art_bad = create_mock_artifact("2", {"table", "gene", "fold_change"})
    mock_session.list_artifacts.return_value = [art_good, art_bad]

    field_extra = {"is_artifact": True, "accepted_data": ["p_value"]}
    result = find_artifacts_for_field(mock_session, field_extra)
    assert [a.id for a in result] == ["1"]


def test_find_artifacts_for_field_trait():
    """A trait requirement (TABULAR) matches via DataTrait.is_satisfied_by, which
    reads the artifact's data_type rather than the semantic-string set."""
    from lore.core.topology.traits import TABULAR

    mock_session = MagicMock()
    art_csv = create_mock_artifact("1", {"csv"}, data_type="csv")
    art_fasta = create_mock_artifact("2", {"fasta"}, data_type="fasta")
    mock_session.list_artifacts.return_value = [art_csv, art_fasta]

    field_extra = {"is_artifact": True, "accepted_data": [TABULAR]}
    result = find_artifacts_for_field(mock_session, field_extra)
    assert [a.id for a in result] == ["1"]


def test_find_artifacts_for_field_non_artifact():
    """Primitive (non-artifact) input slots never match artifacts."""
    mock_session = MagicMock()

    assert find_artifacts_for_field(mock_session, {"is_artifact": False}) == []
    # is_artifact missing entirely is treated as falsy
    assert find_artifacts_for_field(mock_session, {"accepted_data": ["fasta"]}) == []
    mock_session.list_artifacts.assert_not_called()


# --- find_artifact_candidates ---

def test_find_artifact_candidates():
    """Builds a per-field candidate map, skipping primitive (non-artifact) fields."""
    mock_session = MagicMock()
    art_fasta = create_mock_artifact("1", {"fasta"})
    art_table = create_mock_artifact("2", {"table"})
    mock_session.list_artifacts.return_value = [art_fasta, art_table]

    task_def = create_mock_task_def_inputs({
        "seq": {"is_artifact": True, "accepted_data": ["fasta"]},
        "tbl": {"is_artifact": True, "accepted_data": ["table"]},
        "threshold": {"is_artifact": False, "accepted_data": ["*"]},
    })

    candidates = find_artifact_candidates(mock_session, task_def)

    assert set(candidates.keys()) == {"seq", "tbl"}  # primitive field excluded
    assert [a.id for a in candidates["seq"]] == ["1"]
    assert [a.id for a in candidates["tbl"]] == ["2"]


# --- map_artifacts_to_task_inputs ---

def test_map_artifacts_to_task_inputs():
    """Single fields take the first compatible artifact; multiple fields collect all."""
    art_fasta = create_mock_artifact("1", {"fasta"})
    art_table1 = create_mock_artifact("2", {"table"})
    art_table2 = create_mock_artifact("3", {"table"})
    lookup = {"1": art_fasta, "2": art_table1, "3": art_table2}

    mock_session = MagicMock()
    mock_session.get_artifact.side_effect = lambda aid: lookup.get(aid)

    task_def = create_mock_task_def_inputs({
        "seq": {"is_artifact": True, "accepted_data": ["fasta"], "select": "single"},
        "tables": {"is_artifact": True, "accepted_data": ["table"], "select": "multiple"},
    })

    mapping = map_artifacts_to_task_inputs(mock_session, task_def, ["1", "2", "3"])

    assert mapping["seq"] == "1"
    assert mapping["tables"] == ["2", "3"]


def test_map_artifacts_optional_multiple_collects_all():
    """`optional_multiple` is also treated as a list slot (regression: the old code
    only recognized the deprecated 'pair'/'two_or_more' values under the wrong key)."""
    art_table1 = create_mock_artifact("1", {"table"})
    art_table2 = create_mock_artifact("2", {"table"})
    lookup = {"1": art_table1, "2": art_table2}

    mock_session = MagicMock()
    mock_session.get_artifact.side_effect = lambda aid: lookup.get(aid)

    task_def = create_mock_task_def_inputs({
        "tables": {"is_artifact": True, "accepted_data": ["table"], "select": "optional_multiple"},
    })

    mapping = map_artifacts_to_task_inputs(mock_session, task_def, ["1", "2"])
    assert mapping["tables"] == ["1", "2"]


def test_map_artifacts_empty_source():
    """No source IDs yields an empty mapping without touching the session."""
    mock_session = MagicMock()
    task_def = create_mock_task_def_inputs({
        "seq": {"is_artifact": True, "accepted_data": ["fasta"], "select": "single"},
    })

    assert map_artifacts_to_task_inputs(mock_session, task_def, []) == {}
    mock_session.get_artifact.assert_not_called()


def test_map_artifacts_wildcard_field():
    """A wildcard field accepts the first artifact regardless of type."""
    art_fasta = create_mock_artifact("1", {"fasta"})
    mock_session = MagicMock()
    mock_session.get_artifact.side_effect = lambda aid: {"1": art_fasta}.get(aid)

    task_def = create_mock_task_def_inputs({
        "anything": {"is_artifact": True, "accepted_data": ["*"], "select": "single"},
    })

    mapping = map_artifacts_to_task_inputs(mock_session, task_def, ["1"])
    assert mapping["anything"] == "1"


# --- Topology traversal (cross-module: find_valid_upstream_tasks) ---
# NOTE: This exercises lore.core.topology.traversal, not sessions.matcher. It
# lives here because it is the only coverage for JIT upstream-output matching;
# consider relocating to a tests/core/topology/ module.

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
    assert result[0]["valid_outputs"] == ["sequence"]  # It correctly ignored "report"
