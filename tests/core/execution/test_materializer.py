"""
Tests for materializing artifacts from the engine
"""
import json
from collections.abc import Iterator
import pytest
from pydantic import BaseModel

from lore.core.bindings import LiteralBinding
from lore.core.execution.materializer import _materialize_single_artifact, materialize_task_inputs
from lore.core.artifacts.artifact import ArtifactPathBundle
from lore.core.adapters.base import BaseAdapter
from lore.core.tasks import AdapterStrategy, ArtifactInput, Materialization, TaskDefinition


@pytest.fixture
def staged_artifact(temp_session, dummy_jsonl_file):
    """
    Registers the dummy JSONL file into the temporary session 
    and returns the Artifact record.
    """
    temp_session.register_artifact(
        dummy_jsonl_file,
        name="materialization_test",
        data_type="jsonl",
        metadata={"source": "pytest"},
    )
    # Fetch the newly created artifact record
    return temp_session.list_artifacts()[0]


def test_materialization_path(temp_session, staged_artifact):
    """Proves PATH materialization returns an ArtifactPathBundle."""
    result, io_meta = _materialize_single_artifact(
        session=temp_session,
        artifact=staged_artifact,
        materialization=Materialization.PATH,
        accepted_data=["*"],
    )

    assert isinstance(result, ArtifactPathBundle)
    assert "materialization_test.jsonl" in str(result.main)
    assert isinstance(io_meta, dict)


def test_materialization_raw(temp_session, staged_artifact):
    """Proves RAW materialization eagerly loads the full native Python objects."""
    result, io_meta = _materialize_single_artifact(
        session=temp_session,
        artifact=staged_artifact,
        materialization=Materialization.RAW,
        accepted_data=["*"],
    )

    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0]["id"] == 1
    assert isinstance(io_meta, dict)


def test_materialization_raw_stream(temp_session, staged_artifact):
    """Proves RAW_STREAM returns a lazy iterator, not a list"""
    result, io_meta = _materialize_single_artifact(
        session=temp_session,
        artifact=staged_artifact,
        materialization=Materialization.RAW_STREAM,
        accepted_data=["*"],
    )

    # Lazy iterator (generator / itertools.chain), not a materialized list
    assert isinstance(result, Iterator)
    assert not isinstance(result, list)

    # Exhaust the iterator to prove it holds the data
    streamed_data = list(result)
    assert len(streamed_data) == 3


def test_materialization_artifact_record(temp_session, staged_artifact):
    """Proves ARTIFACT materialization bypasses IO and returns the database record."""
    result, io_meta = _materialize_single_artifact(
        session=temp_session,
        artifact=staged_artifact,
        materialization=Materialization.ARTIFACT,
        accepted_data=["*"],
    )

    # We should get the actual Artifact object back
    assert result.id == staged_artifact.id
    assert result.name == "materialization_test"


# --- Shape consistency between a real run (AUTO) and a fast preview (PEEK) ---
# These pin the engine's core promise: strategy controls *how much* is read, never
# *what shape* the handler receives. A preview must be a faithful (possibly
# truncated) mirror of the run, otherwise tasks pass preview and fail on commit.


@pytest.fixture
def uid_artifact(temp_session, tmp_path):
    """A JSON array of records with a 'uid' column (mimics ESearch output)."""
    f = tmp_path / "esearch_out.json"
    f.write_text(json.dumps([
        {"uid": "111", "database": "nuccore"},
        {"uid": "222", "database": "nuccore"},
    ]))
    temp_session.register_artifact(
        f, name="esearch", data_type="uid", metadata={"columns": ["uid", "database"]}
    )
    return temp_session.list_artifacts()[0]


@pytest.fixture
def lines_artifact(temp_session, tmp_path):
    """A headerless single-column TSV (mimics extract_column output)."""
    f = tmp_path / "terms.tsv"
    f.write_text("alpha\nbeta\ngamma\n")
    temp_session.register_artifact(
        f, name="terms", data_type="text", metadata={"header": False}
    )
    return temp_session.list_artifacts()[0]


def _materialize(session, artifact, materialization, accepted_data, strategy):
    data, _ = _materialize_single_artifact(
        session=session, artifact=artifact,
        materialization=materialization, accepted_data=accepted_data, strategy=strategy,
    )
    return data


def test_adapted_series_same_shape_in_preview_and_run(temp_session, uid_artifact):
    """ADAPTED + a matching accepted_data slices the column to a flat series
    identically under AUTO (run) and PEEK (preview). This is the elink regression:
    preview used to return raw dicts while the run returned the string series."""
    run = _materialize(temp_session, uid_artifact, Materialization.ADAPTED, ["uid"], AdapterStrategy.AUTO)
    preview = _materialize(temp_session, uid_artifact, Materialization.ADAPTED, ["uid"], AdapterStrategy.PEEK)

    assert run == ["111", "222"]
    assert preview == ["111", "222"]  # not [{"uid": ...}, ...]


def test_adapted_records_same_shape_in_preview_and_run(temp_session, uid_artifact):
    """ADAPTED with no matching column yields adapted records, same shape in both."""
    run = _materialize(temp_session, uid_artifact, Materialization.ADAPTED, ["nope"], AdapterStrategy.AUTO)
    preview = _materialize(temp_session, uid_artifact, Materialization.ADAPTED, ["nope"], AdapterStrategy.PEEK)

    assert isinstance(run[0], dict) and run[0]["uid"] == "111"
    assert run == preview


def test_materialization_uses_selected_adapter(temp_session, uid_artifact, monkeypatch):
    class PrefixAdapter(BaseAdapter):
        accepted_formats = {"json"}

        def adapt_record(self, record, config=None, **kwargs):
            return {"uid": f"prefix-{record['uid']}"}

    class SuffixAdapter(BaseAdapter):
        accepted_formats = {"json"}

        def adapt_record(self, record, config=None, **kwargs):
            return {"uid": f"{record['uid']}-suffix"}

    adapters = [PrefixAdapter(), SuffixAdapter()]
    monkeypatch.setattr(type(uid_artifact), "get_adapters", lambda self: adapters)

    class Inputs(BaseModel):
        source: str = ArtifactInput(
            accepted_data="*",
            load_as=Materialization.ADAPTED,
        ).to_field_info()

    task_def = TaskDefinition(
        key="test.adapter_selection",
        handler=lambda ctx, **kwargs: None,
        input_model=Inputs,
        output_model=BaseModel,
    )
    result = materialize_task_inputs(
        s=temp_session,
        task_def=task_def,
        bindings={"source": [LiteralBinding(value=uid_artifact.id)]},
        input_adapters={"source": "SuffixAdapter"},
    )

    assert result.data["source"] == [{"uid": "111-suffix"}, {"uid": "222-suffix"}]


def test_adapted_stream_slices_series_like_adapted(temp_session, uid_artifact):
    """ADAPTED_STREAM applies the SAME column slicing as ADAPTED — a matching
    accepted_data column yields a series of strings, lazily. Only the container
    differs from ADAPTED (an iterator, not a list)."""
    run = _materialize(temp_session, uid_artifact, Materialization.ADAPTED_STREAM, ["uid"], AdapterStrategy.AUTO)
    assert isinstance(run, Iterator) and not isinstance(run, list)
    assert list(run) == ["111", "222"]

    preview = _materialize(temp_session, uid_artifact, Materialization.ADAPTED_STREAM, ["uid"], AdapterStrategy.PEEK)
    assert list(preview) == ["111", "222"]


def test_adapted_stream_streams_full_records_when_no_column_match(temp_session, uid_artifact):
    """ADAPTED_STREAM with no matching accepted_data column streams full records
    (the extract_column case: accepted_data is a type, not a column name)."""
    out = list(_materialize(temp_session, uid_artifact, Materialization.ADAPTED_STREAM, ["not_a_column"], AdapterStrategy.AUTO))
    assert out and isinstance(out[0], dict)
    assert out[0]["uid"] == "111"


def test_raw_same_shape_in_preview_and_run(temp_session, lines_artifact):
    """RAW hands back the reader's records (lines) the same way in preview and run."""
    run = _materialize(temp_session, lines_artifact, Materialization.RAW, ["*"], AdapterStrategy.AUTO)
    preview = _materialize(temp_session, lines_artifact, Materialization.RAW, ["*"], AdapterStrategy.PEEK)

    assert [l.strip() for l in run] == ["alpha", "beta", "gamma"]
    assert [l.strip() for l in preview] == ["alpha", "beta", "gamma"]
