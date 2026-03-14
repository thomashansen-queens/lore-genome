"""
Tests for ExecutionContext
"""
from types import GeneratorType
import pytest

from lore.core.execution.resolver import _materialize_single_artifact
from lore.core.tasks import Materialization


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
    """Proves PATH materialization returns a raw string."""
    result = _materialize_single_artifact(
        s=temp_session,
        artifact=staged_artifact,
        materialization=Materialization.PATH,
        accepted_data=["*"],
    )

    assert isinstance(result, str)
    assert "materialization_test.jsonl" in result


def test_materialization_raw(temp_session, staged_artifact):
    """Proves RAW materialization eagerly loads the full native Python objects."""
    result = _materialize_single_artifact(
        s=temp_session,
        artifact=staged_artifact,
        materialization=Materialization.RAW,
        accepted_data=["*"],
    )

    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0]["id"] == 1


def test_materialization_raw_stream(temp_session, staged_artifact):
    """Proves RAW_STREAM returns a lazy generator, not a list"""
    result = _materialize_single_artifact(
        s=temp_session,
        artifact=staged_artifact,
        materialization=Materialization.RAW_STREAM,
        accepted_data=["*"],
    )

    assert isinstance(result, GeneratorType)

    # Exhaust the generator to prove it holds the data
    streamed_data = list(result)
    assert len(streamed_data) == 3


def test_materialization_artifact_record(temp_session, staged_artifact):
    """Proves ARTIFACT materialization bypasses IO and returns the database record."""
    result = _materialize_single_artifact(
        s=temp_session,
        artifact=staged_artifact,
        materialization=Materialization.ARTIFACT,
        accepted_data=["*"],
    )

    # We should get the actual Artifact object back
    assert result.id == staged_artifact.id
    assert result.name == "materialization_test"
