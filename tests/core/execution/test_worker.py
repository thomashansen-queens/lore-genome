"""
Tests for worker execution of Tasks, including materialization and error handling
"""
import json

import pytest

from lore.core.bindings import LiteralBinding
from lore.core.execution.preview import PreviewOutput
from lore.core.tasks import (
    ArtifactInput,
    Cardinality,
    Materialization,
    TaskStatus,
    ValueInput,
    TaskOutput,
)
from lore.core.execution import run_preview_worker, run_task_worker, ExecutionContext, PreviewPayload

# --- DUMMY TASK CONTRACTS ---

class DummyInputs:
    text_to_write = ValueInput(str, default="Default text")

class DummyOutputs:
    greeting_file = TaskOutput(data_type="text", label="some_output", is_primary=True)

# --- DUMMY TASK PLUGINS ---

@pytest.fixture
def failing_task_plugin(isolated_task_registry) -> str:
    """A task guaranteed to raise an exception for testing error handling"""
    @isolated_task_registry.register(
            
        key="test.failing_worker",
        inputs=DummyInputs,
        outputs=DummyOutputs,
        name="Failing Worker Task",
    )
    def handler(ctx: ExecutionContext, text_to_write: str):
        raise RuntimeError("Simulated catastrophic crash!")

    return "test.failing_worker"

# --- Tests ---

def test_run_task_worker_success(temp_runtime, closed_session, dummy_task_plugin):
    """Ensure run_task_worker correctly processes inputs, runs the handler, and saves outputs."""
    with closed_session as s:
        task = s.add_task(
            registry_key=dummy_task_plugin,
            name="Worker Test",
            inputs={"text_to_write": [LiteralBinding(value="Science works!")]},
        )
        task_id = task.id

    run_task_worker(temp_runtime, closed_session.id, task_id)

    with temp_runtime.open_session(closed_session.id, read_only=True) as s:
        finished_task = s.get_task(task_id)

        # 1. State machine assertions
        assert finished_task.status == TaskStatus.COMPLETED
        assert finished_task.error is None
        assert finished_task.started_at is not None
        assert finished_task.completed_at is not None

        # 2. Output schema assertions
        artifact_ids = finished_task.outputs.get("greeting_file")
        assert artifact_ids is not None
        assert len(artifact_ids) == 1

        # 3. Physical file assertions
        artifact = s.get_artifact(artifact_ids[0])
        assert artifact.name == "test_greeting"

        file_path = s.get_artifact_path(artifact.id)
        assert file_path.exists()
        assert file_path.read_text() == "Processed: Science works!"


def test_run_task_worker_graceful_failure(temp_runtime, closed_session, failing_task_plugin):
    """Ensure run_task_worker catches catastrophic handler errors and updates the DB."""
    with closed_session as s:
        task = s.add_task(
            registry_key=failing_task_plugin,
            name="Failing Test",
            inputs={"text_to_write": [LiteralBinding(value="Doesn't matter")]},
        )
        task_id = task.id

    run_task_worker(temp_runtime, closed_session.id, task_id)

    with temp_runtime.open_session(closed_session.id, read_only=True) as s:
        failed_task = s.get_task(task_id)

        assert failed_task.status == TaskStatus.FAILED
        assert failed_task.completed_at is not None
        assert "Simulated catastrophic crash!" in failed_task.error

# --- Preview tests ---

def test_run_preview_worker_isolation(temp_runtime, closed_session, dummy_task_plugin):
    """Ensure preview runs the handler and returns outputs without saving a Task."""
    with closed_session as s:
        initial_task_count = len(s.list_tasks())
        initial_artifact_count = len(s.list_artifacts())

    raw_inputs = {"text_to_write": [LiteralBinding(value="Previewing is fun!")]}

    preview_payload = run_preview_worker(
        rt=temp_runtime,
        session_id=closed_session.id,
        task_key=dummy_task_plugin,
        raw_inputs=raw_inputs,
    )

    assert isinstance(preview_payload, PreviewPayload)

    primary_data = preview_payload.output_previews.get("greeting_file", [])

    assert isinstance(primary_data, PreviewOutput)
    assert primary_data.data[0] == "Processed: Previewing is fun!"

    # The input was a literal (consumed in full) and the tiny text output was
    # read to EOF, so this preview represents the complete result.
    assert primary_data.result_complete is True
    assert primary_data.display_complete is True
    assert primary_data.truncation_reason is None

    with temp_runtime.open_session(closed_session.id, read_only=True) as s:
        assert len(s.list_tasks()) == initial_task_count
        assert len(s.list_artifacts()) == initial_artifact_count
        ghost_tasks = [t for t in s.list_tasks() if t.id.startswith("preview_")]
        assert len(ghost_tasks) == 0


def test_run_preview_worker_propagates_input_truncation(
    tmp_path, temp_runtime, closed_session, isolated_task_registry
):
    """
    When an input is peeked (more rows than the peek limit), the derived output
    must be flagged as a partial preview — even though the small output file is
    itself read to EOF. This proves the input IO is propagated to the output.
    """
    # 1. Register a tabular artifact with more rows than the peek limit
    peek_limit = temp_runtime.settings.preview_peek_limit
    records = [{"id": i, "val": f"row_{i}"} for i in range(peek_limit + 50)]
    src = tmp_path / "big.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    with temp_runtime.open_session(closed_session.id) as s:
        artifact_id = s.register_artifact(src, name="big_table", data_type="jsonl").id

    # 2. A task that echoes its (peeked) input straight back out as the output
    class EchoInputs:
        records = ArtifactInput(
            accepted_data=["*"],
            select=Cardinality.SINGLE,
            load_as=Materialization.RAW,
        )

    class EchoOutputs:
        table = TaskOutput(data_type="table", label="Echoed", is_primary=True)

    @isolated_task_registry.register(
        key="test.echo_table",
        inputs=EchoInputs,
        outputs=EchoOutputs,
        name="Echo Table",
        preview_mode="live",
    )
    def handler(ctx: ExecutionContext, records):
        payload = "\n".join(json.dumps(r) for r in records)
        ctx.materialize_content(payload, output_key="table", extension="jsonl")

    # 3. Preview in PEEK mode (what the workbench sends for a live preview)
    preview_payload = run_preview_worker(
        rt=temp_runtime,
        session_id=closed_session.id,
        task_key="test.echo_table",
        raw_inputs={"records": [LiteralBinding(value=artifact_id)]},
        exec_config={"adapter": {"strategy": "peek"}},
    )

    assert preview_payload.error is None
    out = preview_payload.output_previews["table"]

    # Handler only saw the peeked slice...
    assert len(out.data) == peek_limit
    # ...so the RESULT is a sample, even though the small output file was itself
    # read to EOF (the display is complete).
    assert out.result_complete is False
    assert out.display_complete is True
    assert out.truncation_reason == "sampled"
    # The output's own row count isn't the result's total, so total_rows is unknown
    assert out.io_metadata["total_rows"] is None
