"""
Tests for worker execution of Tasks, including materialization and error handling
"""
import pytest

from lore.core.bindings import LiteralBinding
from lore.core.execution.context import ExecutionContext, PreviewContext
from lore.core.tasks import TaskResults, TaskStatus, ValueInput, TaskOutput
from lore.core.execution.worker import run_preview_worker, run_task_worker

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

    results = run_preview_worker(
        rt=temp_runtime,
        session_id=closed_session.id,
        task_key=dummy_task_plugin,
        raw_inputs=raw_inputs,
    )

    assert isinstance(results, TaskResults)

    primary_data = results.primary_data
    assert len(primary_data) > 0

    first_result = primary_data[0]
    assert first_result.get("is_preview") is True
    assert "Previewing is fun!" in str(first_result.get("data", ""))

    with temp_runtime.open_session(closed_session.id, read_only=True) as s:
        assert len(s.list_tasks()) == initial_task_count
        assert len(s.list_artifacts()) == initial_artifact_count
        ghost_tasks = [t for t in s.list_tasks() if t.id.startswith("preview_")]
        assert len(ghost_tasks) == 0
