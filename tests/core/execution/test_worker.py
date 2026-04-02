"""
Tests for worker execution of Tasks, including materialization and error handling
"""

import pytest

from lore.core.bindings import LiteralBinding
from lore.core.execution.context import ExecutionContext, PreviewContext
from lore.core.tasks import task_registry, TaskResults, ValueInput, TaskOutput, TaskStatus
from lore.core.execution.worker import run_preview_worker, run_task_worker

# --- Dummy task contracts ---

class DummyWorkerInputs:
    text_to_write = ValueInput(str, default="Default text")

class DummyWorkerOutputs:
    greeting_file = TaskOutput(data_type="text", label="some_output", is_primary=True)

# --- Dummy task handler ---

def dummy_success_handler(ctx: ExecutionContext, text_to_write: str):
    """A perfectly behaved handler."""
    out_path = ctx.get_temp_path("output.txt")
    out_path.write_text(f"Processed: {text_to_write}")

    ctx.materialize_file(
        source_path=out_path,
        output_key="greeting_file",
        name="test_greeting"
    )


def dummy_crashing_handler(ctx: ExecutionContext, text_to_write: str):
    """A handler that simulates a catastrophic failure (e.g. out of memory, API timeout)."""
    raise RuntimeError("Simulated catastrophic crash!")


def dummy_preview_handler(ctx: PreviewContext, text_to_write: str):
    """
    A minimal preview handler for testing. Delegates to PreviewContext.materialize_file,
    which intercepts the call and builds the preview payload rather than saving to disk.
    """
    out_path = ctx.get_temp_path("output.txt")
    out_path.write_text(f"Processed: {text_to_write}")

    ctx.materialize_file(
        source_path=out_path,
        output_key="greeting_file",
        name="test_greeting"
    )

# --- Register dummy tasks ---

@pytest.fixture
def dummy_task_plugin(isolated_task_registry):
    """Temporarily registers a successful task for testing."""
    isolated_task_registry.register(
        "test.dummy_worker",
        inputs=DummyWorkerInputs,
        outputs=DummyWorkerOutputs,
        name="Dummy Worker Task",
    )(dummy_success_handler)

    return "test.dummy_worker"


@pytest.fixture
def failing_task_plugin(isolated_task_registry):
    """Temporarily registers a failing task for testing."""
    isolated_task_registry.register(
        "test.failing_worker",
        inputs=DummyWorkerInputs,
        outputs=DummyWorkerOutputs,
        name="Failing Worker Task"
    )(dummy_crashing_handler)

    return "test.failing_worker"


@pytest.fixture
def dummy_preview_plugin(isolated_task_registry):
    """Temporarily registers a task specifically flagged for live_preview."""
    isolated_task_registry.register(
        "test.dummy_preview",
        inputs=DummyWorkerInputs,
        outputs=DummyWorkerOutputs,
        name="Dummy Preview Task",
        live_preview=True,
    )(dummy_preview_handler)

    return "test.dummy_preview"

# --- Tests ---

def test_run_task_worker_success(temp_runtime, closed_session, dummy_task_plugin):
    """Ensure run_task_worker correctly processes inputs, runs the handler, and saves outputs."""
    # ARRANGE
    rt = temp_runtime

    # Close the session block after creating the task 
    with closed_session as s:
        session_id = s.id
        task = s.add_task(
            registry_key=dummy_task_plugin,
            name="Worker Test",
            inputs={"text_to_write": [LiteralBinding(value="Science works!")]},
        )
        task_id = task.id

    # ACT
    run_task_worker(rt, session_id, task_id)

    # ASSERT
    with rt.open_session(session_id) as s:
        finished_task = s.get_task(task_id)

        # DEBUG:
        if finished_task.status == TaskStatus.FAILED:
            print(f"\nWORKER CRASH REASON: {finished_task.error}")

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
    # ARRANGE
    rt = temp_runtime
    with closed_session as s:
        session_id = s.id
        task = s.add_task(
            registry_key=failing_task_plugin,
            name="Failing Test",
            inputs={"text_to_write": [LiteralBinding(value="Doesn't matter")]},
        )
        task_id = task.id

    # ACT
    run_task_worker(rt, session_id, task_id)

    # ASSERT
    with rt.open_session(session_id) as s:
        failed_task = s.get_task(task_id)

        assert failed_task.status == TaskStatus.FAILED
        assert failed_task.completed_at is not None
        assert "Simulated catastrophic crash!" in failed_task.error

# --- Preview tests ---

def test_run_preview_worker_isolation(temp_runtime, closed_session, dummy_preview_plugin):
    """Ensure preview runs the handler and returns outputs without saving a Task."""
    # ARRANGE
    rt = temp_runtime

    with closed_session as s:
        session_id = s.id
        initial_task_count = len(s.list_tasks())
        initial_artifact_count = len(s.list_artifacts())

    raw_inputs = {"text_to_write": [LiteralBinding(value="Previewing is fun!")]}

    # ACT
    payload = run_preview_worker(
        rt=rt,
        session_id=session_id,
        task_key=dummy_preview_plugin,
        raw_inputs=raw_inputs,
    )

    # ASSERT
    assert isinstance(payload, TaskResults)
    assert "greeting_file" in payload
    assert payload["greeting_file"][0]["is_preview"] is True
    assert "Previewing is fun!" in payload["greeting_file"][0]["data"]

    with rt.open_session(session_id) as s:
        assert len(s.list_tasks()) == initial_task_count
        assert len(s.list_artifacts()) == initial_artifact_count
        ghost_tasks = [t for t in s.list_tasks() if t.id.startswith("preview_")]
        assert len(ghost_tasks) == 0
