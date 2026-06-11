"""
Tests for ExecutionContext
"""
import pytest

from lore.core.execution.context import ExecutionContext, PreviewContext
from lore.core.tasks import Task, TaskStatus


def test_context_temp_directory_lifecycle(
    temp_runtime,
    closed_session,
    dummy_task_plugin,
    isolated_task_registry,
):
    """
    Ensures ExecutionContext lazy-loads its temp directory and that it
    cleans up after itself
    """
    with closed_session as s:
        task = s.add_task(dummy_task_plugin, name="Test Task", inputs={})

    ctx = ExecutionContext(
        runtime=temp_runtime,
        session_id=closed_session.id,
        task=task,
        task_def=isolated_task_registry.get(dummy_task_plugin),
    )

    assert ctx._temp_dir is None

    temp_file = ctx.get_temp_path("my_output.csv")

    assert ctx._temp_dir is not None
    assert temp_file.parent.exists()

    ctx.cleanup()

    assert not temp_file.parent.exists()


def test_context_output_tracking(
    temp_runtime,
    closed_session,
    dummy_task_plugin,
    isolated_task_registry,
):
    """
    Ensures ExecutionContext correctly delegates materialize_file to its
    internal TaskResults object, maintaining the facade.
    """
    with closed_session as s:
        task = s.add_task(dummy_task_plugin, name="Test Task", inputs={})

    ctx = ExecutionContext(
        runtime=temp_runtime,
        session_id=closed_session.id,
        task=task,
        task_def=isolated_task_registry.get(dummy_task_plugin),
    )

    output_path = ctx.get_temp_path("test_output.txt")
    output_path.write_text("Hello, world!")

    ctx.materialize_file(
        source_path=output_path,
        output_key="greeting_file",
        name="test_greeting",
    )

    assert "greeting_file" in ctx.results
    output_artifact = ctx.results["greeting_file"][0]

    assert output_artifact.name == "test_greeting"
    assert output_artifact.data_type == "text"

    with temp_runtime.open_session(closed_session.id) as s:
        assert s.get_artifact_path(output_artifact.id) is not None
        assert s.get_artifact_path(output_artifact.id).exists()


def test_preview_context_interception(
    temp_runtime, closed_session, dummy_task_plugin, isolated_task_registry,
):
    """
    Ensures PreviewContext correctly intercepts and handles preview-specific logic.
    """
    preview_task = Task(
        id="preview_test_123",
        registry_key=dummy_task_plugin,
        status=TaskStatus.RUNNING,
        inputs={},
        exec_config={"adapter": {"strategy": "peek"}} 
    )

    ctx = PreviewContext(
        runtime=temp_runtime,
        session_id=closed_session.id,
        task=preview_task,  # carries exec_config with adapter.strategy = 'peek'
        task_def=isolated_task_registry.get(dummy_task_plugin),
    )

    dummy_file = ctx.get_temp_path("test_preview.txt")
    dummy_file.write_text("This is a preview.")

    ctx.materialize_file(
        source_path=dummy_file,
        output_key="greeting_file",
        name="preview_greeting",
    )

    assert len(ctx.results.primary_data) == 1
    assert ctx.results.primary_data[0].get("is_preview") is True
    assert "This is a preview." in ctx.results.primary_data[0].get("data", "")
