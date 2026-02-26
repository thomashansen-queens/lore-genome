"""
Scripts to test the core Manifest functionality.
"""
import pytest

from lore.core.runtime import Runtime

def test_full_workflow(test_runtime: Runtime):
    """
    Test the lifecycle of LoRe Genome:
    Session -> Tasks -> Artifacts -> Manifest
    """
    # Start Session
    with test_runtime.create_session() as ses:
        assert ses.dir.exists()

        # Define a Task
        task_name = "test_task"
        inputs = {"input_1": "abc123", "input_2": "def456"}

        # Register a Task
        t_record = ses.add_task(task_name=task_name, inputs=inputs)
        assert t_record.status == "PENDING"

        # Run a Task
        with ses.start_task(t_record.id) as t:
            assert t.task_name == task_name
            assert t.inputs == inputs

            # Simulate task work and create an output file
            output_file = ses.dir / "test_output.txt"
            output_file.write_text("Hello Pytest", encoding="utf-8")

            # Ingest file as an Artifact
            artifact = ses.artifacts.create(
                path=output_file,
                emitted_by_id=t.task_id,
                metadata={"description": "Test output file", "test": True}
            )

            # Verify Artifact properties
            assert artifact.emitted_by_id == t.task_id
            assert artifact.metadata["description"] == "Test output file"
            assert artifact.size_bytes == len("Hello Pytest")
            assert artifact.data_type == "txt"

            # Complete the Task
            t.complete(outputs={"result": artifact.id})

        # Verify Manifest updates (re-fetches Task from manifest to get final state)
        final_task = ses.manifest.get_task(t_record.id)
        assert final_task is not None
        assert final_task.status == "COMPLETED"
        assert final_task.outputs["result"] == artifact.id

        # Verify file on disk where it should be (in artifacts folder)
        absolute_artifact_path = (ses.manifest.path.parent / artifact.path).resolve()
        assert absolute_artifact_path.exists()
        assert absolute_artifact_path.read_text(encoding="utf-8") == "Hello Pytest"

def test_task_failure(test_runtime: Runtime):
    """
    Test handling exceptions inside a task context manager.
    Should mark test as FAILED in Manifest.
    """
    with test_runtime.create_session() as ses:
        # Register a task
        t_record = ses.add_task(task_name="failing_task", inputs={})

        # Run the task and force a crash
        with pytest.raises(ValueError, match="Intentional Crash"):
            with ses.start_task(t_record.id):
                raise ValueError("Intentional Crash")

        # Inspect the task status in the manifest
        failed_task = ses.manifest.get_task(t_record.id)
        assert failed_task is not None
        assert failed_task.status == "FAILED"
        assert failed_task.error is not None
        assert "Intentional Crash" in failed_task.error
