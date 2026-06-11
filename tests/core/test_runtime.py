"""
Tests for Runtime module
"""
from pathlib import Path
import pytest
from unittest.mock import patch

from lore.core.runtime import build_runtime, Runtime
from lore.core.sessions import Session
from lore.core.tasks import Task, TaskStatus

# --- Runtime factory ---

def test_build_runtime_with_explicit_data_root(tmp_path: Path):
    """
    Tests specifying a data_root when building the Runtime
    """
    rt = build_runtime(data_root=tmp_path)

    assert rt.data_root == tmp_path.resolve()
    assert rt.sessions_dir == tmp_path.resolve() / "sessions"

    assert rt.sessions_dir.exists()
    assert rt.sessions_dir.is_dir()


def test_build_runtime_with_env_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Tests that the factory correctly falls back to the LORE_DATA_ROOT 
    environment variable if no explicit argument is given.
    """
    fake_env_path = tmp_path / "env_root"
    monkeypatch.setenv("LORE_DATA_ROOT", str(fake_env_path))

    rt = build_runtime()

    assert rt.data_root == fake_env_path.resolve()
    assert rt.sessions_dir.exists()
    assert rt.sessions_dir.is_dir()

# --- Session management ---

def test_session_management_lifecycle(temp_runtime: Runtime):
    """
    Tests the complete CRUD lifecycle of a Session orcehstrated by the Runtime.
    """
    session = temp_runtime.create_session(name="Lifecycle Test")
    session_id = session.id

    assert session.name == "Lifecycle Test"
    assert temp_runtime.find_session_dir(session_id) is not None

    summaries = temp_runtime.list_sessions()
    assert len(summaries) >= 1
    assert any(s.id == session_id for s in summaries)

    temp_runtime.rename_session(session_id, "Renamed Session")

    updated_summaries = temp_runtime.list_sessions()
    renamed_summary = next(s for s in updated_summaries if s.id == session_id)
    assert renamed_summary.name == "Renamed Session"

    cloned_session = temp_runtime.clone_session(session_id=session_id, new_name="Cloned Session")
    assert cloned_session.id != session_id
    assert cloned_session.name == "Cloned Session"
    assert temp_runtime.find_session_dir(cloned_session.id) is not None

    temp_runtime.delete_session(session_id)
    temp_runtime.delete_session(cloned_session.id)

    assert temp_runtime.find_session_dir(session_id) is None
    assert not any(s.id == session_id for s in temp_runtime.list_sessions())


def test_session_management_edge_cases(temp_runtime: Runtime):
    """
    Tests edge cases in session management, such as renaming to an existing name or deleting non-existent sessions.
    """
    session1 = temp_runtime.create_session(name="Session 1")
    session2 = temp_runtime.create_session(name="Session 2")

    temp_runtime.rename_session(session2.id, "Session 1")

    assert temp_runtime.find_session_dir(session2.id) is not None
    assert session2.id in str(temp_runtime.find_session_dir(session2.id))
    assert "session_1" in str(temp_runtime.find_session_dir(session2.id))

    temp_runtime.delete_session(session1.id)

    with pytest.raises(ValueError, match="doesn't exist"):
        temp_runtime.delete_session(session1.id)

# --- Session portability ---

def test_session_import_export(temp_runtime: Runtime, tmp_path: Path):
    """
    Tests packaging a session, deleting it, and restoring it from an archive.
    """
    session = temp_runtime.create_session(name="Exported Session")
    session_id = session.id
    archive_path = tmp_path / "exported_session.zip"

    temp_runtime.export_session(session_id, archive_path)
    assert archive_path.exists()
    assert archive_path.stat().st_size > 0

    temp_runtime.delete_session(session_id)
    assert temp_runtime.find_session_dir(session_id) is None

    imported_session = temp_runtime.import_session(archive_path)

    assert imported_session.id == session_id
    assert imported_session.name == "Exported Session"
    assert temp_runtime.find_session_dir(imported_session.id) is not None


# --- Runtime facade for execution ---

def test_execution_facade_validation_errors(
    temp_runtime: Runtime,
    closed_session: Session,
    ephemeral_task: Task,
):
    """
    The Runtime should block execution before starting any workers if the session
    and/or tasks are not in a valid state/not found.
    """
    # 1. Invalid Session
    with pytest.raises(ValueError, match="not found"):
        temp_runtime.execute_task(
            session_id="malformed_session_id",
            task_id="some.task",
        )
    with pytest.raises(ValueError, match="not found"):
        temp_runtime.preview_task(
            session_id="malformed_session_id",
            task_key="some.task",
            raw_inputs={},
        )

    # 2. Invalid Task
    with pytest.raises(ValueError, match="not found"):
        temp_runtime.execute_task(
            session_id=closed_session.id,
            task_id="malformed_task_id",
        )
    with pytest.raises(ValueError, match="not found"):
        temp_runtime.preview_task(
            session_id=closed_session.id,
            task_key="malformed_task_key",
            raw_inputs={},
        )

    # 3. Invalid Task state (i.e. not runnable) and 'force' flag
    ephemeral_task.status = TaskStatus.DRAFT
    with temp_runtime.open_session(closed_session.id) as s:
        s.manifest.tasks[ephemeral_task.id] = ephemeral_task
        s.manifest.save(closed_session.dir / "manifest.json")

    with pytest.raises(ValueError, match="not runnable"):
        temp_runtime.execute_task(
            session_id=closed_session.id,
            task_id=ephemeral_task.id,
        )

    # Don't actually spawn the subprocess in this test
    with patch("subprocess.Popen") as mock_popen:
        temp_runtime.execute_task(
            session_id=closed_session.id,
            task_id=ephemeral_task.id,
            force=True,
        )
        mock_popen.assert_called_once()

    with temp_runtime.open_session(closed_session.id, read_only=True) as s:
        updated_task = s.get_task(ephemeral_task.id)
        assert updated_task is not None
        assert updated_task.status == TaskStatus.READY
        assert updated_task.error is None
