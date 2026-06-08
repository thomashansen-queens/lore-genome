"""
Tests for BaseExecutor and LocalSubprocessExecutor
"""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch, MagicMock

from lore.core.execution.executors import LocalSubprocessExecutor


class DummySleepExecutor(LocalSubprocessExecutor):
    """A test executor that sleeps for 10 seconds"""
    def submit(self, session_id: str, task_id: str) -> None:
        command = [sys.executable, "-c", "import time; time.sleep(10)"]
        proc = subprocess.Popen(command)
        self._active_processes[task_id] = (proc, None)


def test_executor_shutdown_kills_orphans():
    """Orphan-crushing machine"""
    executor = DummySleepExecutor()

    executor.submit(session_id="test_session", task_id="test_task")

    proc, _ = executor._active_processes["test_task"]
    assert proc.poll() is None  # Process should be running

    executor.shutdown()  # Simulates CTRL+C or server shutdown
    proc.wait(timeout=1)  # Block until OS confirms termination

    assert proc.poll() is not None  # Process should be terminated


def test_executor_submit_tracks_processes(tmp_path: Path):
    """
    Ensure that submitted tasks are tracked in LocalSubprocessExecutor's
    active_processes dict.
    """
    executor = LocalSubprocessExecutor()
    dummy_log = tmp_path / "task.log"

    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        mock_popen.return_value = mock_proc

        executor.submit(session_id="sess_123", task_id="task_abc", log_path=dummy_log)

        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        command = args[0]

        assert "lore" in command
        assert "sess_123" in command
        assert "task_abc" in command

        assert "stdout" in kwargs
        assert hasattr(kwargs["stdout"], "write")  # An open file handle for logging

        assert "task_abc" in executor._active_processes
        proc, log_handle = executor._active_processes["task_abc"]
        assert proc.pid == 9999
        assert log_handle is not None
        assert not log_handle.closed

        executor.cancel("task_abc")
        assert log_handle.closed
