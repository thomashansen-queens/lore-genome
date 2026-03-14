"""
Tests for BaseExecutor and LocalSubprocessExecutor
"""
import subprocess
import sys

from lore.core.execution.executors import LocalSubprocessExecutor


class DummySleepExecutor(LocalSubprocessExecutor):
    """A test executor that sleeps for 10 seconds"""
    def submit(self, session_id: str, task_id: str) -> None:
        command = [sys.executable, "-c", "import time; time.sleep(10)"]
        proc = subprocess.Popen(command)
        self._active_processes[task_id] = proc


def test_executor_shutdown_kills_orphans():
    """Orphan-crushing machine"""
    executor = DummySleepExecutor()

    executor.submit(session_id="test_session", task_id="test_task")

    proc = executor._active_processes["test_task"]
    assert proc.poll() is None  # Process should be running

    executor.shutdown()  # Simulates CTRL+C or server shutdown
    proc.wait(timeout=1)  # Block until OS confirms termination

    assert proc.poll() is not None  # Process should be terminated
