"""
Orchestration layer for a Task in a Workflow
Executors determine *how* and *where* a Task is run, but do not run the Task
themselves.
"""

from pathlib import Path
import subprocess
import sys
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger("lore.execution")


class BaseExecutor(ABC):
    """
    Defines the contract for all LoRē Task Executors.
    """
    @abstractmethod
    def submit(self, session_id: str, task_id: str) -> None:
        """Dispatch a Task for execution."""
        pass

    @abstractmethod
    def wait(self, task_id: str) -> int | None:
        """Blocks until the Task completes. Returns the exit code."""
        pass

    @abstractmethod
    def cancel(self, task_id: str) -> bool:
        """Attempt to cancel a running Task."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up any resources used by the Executor upon termination."""
        pass


class LocalSubprocessExecutor(BaseExecutor):
    """
    Executes tasks locally using isolated OS subprocesses via the CLI entrypoint.
    Tracks PIDs to ensure all processes are killed when the main server ("app") 
    shuts down. Simply put, if you click the X-button, all jobs are killed.
    """
    def __init__(self):
        # Maps task_id -> active subprocess.Popen object
        self._active_processes: dict[str, subprocess.Popen] = {}

    def submit(self, session_id: str, task_id: str, log_path: Path) -> None:
        # 1. Run command (sys.executable for consistent Python environment)
        command = [
            sys.executable, "-m", "lore",
            "_worker-run-task",
            "--session", session_id,
            "--task", task_id
        ]

        logger.info("Submitting Task %s to LocalSubprocessExecutor", task_id)

        # 2. Spawn isolated OS process
        with open(log_path, "a") as f:
            proc = subprocess.Popen(
                command,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        self._active_processes[task_id] = proc

    def wait(self, task_id: str) -> int | None:
        proc = self._active_processes.get(task_id)
        if not proc:
            return None

        logger.info("Waiting for Task %s (PID: %s) to complete", task_id, proc.pid)
        return_code = proc.wait()
        logger.info("Task %s (PID: %s) completed with exit code %s", task_id, proc.pid, return_code)

        # Clean up
        del self._active_processes[task_id]
        return return_code

    def cancel(self, task_id: str) -> bool:
        proc = self._active_processes.get(task_id)
        if proc and proc.poll() is None:  # poll() is None means it is still running
            logger.info("Terminating Task %s (PID: %s)", task_id, proc.pid)
            proc.terminate()
            return True
        return False

    def shutdown(self) -> None:
        """Slaughter all orphaned background jobs on exit."""
        count = 0
        for task_id, proc in list(self._active_processes.items()):
            if proc.poll() is None:
                proc.terminate()
                count += 1

        if count > 0:
            logger.info("Graceful shutdown: Cleaned up %d orphaned background processes.", count)
