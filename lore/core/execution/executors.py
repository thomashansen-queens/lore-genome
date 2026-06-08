"""
Orchestration layer for a Task in a Workflow
Executors determine *how* and *where* a Task is run, but do not run the Task
themselves.
Currently, only LocalSubprocessExecutor because ThreadPoolExecutor was never useful.
In the future, should add SlurmExecutor.
"""

from pathlib import Path
import subprocess
import sys
import logging
from abc import ABC, abstractmethod
from typing import IO, Any

logger = logging.getLogger("lore.execution")


class BaseExecutor(ABC):
    """
    Defines the contract for all LoRē Task Executors.
    """
    @abstractmethod
    def submit(self, session_id: str, task_id: str, log_path: Path | None = None) -> None:
        """Dispatch a Task for execution."""
        pass

    @abstractmethod
    def wait(self, task_id: str) -> int | None:
        """Blocks until the Task completes. Returns the exit code."""
        pass

    @abstractmethod
    def cancel(self, task_id: str) -> bool:
        """Attempt to cancel a running Task. Returns True if cancellation was successful."""
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
        # Maps task_id -> (active subprocess.Popen object, open file descriptor for logs)
        self._active_processes: dict[str, tuple[subprocess.Popen, IO[Any] | None]] = {}

    def submit(self, session_id: str, task_id: str, log_path: Path | None = None) -> None:
        # 1. Run command (sys.executable for consistent Python environment)
        command = [
            sys.executable, "-m", "lore",
            "_worker-run-task",
            "--session", session_id,
            "--task", task_id
        ]

        logger.info("Submitting Task %s to LocalSubprocessExecutor", task_id)

        # 2. Spawn isolated OS process
        if log_path:
            f = open(log_path, "a", encoding="utf-8")
            proc = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
        else:
            f = None
            proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Track PID and log file handle for cleanup on shutdown
        self._active_processes[task_id] = (proc, f)

    def wait(self, task_id: str) -> int | None:
        if task_id not in self._active_processes:
            return None

        proc, f = self._active_processes[task_id]
        logger.info("Waiting for Task %s (PID: %s) to complete", task_id, proc.pid)

        return_code = proc.wait()
        logger.info(
            "Task %s (PID: %s) completed with exit code %s", task_id, proc.pid, return_code,
        )

        # Clean up
        if f:
            f.close()
        del self._active_processes[task_id]

        return return_code

    def cancel(self, task_id: str) -> bool:
        if task_id not in self._active_processes:
            return False

        proc, f = self._active_processes[task_id]
        if proc.poll() is None:  # poll() is None means it is still running
            logger.info("Terminating Task %s (PID: %s)", task_id, proc.pid)
            proc.terminate()
            if f:
                f.close()
            del self._active_processes[task_id]
            return True

        return False

    def shutdown(self) -> None:
        """Slaughter all orphaned background jobs and close file handles on exit."""
        count = 0
        for task_id, (proc, f) in list(self._active_processes.items()):
            if proc.poll() is None:
                proc.terminate()
                count += 1
            if f:
                f.close()

        self._active_processes.clear()

        if count > 0:
            logger.info("Graceful shutdown: Cleaned up %d orphaned background processes.", count)
