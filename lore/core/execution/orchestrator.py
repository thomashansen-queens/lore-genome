"""
Orchestrator to manage Executors and coordinate Task execution in a Workflow.
"""

import logging
from pathlib import Path

from lore.core.runtime import Runtime
from lore.core.execution.executors import LocalSubprocessExecutor
from lore.core.topology.traversal import sort_dag_dfs, get_task_descendants
from lore.core.bindings import ReferenceBinding
from lore.core.tasks.models import TaskStatus


logger = logging.getLogger(__name__)


class SequentialOrchestrator:
    """
    Coordinates the sequential execution of Tasks in a Session.
    Runs in its own isolated OS process.
    """
    def __init__(self, rt: Runtime):
        self.rt = rt
        self.executor = LocalSubprocessExecutor()

    def _get_task_log_path(self, session_id: str, task_id: str) -> Path:
        """
        Helper to resolve the Session directory and construct a log file path.
        """
        session_dir = self.rt.find_session_dir(session_id)
        if not session_dir:
            raise FileNotFoundError(f"Session {session_id} vanished before task {task_id} could start.")

        log_dir = session_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{task_id}.log"
        return log_path

    def _get_execution_order(self, session_id: str) -> list[str]:
        """
        Builds a dependency map from the Session's Tasks and returns a topologically
        sorted list of Task IDs.
        """
        with self.rt.open_session(session_id, read_only=True) as s:
            tasks = s.list_tasks()

            dependency_map = {}
            for task in tasks:
                upstream_ids = set()

                for bindings in task.inputs.values():
                    for b in bindings:
                        if isinstance(b, ReferenceBinding):
                            upstream_ids.add(b.source_id)

                dependency_map[task.id] = list(upstream_ids)
        return sort_dag_dfs(dependency_map)

    def _propagate_completion(self, session_id: str, task_id: str) -> None:
        """
        Walks the DAG downstream from the source task. Recheks readiness of down-
        stream tasks. TODO: Also mark TaskIntegrity using signatures.
        """
        with self.rt.open_session(session_id, read_only=False) as s:
            # 1. Find immediate children of the completed Task
            child_ids = get_task_descendants(s.list_tasks(), task_id, generations=1)

            # 2. Apply ripple effect
            for d_id in child_ids:
                d_task = s.get_task(d_id)
                if not d_task:
                    logger.warning("Downstream Task %s vanished. This shouldn't happen.", d_id)
                    continue
                # Task will recheck its state
                d_task.update()

            s.mark_dirty()

    def run_single(self, session_id: str, task_id: str) -> None:
        """
        Executes a single Task in the Session.
        """
        logger.info("Orchestrator executing single Task %s in Session %s", task_id, session_id)

        # 1. Verify runnable state
        with self.rt.open_session(session_id, read_only=True) as s:
            task = s.get_task(task_id)
            if not task:
                logger.error("Task %s not found in Session %s", task_id, session_id)
                return
            if not task.status.is_runnable:
                logger.warning("Task %s cannot be run (Status: %s)", task.id, task.status)
                return

        # 2. Run the Task using the Executor
        log_path = self._get_task_log_path(session_id, task_id)
        self.executor.submit(session_id, task_id, log_path)
        exit_code = self.executor.wait(task_id)
        self.executor.shutdown()

        # 3. Ripple state outward
        if exit_code != 0:
            logger.error("Task %s failed with exit code %s", task_id, exit_code)
        else:
            logger.info("Task %s completed successfully", task_id)
            self._propagate_completion(session_id, task_id)

    def run_cascade(self, session_id: str) -> None:
        """
        Executes all Tasks in a Session sequentially.
        """
        logger.info("Orchestrator starting cascade execution for Session %s", session_id)

        # 1. Get the correct execution order of Tasks based on their dependencies
        try:
            sorted_task_ids = self._get_execution_order(session_id)
        except Exception as e:
            logger.error("Failed to determine execution order: %s", e)
            return

        # 2. Check state
        for task_id in sorted_task_ids:
            task = None

            with self.rt.open_session(session_id, read_only=False) as s:
                # TODO: Use a TaskSignature to check if Task needs to be re-run (idempotent)
                task = s.get_task(task_id)
                if not task:
                    logger.error("Task %s vanished from Session %s", task_id, session_id)
                    break

                # Task will determine its readiness
                task.update()
                s.mark_dirty()

                if task.status == TaskStatus.COMPLETED:
                    logger.info("Skipping Task %s (status: COMPLETED)", task_id)
                    continue
                elif task.status == TaskStatus.FAILED:
                    logger.error("Re-running Task %s (status: FAILED)", task_id)
                    pass
                elif task.status == TaskStatus.RUNNING:
                    logger.warning("Re-running Task %s (status: RUNNING, likely orphaned)", task_id)
                    pass

                if not task.status.is_runnable:
                    logger.error("Task %s is not runnable (status: %s)", task_id, task.status)
                    continue

            # 3. Execution phase (short lock on Session to update Task status)
            logger.info("Submitting Task %s (%s)", task.id, task.registry_key)
            log_path = self._get_task_log_path(session_id, task_id)
            self.executor.submit(session_id, task.id, log_path)
            exit_code = self.executor.wait(task.id)

            # 4. Post-execution verification
            if exit_code != 0:
                logger.error("Task %s failed with exit code %s", task_id, exit_code)
                break
            else:
                logger.info("Task %s completed successfully", task_id)
                self._propagate_completion(session_id, task_id)

        logger.info("Cascade finished for Session %s", session_id)
        self.executor.shutdown()
