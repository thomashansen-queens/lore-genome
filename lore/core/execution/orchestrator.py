"""
Orchestrator to manage Executors and coordinate Task execution in a Workflow.
"""

import logging

from lore.core.runtime import Runtime
from lore.core.execution.executors import LocalSubprocessExecutor
from lore.core.topology.traversal import _dfs_sort_dag
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
        return _dfs_sort_dag(dependency_map)

    def run_cascade(self, session_id: str) -> None:
        """
        Executes all Tasks in a Session sequentially.
        """
        logger.info("Orchestrator starting cascade execution for Session %s", session_id)
        try:
            # 1. Get the correct execution order of Tasks based on their dependencies
            sorted_task_ids = self._get_execution_order(session_id)
        except Exception as e:
            logger.error("Failed to determine execution order: %s", e)
            return

        for task_id in sorted_task_ids:
            task = None

            with self.rt.open_session(session_id, read_only=True) as s:
                # 2. Check state
                # TODO: Use a TaskSignature to check if Task needs to be re-run (idempotent)
                task = s.get_task(task_id)
                if not task:
                    logger.error("Task %s not found in Session %s", task_id, session_id)
                    break
                if task.status == TaskStatus.COMPLETED:
                    logger.info("Skipping Task %s (already COMPLETED)", task_id)
                    continue
                if task.status == TaskStatus.FAILED:
                    logger.error("Halting cascade: Upstream Task %s has FAILED", task_id)
                    break
                if task.status == TaskStatus.RUNNING:
                    logger.warning("Task %s is marked RUNNING (likely oprhaned). Re-submitting...", task_id)
                    pass

            # 3. Execution phase (short lock on Session to update Task status)
            with self.rt.open_session(session_id, read_only=False) as s:
                task = s.get_task(task_id)
                if task is None:
                    raise ValueError(f"Task {task_id} vanished from Session {session_id} during execution")

                if task.status == TaskStatus.QUEUED:
                    logger.debug("Resolving ReferenceBindings for Task: %s", task.id)
                    resolved_inputs = {}

                    # Dig through the Task's inputs
                    for key, bindings in task.inputs.items():
                        resolved_list = []
                        for b in bindings:
                            if b.type == "reference":
                                # 1. Find the upstream Task
                                upstream_task = s.get_task(b.source_id)
                                if not upstream_task or upstream_task.status != TaskStatus.COMPLETED:
                                    raise ValueError(f"Cannot resolve input reference for Task {task.id}: Upstream Task {b.source_id} is not COMPLETED")

                                # 2. Extract the generated Artifact IDs
                                artifact_ids = upstream_task.outputs.get(b.output_key, [])

                                # 3. Swap the ReferenceBinding for a LiteralBinding with the actual Artifact IDs
                                from lore.core.bindings import LiteralBinding
                                for aid in artifact_ids:
                                    resolved_list.append(LiteralBinding(value=aid))
                            else:
                                # Keep LiteralBindings and UserInputBindings intact
                                resolved_list.append(b)

                        resolved_inputs[key] = resolved_list

                    # 4. Ask the Task if its references have been resolved
                    task.update(inputs=resolved_inputs)
                    s.mark_dirty()

            if task.status != TaskStatus.READY:
                logger.error("Task %s failed to become READY after resolution. Status: %s. Error: %s",
                             task.id, task.status, task.error)
                break

            logger.info("Submitting Task %s (%s)", task.id, task.registry_key)
            self.executor.submit(session_id, task.id)

            # 5. Block and wait
            exit_code = self.executor.wait(task.id)

            # 6. Post-execution verification
            with self.rt.open_session(session_id, read_only=True) as s:
                updated_task = s.get_task(task_id)
                if not updated_task:
                    logger.error("Task %s vanished from Session %s after execution", task_id, session_id)
                    break

                if exit_code != 0 or updated_task.status == TaskStatus.FAILED:
                    logger.error("Cascade halted: Task %s failed with exit code %s", task_id, exit_code)
                    break

        logger.info("Cascade finished for Session %s", session_id)
        self.executor.shutdown()
