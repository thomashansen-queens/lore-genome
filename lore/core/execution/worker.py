"""
Isolated execution environment for a Task.
This code runs as a separate process entirely decoupled from the main process.
"""

from datetime import datetime, timezone
import traceback
from typing import TYPE_CHECKING
from uuid import uuid4
import logging
import sys

from lore.core.tasks import task_registry, TaskResults, Task, TaskStatus
from lore.core.execution.context import ExecutionContext, PreviewContext
from lore.core.execution.materializer import materialize_task_inputs

if TYPE_CHECKING:
    from lore.core.runtime import Runtime

logger = logging.getLogger(__name__)


def run_task_worker(rt: "Runtime", session_id: str, task_id: str) -> None:
    """
    Worker function that runs as a background process.
    Manages the Task state machine (RUNNING -> COMPLETED/FAILED) and file locks.
    On error, uses sys.exit(1) to signal failure to the Orchestrator.
    """
    # 1a. Top bread (fast initialization, short lock) - mutates Session manifest
    try:
        with rt.open_session(session_id, read_only=False) as s:
            task = s.get_task(task_id)
            if not task:
                rt.logger.error("Task ID: '%s' not found in Session %s", task_id, session_id)
                sys.exit(1)
            if not task.status.is_runnable:
                rt.logger.warning("Task ID: '%s' is %s: skipping execution.)", task_id, task.status)
                sys.exit(1)

            task.validate_and_serialize()  # sanity check
            task_def = task_registry[task.registry_key]
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(tz=timezone.utc)

            s.mark_dirty()
            rt.logger.info("Task ID: '%s' is now RUNNING", task_id)
    except Exception as e:
        rt.logger.error("Worker failed to initialize task: %s", e)
        sys.exit(1)

    # 1b. Top bread continued (read-only, separated so heavy i/o for 
    #     materialization doesn't hold lock)
    try:
        with rt.open_session(session_id, read_only=True) as s:
            # Resolve references to concrete values (e.g. Artifact IDs to Artifacts)
            resolved_kwargs, input_artifacts = materialize_task_inputs(
                s=s,
                task_def=task_def,
                bindings=task.inputs,
            )
    except Exception as e:
        logger.error("Worker failed during materialization: %s", e)
        # Short lock on session again just to log fail state
        with rt.open_session(session_id, read_only=False) as s:
            task = s.get_task(task_id)
            if not task:
                rt.logger.error(
                    "Task ID: '%s' vanished from Session %s during input materialization.",
                    task_id, session_id,
                )
                sys.exit(1)
            task.status = TaskStatus.FAILED
            task.error = f"Input resolution or materialization failed: {str(e)}"
            s.mark_dirty()
        sys.exit(1)

    # 2. The meat (slow execution, no lock)
    ctx = None
    # task_handler = _attach_task_logger(rt, session_id, task_id)

    try:
        ctx = ExecutionContext(
            runtime=rt,
            session_id=session_id,
            task=task,
            task_def=task_def,
            input_artifacts=input_artifacts,
        )
        rt.logger.info("Executing Task ID: '%s' with handler %s", task_id, task_def.handler.__name__)
        task_def.handler(ctx=ctx, **resolved_kwargs)

        # 3. Bottom bread (fast cleanup, short lock)
        with rt.open_session(session_id) as s:
            task = s.get_task(task_id)  # Re-fetch the Task to ensure latest state

            if task is None:
                rt.logger.error("Task ID: '%s' vanished during execution in Session %s", task_id, session_id)
                sys.exit(1)
            if task.status != TaskStatus.RUNNING:
                rt.logger.warning("Task ID: '%s' is %s during completion phase: skipping finalization.)", 
                            task_id, task.status)
                sys.exit(1)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(tz=timezone.utc)
            task.outputs = ctx.results.to_dict()
            s.mark_dirty()
            rt.logger.info("Task completed successfully.")

    except Exception as e:
        error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
        rt.logger.error("Task %s FAILED: %s", task_id, str(e))

        try:
            with rt.open_session(session_id) as s:
                task = s.get_task(task_id)
                if task is None:
                    rt.logger.error("Task ID: '%s' vanished during execution in Session %s", task_id, session_id)
                    sys.exit(1)

                task.status = TaskStatus.FAILED
                task.error = error_msg
                task.completed_at = datetime.now(tz=timezone.utc)
                s.mark_dirty()

        except Exception as inner_e:
            rt.logger.error("Failed to save FAILED status to manifest: %s", inner_e)

    finally:
        if ctx:
            ctx.cleanup()  # Always clean up, even on failure

        # rt.logger.removeHandler(task_handler)
        # task_handler.close()


def run_preview_worker(
    rt: "Runtime",
    session_id: str,
    task_key: str,
    raw_inputs: dict,
    exec_config: dict | None = None,
) -> TaskResults:
    """
    Execute a Task purely in memory. Is synchronous and meant for quick previews
    in the UI. Does not modify the Manifest or create Artifacts.
    Errors raise or return, rather than sys.exit.
    """
    rt.logger.info("Running preview for '%s' in Session ID: '%s'", task_key, session_id)

    task_def = task_registry.get(task_key)

    if not task_def:
        raise ValueError(f"Task key: '{task_key}' not found in Task Registry.")
    if not task_def.live_preview:
        rt.logger.info("Generating preview for '%s'", task_key)

    # 1. Auto-wrap primitives into LiteralBindings for ergonomics
    # binding_inputs = wrap_in_bindings(raw_inputs)

    # 2. Create ephemeral Task
    ephemeral_task = Task(
        id=f"preview_{uuid4().hex[:8]}",
        registry_key=task_key,
        status=TaskStatus.RUNNING,
        inputs=raw_inputs,
    )
    ephemeral_task.exec_config = ephemeral_task.validate_config(exec_config or {})

    try:
        ephemeral_task.validate_and_serialize()
    except Exception as e:
        raise ValueError(f"Input validation failed: {str(e)}") from e

    # 3. Resolve inputs
    with rt.open_session(session_id, read_only=True) as s:
        resolved_inputs, input_artifacts = materialize_task_inputs(
            s,
            task_def,
            ephemeral_task.inputs,
        )
    rt.logger.info("PREVIEW BINDINGS: %s", ephemeral_task.inputs)
    rt.logger.info("PREVIEW RESOLVED ARTIFACTS: %s", input_artifacts.keys())
    rt.logger.info("PREVIEW KWARGS: %s", resolved_inputs.keys())

    # 4. Execute handler
    ctx = None
    try:
        ctx = PreviewContext(
            runtime=rt,
            session_id=session_id,
            task=ephemeral_task,
            task_def=task_def,
            input_artifacts=input_artifacts,
        )
        task_def.handler(ctx, **resolved_inputs)

        return ctx.results

    except Exception as e:
        rt.logger.error("Preview failed for '%s': %s", task_key, str(e), exc_info=True)
        raise ValueError(f"Preview execution failed: {str(e)}") from e

    finally:
        if ctx:
            ctx.cleanup()
