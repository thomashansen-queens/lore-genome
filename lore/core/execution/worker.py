"""
Isolated execution environment for a Task.
This code runs as a separate process entirely decoupled from the main process.
"""

from datetime import datetime, timezone
import traceback
from typing import TYPE_CHECKING
from uuid import uuid4
import logging

from lore.core.settings import LOG_FORMAT
from lore.core.tasks import task_registry, TaskResults, Task
from lore.core.execution.context import ExecutionContext, PreviewContext
from lore.core.execution.resolver import resolve_task_inputs

if TYPE_CHECKING:
    from lore.core.runtime import Runtime


def _attach_task_logger(rt: "Runtime", session_id: str, task_id: str) -> logging.FileHandler:
    """
    Creates a unique log file for the task and attaches it to the root logger.
    Returns the logging handler.
    """
    session_dir = rt._find_session_dir(session_id)
    if not session_dir:
        raise FileNotFoundError(f"Session {session_id} vanished before task {task_id} could start.")

    log_dir = session_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task_id}.log"

    # 1. Create the FileHandler
    task_handler = logging.FileHandler(log_path, mode="a")
    
    # 2. Use the same formatting as the central logger
    formatter = logging.Formatter(LOG_FORMAT)
    task_handler.setFormatter(formatter)

    # 3. Attach it to the main logger
    rt.logger.addHandler(task_handler)

    return task_handler


def run_task_worker(rt: "Runtime", session_id: str, task_id: str) -> None:
    """
    Worker function that runs as a background process.
    Manages the Task state machine (RUNNING -> COMPLETED/FAILED) and file locks.
    """
    # 1. Top bread (fast initialization, short lock)
    try:
        with rt.open_session(session_id) as s:
            task = s.get_task(task_id)
            if not task:
                rt.logger.error("Task ID: '%s' not found in Session %s", task_id, session_id)
                return
            if task.status != "PENDING":
                rt.logger.warning("Task ID: '%s' is %s: skipping execution.)", task_id, task.status)
                return

            task_def = task_registry[task.registry_key]

            # Resolve Task inputs
            try:
                resolved_kwargs, input_artifacts = resolve_task_inputs(
                    s=s,
                    task_def=task_def,
                    raw_inputs=task.inputs,
                )
            except Exception as e:
                task.status = "FAILED"
                task.error = f"Input resolution failed: {str(e)}"
                rt.logger.error("Input resolution failed: %s", e)
                s.mark_dirty()
                return

            task.status = "RUNNING"
            task.started_at = datetime.now(tz=timezone.utc)
            s.mark_dirty()
            rt.logger.info("Task ID: '%s' is now RUNNING", task_id)

    except Exception as e:
        rt.logger.error("Worker failed to initialize session: %s", e)
        return

    # 2. The meat (slow execution, no lock)
    ctx = None
    task_handler = _attach_task_logger(rt, session_id, task_id)

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
                return
            if task.status != "RUNNING":
                rt.logger.warning("Task ID: '%s' is %s during completion phase: skipping finalization.)", 
                            task_id, task.status)
                return

            task.status = "COMPLETED"
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
                    return

                task.status = "FAILED"
                task.error = error_msg
                task.completed_at = datetime.now(tz=timezone.utc)
                s.mark_dirty()

        except Exception as inner_e:
            rt.logger.error("Failed to save FAILED status to manifest: %s", inner_e)

    finally:
        if ctx:
            ctx.cleanup()  # Always clean up, even on failure

        rt.logger.removeHandler(task_handler)
        task_handler.close()


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
    """
    rt.logger.info("Running preview for '%s' in Session ID: '%s'", task_key, session_id)

    task_def = task_registry.get(task_key)

    if not task_def:
        raise ValueError(f"Task key: '{task_key}' not found in Task Registry.")
    if not task_def.live_preview:
        rt.logger.info("Generating preview for '%s'", task_key)

    # 1. Create ephemeral Task
    ephemeral_task = Task(
        id=f"preview_{uuid4().hex[:8]}",
        registry_key=task_key,
        status="RUNNING",
    )
    ephemeral_task.inputs = ephemeral_task.validate_and_serialize(raw_inputs)
    ephemeral_task.exec_config = ephemeral_task.validate_config(exec_config or {})

    # 2. Resolve inputs (requires a short lock on the session)
    with rt.open_session(session_id, read_only=True) as s:
        resolved_inputs, input_artifacts = resolve_task_inputs(s, task_def, ephemeral_task.inputs)

    # 3. Execute handler
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
