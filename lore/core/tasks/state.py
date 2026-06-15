"""
Task state and integrity.
"""
from enum import StrEnum


class TaskStatus(StrEnum):
    """Execution state of the Task in the engine."""
    DRAFT = "draft"  # Missing config or fails validation
    READY = "ready"  # Validated and ready for the user to click 'Run'
    QUEUED = "queued"  # Waiting for engine resources OR upstream FutureArtifacts
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Success!
    FAILED = "failed"  # Errored out (check task.error)
    CANCELLED = "cancelled"  # Stopped by user
    UNKNOWN = "unknown"  # Fallback state
    TEMPLATE = "template"  # Workflow-only status

    @property
    def is_active(self) -> bool:
        """Currently doing something or about to."""
        return self in (TaskStatus.READY, TaskStatus.QUEUED, TaskStatus.RUNNING)

    @property
    def is_terminal(self) -> bool:
        """Will not change state unless the user does something."""
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def is_runnable(self) -> bool:
        """User can try to run this task."""
        return self in (
            TaskStatus.READY,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )


class TaskIntegrity(StrEnum):
    """
    Data continuity state of the Task within the DAG.
    Degraded is when output files are missing/changed, stale is when upstream inputs are modified.
    """
    INTACT = "intact"
    DEGRADED = "degraded"
    STALE = "stale"
    PENDING = "pending"
    UNKNOWN = "unknown"