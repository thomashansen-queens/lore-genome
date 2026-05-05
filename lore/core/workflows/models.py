"""
Defines the component parts of a Workflow.
A Workflow is a DAG (directed acyclic graph) of Tasks.
"""

from datetime import datetime, timezone
from pydantic import BaseModel, Field, model_validator

from lore.core.tasks.models import Task


class Workflow(BaseModel):
    """
    A complete, shareable template for a pipeline of Tasks.
    """

    id: str = Field(..., description="Unique ID for this workflow.")
    name: str = Field(..., description="Human-readable name for this workflow instance.")
    description: str = Field(
        default="",
        description="Optional longer description of what this workflow does.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when this workflow was created.",
    )

    tasks: list[Task] = Field(
        default_factory=list,
        description="The individual Tasks that make up this workflow.",
    )

    @model_validator(mode="after")
    def check_unique_task_ids(self) -> "Workflow":
        seen = set()
        for task in self.tasks:
            if task.id in seen:
                raise ValueError(f"Invalid Workflow: Duplicate task ID found '{task.id}'")
            seen.add(task.id)
        return self

    def get_task(self, task_id: str) -> Task | None:
        """Helper to find a Task by ID."""
        return next((t for t in self.tasks if t.id == task_id), None)

    @property
    def entry_tasks(self) -> list[Task]:
        """
        Returns tasks that have NO dependencies on other tasks. 
        These are the starting nodes of the DAG and can be run first.
        """
        from lore.core.bindings import ReferenceBinding

        entry_nodes = []
        for task in self.tasks:
            has_upstream_dependency = any(
                isinstance(b, ReferenceBinding) 
                for bindings in task.inputs.values() 
                for b in bindings
            )

            if not has_upstream_dependency:
                entry_nodes.append(task)

        return entry_nodes
