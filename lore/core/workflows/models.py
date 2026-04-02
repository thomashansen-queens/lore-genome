"""
Defines the component parts of a Workflow.
A Workflow is a DAG (directed acyclic graph) of Tasks.
"""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator

from lore.core.bindings import Binding, ReferenceBinding


class WorkflowStep(BaseModel):
    """
    A single node in a workflow. Represents one Task execution.
    """

    id: str = Field(..., description="Unique ID for this step within the workflow.")
    task_key: str = Field(..., description="Registry key of the Task to execute.")
    inputs: dict[str, list[Binding]] = Field(
        default_factory=dict,
        description=(
            "Configuration for the Task. Can include static values or references to other "
            "steps (e.g. {'e_value': 0.001, 'query_fasta': 'ref:step_1.output_fasta'})"
        ),
    )

    name: str | None = Field(None, description="Human-readable name for this specific step.")
    description: str = Field(
        default="",
        description="Optional longer explanation of how/what/why this step exists.",
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def inputs_to_bindings(cls, v: Any) -> Any:
        if isinstance(v, dict):
            from lore.core.bindings import wrap_in_bindings

            return wrap_in_bindings(v)
        return v

    @property
    def upstream_step_ids(self) -> list[str]:
        """
        Parses the inputs to find which step IDs this step depends on.
        """
        upstream_ids = set()
        for bindings in self.inputs.values():
            for b in bindings:
                if isinstance(b, ReferenceBinding):
                    upstream_ids.add(b.source_id)
        return list(upstream_ids)


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

    steps: list[WorkflowStep] = Field(
        default_factory=list,
        description="The individual steps (Tasks) that make up this workflow.",
    )

    @model_validator(mode="after")
    def check_unique_step_ids(self) -> "Workflow":
        seen = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"Invalid Workflow: Duplicate step ID found '{step.id}'")
            seen.add(step.id)
        return self

    def get_step(self, step_id: str) -> WorkflowStep | None:
        """Helper to find a step by ID."""
        return next((s for s in self.steps if s.id == step_id), None)

    @property
    def entry_steps(self) -> list[WorkflowStep]:
        """
        Returns steps that have NO dependencies on other steps. These can be run
        first and concurrently.
        """
        return [s for s in self.steps if not s.upstream_step_ids]
