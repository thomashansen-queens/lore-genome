"""
A Workflow in LoRe is a self-executing chain of Tasks. Topographically, it is a
directed acyclic graph (DAG) where the nodes are Tasks and the edges represent
data dependencies (i.e. Task A produces an Artifact that is consumed by Task B).
"""

from pydantic import BaseModel, Field
from typing import Any


class WorkflowStep(BaseModel):
    """
    A single node in a Workflow
    """
    id: str
    task_key: str
    # config can be a static value OR a reference to another step
    # i.e. {"source": "ref:step1.output"}
    config: dict[str, Any] = Field(default_factory=dict)


class Workflow(BaseModel):
    """
    A Workflow is a collection of WorkflowSteps and their dependencies.
    """
    id: str
    name: str
    description: str = ""
    steps: list[WorkflowStep] = Field(default_factory=list)

    def get_dependencies(self, step_id: str) -> list[str]:
        """
        Analyzes the Workflow to see which steps must run before this one
        Assuming ':' syntax for referencing other steps (e.g. "ref:step1.output")
        """
        deps = []
        step = next(s for s in self.steps if s.id == step_id)
        for val in step.config.values():
            if isinstance(val, str) and val.startswith("ref:"):
                deps.append(val.split(":")[1].split(".")[0])
        return deps
