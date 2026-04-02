"""
Directed acyclic graph (DAG) validation and sorting logic for Workflows.

TODO:
- Enable parallelization with Kahn's Algorithm (BFS)
- Add logic gates
- Dynamic task generation (e.g. for each item in a list, run these steps)
- Sub-graphs / nested workflows
"""

from typing import List
from lore.core.workflows.models import Workflow, WorkflowStep


class DAGValidationError(Exception):
    """
    Raised when a Workflow is mathematically invalid
    (e.g. contains cycles or missing dependencies).
    """

    pass


def solve_dag(dependency_map: dict[str, list[str]]) -> list[str]:
    """
    Topological sort using Depth-First Search (DFS).
    Takes a dictionary mapping {node_id: [list_of_upstream_node_ids]}.
    Returns a topologically sorted list of node_ids suitable for linear execution.
    """
    visited = set()
    recursion_stack = set()
    sorted_ids = []

    def visit(node_id: str):
        """Depth-first sort (DFS)"""
        if node_id in recursion_stack:
            raise DAGValidationError(f"Cycle detected at node: '{node_id}'")
        if node_id in visited:
            return  # already processed this branch

        # 1. Start at a node: Mark as currently exploring
        recursion_stack.add(node_id)

        # 2. Build the stack: Traverse all upstream nodes
        for upstream_id in dependency_map.get(node_id, []):
            visit(upstream_id)

        # 3. Resolve the stack: Done exploring this branch
        recursion_stack.remove(node_id)
        visited.add(node_id)
        sorted_ids.append(node_id)

    # Execute the check for all nodes
    for node_id in dependency_map:
        if node_id not in visited:
            visit(node_id)

    return sorted_ids


def validate_and_sort_workflow(workflow: Workflow) -> List[WorkflowStep]:
    """
    Validates the Workflow DAG for continuity and cycles.
    Returns a topologically sorted list of steps for linear execution.
    """
    # 1. Continuity check: All upstream steps exist
    step_ids = {step.id for step in workflow.steps}
    for step in workflow.steps:
        for upstream_id in step.upstream_step_ids:
            if upstream_id not in step_ids:
                raise DAGValidationError(
                    f"Step '{step.id}' references unknown upstream step: '{upstream_id}'"
                )

    # 2. Topological sort
    step_map = {step.id: step for step in workflow.steps}
    dependency_map = {step.id: step.upstream_step_ids for step in workflow.steps}

    sorted_ids = solve_dag(dependency_map)
    sorted_steps = [step_map[step_id] for step_id in sorted_ids]

    return sorted_steps
