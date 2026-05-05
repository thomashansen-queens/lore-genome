"""
Directed acyclic graph (DAG) validation and sorting logic for Workflows.

TODO:
- Enable parallelization with Kahn's Algorithm (BFS)
- Add logic gates
- Dynamic task generation (e.g. for each item in a list, run these steps)
- Sub-graphs / nested workflows
- Editable edges (simple logic/retries can go there)
"""

from typing import Any
from lore.core.topology.matcher import is_output_compatible
from lore.core.tasks.models import Task
from lore.core.bindings import ReferenceBinding
from lore.core.tasks.registry import task_registry


class DAGValidationError(Exception):
    """
    Raised when a Graph is mathematically invalid
    (e.g. contains cycles or missing dependencies).
    """

    pass


def get_parent_ids(task: Task) -> list[str]:
    """
    Helper to extract all upstream dependency IDs from a Task's inputs.
    """
    upstream = set()
    for bindings in task.inputs.values():
        for b in bindings:
            if isinstance(b, ReferenceBinding):
                upstream.add(b.source_id)
    return list(upstream)


def get_task_descendants(tasks: list[Task], start_task_id: str) -> set[str]:
    """
    Finds all tasks that rely on the start_task_id, directly or indirectly.
    """
    # 1. Build a quick adjacency list: { source_id: [dependent_task_ids] }
    adj = {t.id: [] for t in tasks}
    for t in tasks:
        for bindings in t.inputs.values():
            for b in bindings:
                # Duck-type check or isinstance check depending on your setup
                if isinstance(b, ReferenceBinding) and b.source_id in adj:
                    adj[b.source_id].append(t.id)

    # 2. Breadth-first search (BFS) to find all descendants
    descendants = set()
    queue = [start_task_id]

    while queue:
        current = queue.pop(0)
        for child_id in adj.get(current, []):
            if child_id not in descendants:
                descendants.add(child_id)
                queue.append(child_id)

    return descendants


def _dfs_sort_dag(dependency_map: dict[str, list[str]]) -> list[str]:
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


def sort_tasks_topologically(tasks: list[Task]) -> list[Task]:
    """
    Validates a list of Tasks for continuity and cycles.
    Returns a topologically sorted list of Tasks for linear execution.
    """
    task_ids = {task.id for task in tasks}
    task_map = {task.id: task for task in tasks}
    dependency_map = {}

    # 1. Continuity check & build dependency map
    for task in tasks:
        upstream_ids = get_parent_ids(task)
        for upstream_id in upstream_ids:
            if upstream_id not in task_ids:
                raise DAGValidationError(
                    f"Task '{task.id}' references unknown upstream task: '{upstream_id}'"
                )
        dependency_map[task.id] = upstream_ids

    # 2. Topological sort
    sorted_ids = _dfs_sort_dag(dependency_map)

    return [task_map[tid] for tid in sorted_ids]


def find_valid_upstream_tasks(
    current_task_id: str | None,
    tasks: list[Task],
    field_extra: dict,
) -> list[dict[str, Any]]:
    """
    Looks at previous tasks in a topological sequence and checks their theoretical 
    output schemas to see if they can satisfy the current field's data requirements.
    """
    valid_upstream = []
    # 1. Find invalid tasks (self + descendants)
    invalid_ids = set()
    if current_task_id:
        invalid_ids.add(current_task_id)
        invalid_ids.update(get_task_descendants(tasks, current_task_id))

    # 2. Topologically sort the Tasks
    try:
        search_list = sort_tasks_topologically(tasks)
    except Exception:
        search_list = tasks  # Fallback gracefully if the graph is invalid (e.g. cycles)

    # 3. Check compatibility (in sorted order)
    for upstream_task in search_list:
        # A. Skip self and descendants to prevent cycles
        if upstream_task.id in invalid_ids:
            continue

        # B. If task definition is unknown (imported workflows), can't offer it as an option
        upstream_def = task_registry.get_safe(upstream_task.registry_key)
        if not upstream_def or not upstream_def.output_model:
            continue

        # C. Defer to the Matcher engine to check if output can satisfy the input requirements
        valid_outputs = []
        for out_key, out_field in upstream_def.output_model.model_fields.items():
            _, out_extra = upstream_def.field_meta(out_key, is_output=True)

            out_adapters = upstream_def.get_adapters_for_output(out_key)
            if is_output_compatible(
                source_extra=out_extra,
                target_extra=field_extra,
                adapters=out_adapters,
            ):
                valid_outputs.append(out_key)

        if valid_outputs:
            valid_upstream.append({
                "task": upstream_task,
                "valid_outputs": valid_outputs
            })

    return valid_upstream
