"""
Directed acyclic graph (DAG) validation and sorting logic for Workflows.

TODO:
- Enable parallelization with Kahn's Algorithm (BFS)
- Add logic gates
- Dynamic task generation (e.g. for each item in a list, run these steps)
- Sub-graphs / nested workflows
- Editable edges (simple logic/retries can go there)
"""

import logging
from typing import Any, TYPE_CHECKING
from lore.core.bindings import ReferenceBinding

if TYPE_CHECKING:
    from lore.core.sessions.session import Session
    from lore.core.tasks.models import Task
    from lore.core.workflows.models import Workflow

logger = logging.getLogger(__name__)


class DAGValidationError(Exception):
    """
    Raised when a Graph is mathematically invalid
    (e.g. contains cycles or missing dependencies).
    """

    pass


def get_parent_ids(task: "Task") -> list[str]:
    """
    Helper to grab immediately upstream dependency IDs from a Task's inputs.
    """
    upstream = set()
    for bindings in task.inputs.values():
        for b in bindings:
            if isinstance(b, ReferenceBinding):
                upstream.add(b.source_id)
    return list(upstream)


def get_task_descendants(
    tasks: list["Task"],
    start_task_id: str,
    generations: int | None = None,
) -> set[str]:
    """
    Finds downstream tasks that rely on the start_task_id.
    If generations is set, limits the depth of the search (e.g. 1 = immediate children)
    """
    # 1. Build a quick adjacency list: { source_id: [dependent_task_ids] }
    adj = {t.id: [] for t in tasks}
    for t in tasks:
        for parent_id in get_parent_ids(t):
            if parent_id in adj:
                adj[parent_id].append(t.id)

    # 2. Breadth-first search (BFS) to find all descendants
    descendants = set()
    queue = [(start_task_id, 0)]

    while queue:
        current, depth = queue.pop(0)

        # Stop exploring this branch if we've hit the generation limit
        if generations is not None and depth >= generations:
            break

        for child_id in adj.get(current, []):
            if child_id not in descendants:
                descendants.add(child_id)
                queue.append((child_id, depth + 1))

    return descendants


def get_task_ancestors(
    tasks: list["Task"],
    start_task_id: str,
    generations: int | None = None,
) -> set[str]:
    """
    Finds upstream tasks that the start_task_id relies on.
    If generations is set, limits the depth of the search (e.g. 1 = immediate parents)
    """
    # 1. Build a quick reverse adjacency list: { task_id: [parent_ids] }
    rev_adj = {t.id: get_parent_ids(t) for t in tasks}

    # 2. Breadth-first search (BFS) to find all ancestors
    ancestors = set()
    queue = [(start_task_id, 0)]

    while queue:
        current, depth = queue.pop(0)

        # Stop exploring this branch if we've hit the generation limit
        if generations is not None and depth >= generations:
            break

        for parent_id in rev_adj.get(current, []):
            if parent_id not in ancestors:
                ancestors.add(parent_id)
                queue.append((parent_id, depth + 1))

    return ancestors


def sort_dag_dfs(dependency_map: dict[str, list[str]]) -> list[str]:
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


def sort_tasks_topologically(tasks: list["Task"]) -> list["Task"]:
    """
    Validates a list of Tasks for continuity and cycles.
    Returns a topologically sorted list of Tasks for linear execution.
    Missing upstream dependencies are gracefully ignored (e.g. if an 
    upstream Task is deleted).
    """
    task_ids = {task.id for task in tasks}
    task_map = {task.id: task for task in tasks}
    dependency_map = {}

    # 1. Continuity check & build dependency map
    for task in tasks:
        upstream_ids = get_parent_ids(task)
        valid_upstream_ids = []

        for upstream_id in upstream_ids:
            if upstream_id not in task_ids:
                logger.warning(
                    "Task '%s' is missing upstream dependency '%s'.",
                    task.id,
                    upstream_id,
                )
            else:
                valid_upstream_ids.append(upstream_id)

        dependency_map[task.id] = valid_upstream_ids

    # 2. Topological sort
    sorted_ids = sort_dag_dfs(dependency_map)

    return [task_map[tid] for tid in sorted_ids]


def find_valid_upstream_tasks(
    container: "Session | Workflow",
    current_task_id: str | None,
    tasks: list["Task"],
    field_extra: dict,
) -> list[dict[str, Any]]:
    """
    Looks at previous tasks in a topological sequence and checks their theoretical 
    output schemas to see if they can satisfy the current field's data requirements.
    Args:
    - container: Are we looking in a Session or a Workflow (affects dynamic type resolution)
    - current_task_id: The ID of the Task we're trying to find upstream links for
    - tasks: The list of all tasks in the current Session or Workflow
    - field_extra: The metadata dict for the input slot we're trying to satisfy
    """
    # Lazy import to avoid circular dependency
    from lore.core.adapters import adapter_registry
    from lore.core.tasks.registry import task_registry

    valid_upstream = []

    # 1. Exclude self + descendants from search
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

            # i. Resolve dynamic Passthrough types with static TaskDefinition metadata
            source_extra = out_extra.copy()
            source_extra.update(upstream_task.resolve_output_type(out_key, container))
            data_type = source_extra.get("data_type")
            is_target_artifact = field_extra.get("is_artifact", False)
            is_source_artifact = source_extra.get("is_artifact", True)

            # Mismatch: Artifacts cannot plug into Python types and vice versa
            if is_target_artifact != is_source_artifact:
                continue

            # ii. Path A: Artifact matching (LoRē semantic type + format through adapters)
            from lore.core.topology.matcher import is_output_compatible, is_primitive_compatible

            if is_target_artifact:
                out_adapters = []
                if isinstance(data_type, str):
                    out_adapters = adapter_registry.get_adapters_by_type(
                        data_type=data_type,
                        extension=source_extra.get("format", "*"),
                    )

                # Check if any of the candidate adapters can satisfy the input requirements
                if is_output_compatible(
                    source_extra=source_extra,
                    target_extra=field_extra,
                    adapters=out_adapters,
                ):
                    valid_outputs.append(out_key)

            # iii. Path B: Primitive matching (strict Python type compatibility)
            else:
                if is_primitive_compatible(
                    source_type=data_type,
                    target_extra=field_extra,
                ):
                    valid_outputs.append(out_key)

        if valid_outputs:
            valid_upstream.append({
                "task": upstream_task,
                "valid_outputs": valid_outputs
            })

    return valid_upstream
