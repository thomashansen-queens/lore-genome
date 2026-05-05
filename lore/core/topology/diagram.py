"""
Translates Workflow objects into Graph structures for visualization
"""

from typing import Any, Iterable, Literal

from lore.core.bindings import ReferenceBinding, UserInputBinding
from lore.core.tasks.models import Task
from lore.core.tasks.registry import TaskRegistry
from lore.viz.graph import DiagramResult, Graph, DagLayout


def generate_dag_diagram(
    tasks: Iterable[Task],
    task_registry: TaskRegistry,
    direction: Literal["TB", "LR"] = "LR",
) -> DiagramResult:
    """
    Reads a Workflow's DAG and generates layout coordinates for rendering.
    """
    graph = Graph()

    # 1. Add all Tasks as nodes
    for task in tasks:
        task_def = task_registry.get_safe(task.registry_key)

        default_name = task_def.name if task_def else "Unknown Task"
        label = task.name if task.name else default_name
        graph.add_node(
            node_id=task.id,
            label=label,
            payload=task,  # include the full task config
        )

    # 2. Add ReferenceBindings as edges
    #    also adds virtual nodes for UserInputBindings
    for task in tasks:
        task_def = task_registry.get_safe(task.registry_key)

        for key, bindings in task.inputs.items():
            input_label = key.replace("_", " ").title()

            if task_def:
                try:
                    _, extra = task_def.field_meta(key)
                    input_label = extra.get("label") or input_label
                except (KeyError, ValueError):
                    # fallback to default label if metadata is missing or malformed
                    pass

            for b in bindings:
                if isinstance(b, ReferenceBinding):
                    graph.add_edge(
                        source_id=b.source_id,
                        target_id=task.id,
                        label=b.output_key,
                    )

                elif isinstance(b, UserInputBinding):
                    # 1. Invent a unique ID for this virtual node
                    virtual_id = f"input_{task.id}_{key}"

                    # 2. Add the virtual node with a special label
                    graph.add_node(
                        node_id=virtual_id,
                        label=input_label,
                        width=120.0,
                        height=40.0,
                        is_virtual=True,
                    )

                    # 3. Draw the edge from the virtual node to the task
                    graph.add_edge(
                        source_id=virtual_id,
                        target_id=task.id,
                        label=key,
                    )

    # 3. Hand the generic graph to the layout engine
    layout = DagLayout(graph, direction=direction)
    return layout.compute()
