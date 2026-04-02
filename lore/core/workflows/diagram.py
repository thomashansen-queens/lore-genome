"""
Translates Workflow objects into Graph structures for visualization
"""

from typing import Literal

from lore.core.tasks.registry import TaskRegistry
from lore.core.workflows.models import Workflow
from lore.viz.graph import DiagramResult, Graph, DagLayout

def generate_workflow_diagram(
    workflow: Workflow,
    task_registry: TaskRegistry,
    direction: Literal["TB", "LR"] = "LR",
) -> DiagramResult:
    """
    Reads a Workflow's DAG and generates layout coordinates for rendering.
    """
    graph = Graph()

    # 1. Add all steps as nodes
    for step in workflow.steps:
        task_def = task_registry.get_safe(step.task_key)

        default_name = task_def.name
        label = step.name if step.name else default_name
        graph.add_node(
            node_id=step.id,
            label=label,
            payload=step,  # include the full step config
        )

    # 2. Add ReferenceBindings as edges
    #    also adds virtual nodes for UserInputBindings
    for step in workflow.steps:
        task_def = task_registry.get_safe(step.task_key)

        for key, bindings in step.inputs.items():
            default_label = key.replace("_", " ").title()

            _, extra = task_def.field_meta(key)
            input_label = extra.get("label") or default_label

            for b in bindings:
                if b.type == "reference":
                    graph.add_edge(
                        source_id=b.source_id,
                        target_id=step.id,
                        label=b.output_key,
                    )

                elif b.type == "user_input":
                    # 1. Invent a unique ID for this virtual node
                    virtual_id = f"input_{step.id}_{key}"

                    # 2. Add the virtual node with a special label
                    graph.add_node(
                        node_id=virtual_id,
                        label=input_label,
                        width=120.0,
                        height=40.0,
                        is_virtual=True,
                    )

                    # 3. Draw the edge from the virtual node to the step
                    graph.add_edge(
                        source_id=virtual_id,
                        target_id=step.id,
                        label=key,
                    )

    # 3. Hand the generic graph to the layout engine
    layout = DagLayout(graph, direction=direction)
    return layout.compute()
