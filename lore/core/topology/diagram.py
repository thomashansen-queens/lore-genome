"""
Translates Workflow objects into Graph structures for visualization
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from lore.core.bindings import ReferenceBinding, UserInputBinding
from lore.core.tasks.registry import TaskRegistry
from lore.viz.graph import Graph, GraphNode, GraphPort, SugiyamaLayout, Direction

if TYPE_CHECKING:
    from lore.core.tasks.models import Task


@dataclass
class TaskNode(GraphNode):
    """A standard LoRē Task."""
    width: float = 200.0
    height: float = 60.0
    node_type: str = "task"
    css_class: str = "node-task"


@dataclass
class UserInputNode(GraphNode):
    """An entry point for a Workflow."""
    width: float = 120.0
    height: float = 40.0
    node_type: str = "input"
    css_class: str = "node-input"


@dataclass
class ConditionNode(GraphNode):
    """A logic gate. TODO: Implement this!"""
    width: float = 100.0
    height: float = 100.0
    node_type: str = "diamond"
    css_class: str = "node-condition"


def generate_dag_diagram(
    tasks: Iterable["Task"],
    task_registry: TaskRegistry,
    direction: Direction = Direction.LR,
) -> Graph:
    """
    Reads a LoRē Task DAG and generates layout coordinates for rendering.
    """
    graph = Graph()

    # 1. Add all Tasks as nodes and build their ports
    for task in tasks:
        task_def = task_registry.get_safe(task.registry_key)

        default_name = task_def.name if task_def else "Unknown Task"
        label = task.name if task.name else default_name
        in_ports = []
        out_ports = []

        if task_def:
            # 1. Build input ports from the TaskDefinition's input model
            for k in task_def.input_model.model_fields.keys():
                # A. Check if this input is bound
                input_bindings = task.inputs.get(k) or []
                if not any(
                    isinstance(b, (ReferenceBinding, UserInputBinding))
                    for b in input_bindings
                ):
                    # B. If unbound, is it required? (i.e. has no default and isn't optional)
                    field_info = task_def.input_model.model_fields[k]
                    if not field_info.is_required():
                        continue

                # C. Get label from TaskDefinition metadata if available
                try:
                    _, extra = task_def.field_meta(k)
                    port_label = extra.get("label") or k.replace("_", " ").capitalize()
                except (KeyError, ValueError):
                    port_label = k.replace("_", " ").capitalize()
                
                in_ports.append(GraphPort(id=k, label=port_label))

            # 2. Build output ports
            # TODO: Enrich output ports with metadata from actual outputs? Or compute dynamically from Task instance?
            for k in task_def.output_model.model_fields.keys():
                try:
                    _, extra = task_def.field_meta(k, is_output=True)
                    port_label = extra.get("label") or k.replace("_", " ").capitalize()
                except (KeyError, ValueError):
                    port_label = k.replace("_", " ").capitalize()
                out_ports.append(GraphPort(id=k, label=port_label))

        task_node = TaskNode(
            id=task.id,
            label=label,
            payload=task,
            inputs=in_ports,
            outputs=out_ports,
        )
        graph.add_node(task_node)

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
                    # TODO: Enrich edges with actual artifact metadata?
                    pin_label = b.artifact_id[:6] if b.artifact_id else ""
                    graph.add_edge(
                        source_id=b.source_id,
                        target_id=task.id,
                        label=pin_label,
                        source_port=b.output_key,
                        target_port=key,
                        payload={
                            "type": "reference",
                            "is_pinned": bool(b.artifact_id),
                        }
                    )

                elif isinstance(b, UserInputBinding):
                    # 1. Invent a unique ID for this virtual node
                    virtual_id = f"input_{task.id}_{key}"

                    # 2. Add the virtual node with a special label
                    user_input_node = UserInputNode(
                        id=virtual_id,
                        label=input_label,
                        width=120.0,
                        height=40.0,
                    )
                    graph.add_node(user_input_node)

                    # 3. Draw the edge from the virtual node to the task
                    graph.add_edge(
                        source_id=virtual_id,
                        target_id=task.id,
                        label="",
                        target_port=key,
                        payload={
                            "type": "user_input",
                        }
                    )

    # 3. Hand the generic graph to the layout engine
    layout = SugiyamaLayout(graph, direction=direction)
    return layout.compute()
