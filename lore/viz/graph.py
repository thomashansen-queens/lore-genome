"""
A little DIY graph visualization utility for LoRe task graphs. Implements the
Sugiyama Method for layered graph drawing. Could be useful for figures down the 
line.

Reference:
Sugiyama method
Kozo Sugiyama, Shoji Tagawa, and Mitsuhiko Toda. 1981. 
Methods for visual understanding of hierarchical system structures.
https://ieeexplore.ieee.org/document/4308636/

Also consulted:
Mazetti & Sorensson, 2012
https://publications.lib.chalmers.se/records/fulltext/161388.pdf

Four steps:
1. Cycle removal (Workflows are DAGs so this is not needed)
2. Layer assignment
3. Vertex ordering (minimize edge crossings)
4. Coordinate assignment (vertex placement creates a balanced graph)
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Direction(StrEnum):
    LR = "LR"
    TB = "TB"


@dataclass
class GraphPort:
    """
    A single input or output port/slot on a node. A node can have 0 or more
    ports, which can indicate different data types or connections.
    TODO: Should this have a payload?
    """
    id: str     # e.g., "protein_sequence"
    label: str  # e.g., "Protein Sequence"


# --- Graph Nodes/Vertices ---


@dataclass
class GraphNode:
    """
    The base node type in the graph. Defines Node identity, ports, and geometry
    for the layout engine (calculated by a layout algorithm).
    _layout_dir and the main_* and cross_* properties allows the same code
    to create left-right and top-bottom layouts without duplicating logic.
    """
    # Identity
    id: str
    label: str
    node_type: str = "base"
    payload: Any = None  # Arbitrary data associated with a node
    layer: int = 0

    # Ports/Slots/Connections
    inputs: list[GraphPort] = field(default_factory=list)
    outputs: list[GraphPort] = field(default_factory=list)

    # Geometry
    x: float = 0.0
    y: float = 0.0
    width: float = 200.0
    height: float = 60.0
    _layout_dir: Direction = Direction.LR  # set by layout engine for main/cross context

    # Presentation
    node_type: str = "base"
    css_class: str = "node-default"

    @property
    def main_pos(self) -> float:
        return self.x if self._layout_dir == Direction.LR else self.y

    @main_pos.setter
    def main_pos(self, value: float):
        if self._layout_dir == Direction.LR:
            self.x = value
        else:
            self.y = value

    @property
    def cross_pos(self) -> float:
        if self._layout_dir == Direction.LR:
            return self.y
        else:
            return self.x

    @cross_pos.setter
    def cross_pos(self, value: float):
        if self._layout_dir == Direction.LR:
            self.y = value
        else:
            self.x = value

    @property
    def main_size(self) -> float:
        return self.width if self._layout_dir == Direction.LR else self.height

    @property
    def cross_size(self) -> float:
        return self.height if self._layout_dir == Direction.LR else self.width


@dataclass
class DummyNode(GraphNode):
    node_type: str = "dummy"
    width: float = 0.0
    height: float = 0.0
    css_class: str = "node-dummy"


# --- Graph edges ---

@dataclass
class GraphEdgeSegment:
    """A physical routing line between exactly two adjacent layers."""
    source_id: str
    target_id: str
    source_port: str | None = None
    target_port: str | None = None
    path_d: str = ""


@dataclass
class GraphEdge:
    """
    An edge connecting two nodes. Can carry a label and payload of data.
    """
    source_id: str
    target_id: str
    label: str
    source_port: str | None = None  
    target_port: str | None = None
    payload: Any = None

    # Geometry for layout (populated by the layout engine)
    segments: list[GraphEdgeSegment] = field(default_factory=list)


class Graph:
    """
    Pure mathematical graph structure of nodes and edges.
    Once passed through a layout engine, holds bound geometry.
    """
    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

        self.canvas_width: float = 0.0
        self.canvas_height: float = 0.0

    def add_node(self, node: GraphNode):
        """
        Add a node to the graph.
        """
        self.nodes[node.id] = node

    def add_edge(self, source_id: str, target_id: str, label: str, **kwargs):
        """
        Add an edge to the graph. Optional kwargs:
        - payload: Arbitrary data included with edge
        - source_port: The port on the source node
        - target_port: The port on the target node
        """
        self.edges.append(GraphEdge(source_id=source_id, target_id=target_id, label=label, **kwargs))


class SugiyamaLayout:
    """
    Layout engine for the Sugiyama method of drawing DAGs.
    Takes a Graph topology and stamps it with X,Y geometry.
    """
    def __init__(
        self,
        graph: Graph,
        direction: Direction = Direction.LR,
        arrow_margin: float = 5.0,
        gap_nodes: float = 25.0,
        gap_layers: float = 50.0,
        **overrides,
    ):
        self.graph = graph
        self.direction = direction  # top-bottom or left-right
        self.arrow_margin = arrow_margin
        self.gap_nodes = gap_nodes  # spacing between nodes
        self.gap_layers = gap_layers  # spacing between layers
        self._overrides = overrides  # useful for testing

    @property
    def padding(self) -> float:
        return self._overrides.get("padding", self.gap_nodes)

    @property
    def main_step(self) -> float:
        """The grid spacing in the direction of flow (e.g. Column Width in LR)"""
        if "main_step" in self._overrides:
            return self._overrides["main_step"]
        max_main = max((n.main_size for n in self.graph.nodes.values()), default=200.0)
        return max_main + self.gap_layers

    def compute(self) -> Graph:
        """
        Runs the layout algorithm and returns the mutated Graph.
        """
        # 1. Inject the flexbox-style main/cross context into nodes
        for node in self.graph.nodes.values():
            node._layout_dir = self.direction

        # 2. Run the Sugiyama method steps
        self._assign_layers()
        self._create_proper_hierarchy()
        self._build_layer_map()
        self._order_vertices()

        # 3. Apply geometry
        self._assign_coordinates()  # assigns (x,y) to nodes based on layer and order
        self._straighten_dummies()  # de-lumpifies dummy nodes to make smooth edges
        self._route_edges()         # calculates bezier control points for edge paths

        return self.graph

    def _assign_layers(self):
        """
        Step 1a: Assign nodes to layers using Longest Path (Kahn's Algorithm)
        Guarantees O(V + E) for DAGs
        """
        # 1. Calculate in-degrees for all nodes
        in_degree = {node_id: 0 for node_id in self.graph.nodes}
        for edge in self.graph.edges:
            if edge.target_id in in_degree:
                in_degree[edge.target_id] += 1

        # 2. Find roots (in-degree 0) and put them in layer 0
        queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
        for n_id in queue:
            self.graph.nodes[n_id].layer = 0

        # 3. Propagate layer assignment downwards
        while queue:
            u_id = queue.pop(0)
            u_layer = self.graph.nodes[u_id].layer

            for edge in self.graph.edges:
                if edge.source_id == u_id:
                    v_id = edge.target_id
                    if v_id in self.graph.nodes:
                        # Push the child down to the longest path from any parent
                        self.graph.nodes[v_id].layer = max(
                            self.graph.nodes[v_id].layer, u_layer + 1
                        )
                        in_degree[v_id] -= 1
                        if in_degree[v_id] == 0:
                            queue.append(v_id)

        # 4. Kahn's algorithm is done, now adjust terminal nodes to go one layer
        #    up from their children to reduce dummy nodes in the next step
        for node in self.graph.nodes.values():
            if node.node_type in ("input", "start", "source", "root"):
                targets = [e.target_id for e in self.graph.edges if e.source_id == node.id]
                valid_target_layers = [
                    self.graph.nodes[t].layer for t in targets if t in self.graph.nodes
                ]

                if valid_target_layers:
                    # one layer before earliest target
                    node.layer = max(0, min(valid_target_layers) -  1)

    def _create_proper_hierarchy(self):
        """
        Step 1b: Convert to a 'proper hierarchy' by breaking long edges with 
        dummy nodes.
        """
        for edge in self.graph.edges:
            edge.segments = []

            if edge.source_id not in self.graph.nodes or edge.target_id not in self.graph.nodes:
                continue

            source = self.graph.nodes[edge.source_id]
            target = self.graph.nodes[edge.target_id]
            span = target.layer - source.layer

            if span <= 1:
                # Normal edge
                edge.segments.append(GraphEdgeSegment(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    source_port=edge.source_port,
                    target_port=edge.target_port,
                ))
            else:
                # Long span, use dummy nodes
                prev_id = edge.source_id

                for current_layer in range(source.layer + 1, target.layer):
                    # 1. Create a dummy node
                    dummy_id = f"dummy_{edge.source_id}_{edge.target_id}_{current_layer}"
                    dummy_node = DummyNode(
                        id=dummy_id,
                        label="",
                        layer=current_layer,
                    )
                    self.graph.add_node(dummy_node)

                    # 2. Link previous node to dummy
                    is_first_segment = (prev_id == edge.source_id)
                    edge.segments.append(GraphEdgeSegment(
                        source_id=prev_id,
                        target_id=dummy_id,
                        source_port=edge.source_port if is_first_segment else None,
                        target_port=None,
                    ))
                    prev_id = dummy_id

                # 3. Link last dummy to target
                edge.segments.append(GraphEdgeSegment(
                    source_id=prev_id,
                    target_id=edge.target_id,
                    source_port=None,
                    target_port=edge.target_port,
                ))

    def _build_layer_map(self):
        """Helper to group nodes by layer."""
        self._layer_map = {}
        for node in self.graph.nodes.values():
            self._layer_map.setdefault(node.layer, []).append(node)

    def _order_vertices(self):
        """Step 2: Order vertices within layers to minimize edge crossings (barycenter heuristic)."""
        if not self._layer_map:
            return  # No nodes to order

        max_layer = max(self._layer_map.keys())
        # Build adjacency by direction
        parents = {}   # node_id -> [source_ids]
        children = {}  # node_id -> [target_ids]
        for e in self.graph.edges:
            children.setdefault(e.source_id, []).append(e.target_id)
            parents.setdefault(e.target_id, []).append(e.source_id)

        # Sweep downwards
        for _ in range(4):
            # Forward pass: Sort layer by average position of parents in previous layer
            for l in range(1, max_layer + 1):
                nodes = self._layer_map.get(l, [])
                if len(nodes) <= 1:
                    continue
                prev_pos = {n.id: i for i, n in enumerate(self._layer_map.get(l - 1, []))}

                def bary_up(n, current_idx):
                    hits = [prev_pos[pid] for pid in parents.get(n.id, []) if pid in prev_pos]
                    return sum(hits) / len(hits) if hits else current_idx

                nodes_with_idx = list(enumerate(nodes))
                nodes_with_idx.sort(key=lambda x: bary_up(x[1], x[0]))
                self._layer_map[l] = [n for idx, n in nodes_with_idx]

            # Backward pass: Sort layer by average position of children in next layer
            for l in range(max_layer - 1, -1, -1):
                nodes = self._layer_map.get(l, [])
                if len(nodes) <= 1:
                    continue
                next_pos = {n.id: i for i, n in enumerate(self._layer_map.get(l + 1, []))}

                def bary_down(n, current_idx):
                    hits = [next_pos[cid] for cid in children.get(n.id, []) if cid in next_pos]
                    return sum(hits) / len(hits) if hits else current_idx

                nodes_with_idx = list(enumerate(nodes))
                nodes_with_idx.sort(key=lambda x: bary_down(x[1], x[0]))
                self._layer_map[l] = [n for idx, n in nodes_with_idx]

    def _assign_coordinates(self):
        """
        Step 4: Assign (x,y) coordinates to nodes based on layer and order.
        In an LR layout, 'main' is X and 'cross' is Y.
        """
        if not self.graph.nodes:
            return

        # 1. Calculate column cross sizes for centering
        padding = self.padding
        gap = self.gap_nodes

        layer_cross_size = {
            layer_idx: sum(n.cross_size for n in nodes) + max((len(nodes) - 1), 0) * gap
            for layer_idx, nodes in self._layer_map.items()
        }

        max_cross = max(layer_cross_size.values(), default=0) + (padding * 2)
        max_main = 0.0

        # 2. Calculate where this layer should start for centering
        for layer_idx, nodes in self._layer_map.items():
            # current_cross = center - half of layer size
            current_cross = (max_cross / 2) - (layer_cross_size[layer_idx] / 2)

            for node in nodes:
                # Center node in its layer
                main_offset = (self.main_step - node.main_size) / 2
                node.main_pos = padding + (layer_idx * self.main_step) + main_offset

                # Cross position comes from cursor
                node.cross_pos = current_cross

                # Advance the cursor and bounding box
                current_cross += node.cross_size + gap
                max_main = max(max_main, node.main_pos + node.main_size)

        # 4. Final canvas size with padding into Graph object
        if self.direction == Direction.LR:
            self.graph.canvas_width = max_main + (padding * 2)
            self.graph.canvas_height = max_cross
        else:
            self.graph.canvas_width = max_cross
            self.graph.canvas_height = max_main + (padding * 2)

    def _straighten_dummies(self):
        """
        Step 4b: Aligns dummy nodes along a perfect linear interpolation 
        between their original source and target ports to eliminate wavy edges.
        """
        for edge in self.graph.edges:
            if len(edge.segments) <= 2:
                continue

            # Extract just the dummy nodes in this edge's path
            dummy_ids = [s.source_id for s in edge.segments if s.source_id.startswith("dummy_")]
            if not dummy_ids:
                continue

            dummies = [self.graph.nodes[did] for did in dummy_ids]

            # "Rubber-band" heuristic: flatten to average cross poosition
            # TODO: Use a max-min approach to truly avoid collisions
            avg_cross = sum(d.cross_pos for d in dummies) / len(dummies)

            for dummy in dummies:
                dummy.cross_pos = avg_cross

    def _get_port_coords(
        self,
        node: GraphNode,
        port_id: str | None,
        is_source: bool,
    ) -> tuple[float, float]:
        """Calculates the exact (main, cross) coordinate for an edge connection."""
        # 1. Early exit: 'Virtual nodes' are routing points
        if node.node_type == "dummy":
            return node.main_pos, node.cross_pos

        # 2. Main Axis (side of the node bounding box)
        if is_source:
            main = node.main_pos + node.main_size
            ports = node.outputs
        else:
            main = node.main_pos - self.arrow_margin
            ports = node.inputs

        # 2. Cross Axis (distributed along the side of the node)
        if port_id and ports:
            try:
                idx = next(i for i, p in enumerate(ports) if p.id == port_id)
                cross = node.cross_pos + (node.cross_size * (idx + 1) / (len(ports) + 1))
            except StopIteration:
                cross = node.cross_pos + (node.cross_size / 2) # Fallback to center
        else:
            cross = node.cross_pos + (node.cross_size / 2) # Fallback to center

        return main, cross

    def _route_edges(self):
        """Step 5: Calculates the Bezier curve paths connection the nodes."""
        for edge in self.graph.edges:
            for seg in edge.segments:
                source = self.graph.nodes[seg.source_id]
                target = self.graph.nodes[seg.target_id]

                s_main, s_cross = self._get_port_coords(source, seg.source_port, is_source=True)
                t_main, t_cross = self._get_port_coords(target, seg.target_port, is_source=False)

                # Bezier control point distance
                ctrl_dist = abs(t_main - s_main) * 0.5

                # SVG Paths require explicit X,Y coordinates
                # Translate main/cross to X,Y based on layout direction
                if self.direction == Direction.LR:
                    seg.path_d = (
                        f"M {s_main},{s_cross} "
                        f"C {s_main + ctrl_dist},{s_cross}, "
                        f"{t_main - ctrl_dist},{t_cross}, "
                        f"{t_main},{t_cross}"
                    )
                else:
                    seg.path_d = (
                        f"M {s_cross},{s_main} "
                        f"C {s_cross},{s_main + ctrl_dist}, "
                        f"{t_cross},{t_main - ctrl_dist}, "
                        f"{t_cross},{t_main}"
                    )
