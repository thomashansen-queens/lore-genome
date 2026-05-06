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

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class GraphNode:
    id: str
    label: str
    payload: Any = None  # Arbitrary data associated with a node
    layer: int = 0
    x: float = 0.0
    y: float = 0.0
    width: float = 200.0
    height: float = 60.0
    is_virtual: bool = False

@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    label: str
    path_d: str = ""  # SVG path data for edge routing


class Graph:
    """A generic container for graph data."""
    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def add_node(self, node_id: str, label: str, **kwargs):
        self.nodes[node_id] = GraphNode(id=node_id, label=label, **kwargs)

    def add_edge(self, source_id: str, target_id: str, label: str, **kwargs):
        self.edges.append(GraphEdge(source_id=source_id, target_id=target_id, label=label, **kwargs))


@dataclass
class DiagramResult:
    """Final output of the graph layout for rendering."""
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    canvas_width: float
    canvas_height: float


class DagLayout:
    """Layered DAG diagram layout (Sugiyama method)."""
    def __init__(
        self,
        graph: Graph,
        direction: Literal["TB", "LR"] = "LR",
        arrow_margin: float = 5.0,
        gap: float = 50.0,
        **overrides,
    ):
        self.graph = graph
        self.direction = direction.upper()  # top-bottom or left-right
        self.arrow_margin = arrow_margin
        self.gap = gap
        self._overrides = overrides  # useful for testing
        self.canvas_width = 0.0
        self.canvas_height = 0.0

    @property
    def padding(self) -> float:
        return self._overrides.get("padding", self.gap)

    @property
    def column_width(self) -> float:
        if "column_width" in self._overrides:
            return self._overrides["column_width"]
        max_w = max((n.width for n in self.graph.nodes.values()), default=200.0)
        return max_w + self.gap

    @property
    def row_height(self) -> float:
        if "row_height" in self._overrides:
            return self._overrides["row_height"]
        max_h = max((n.height for n in self.graph.nodes.values()), default=60.0)
        return max_h + self.gap

    def compute(self) -> DiagramResult:
        """
        Step 1: Input is DAG, so already acyclic.
        Runs the layout algorithm and returns nodes, edges, and canvas dimensions.
        """
        self._assign_layers()
        self._build_layer_map()     # keeps ordered layers in scope for edge routing
        self._order_vertices()      # group nodes by layer to be near parents/children
        self._assign_coordinates()  # assigns (x,y) to nodes based on layer and order
        self._route_edges()         # calculates bezier control points for edge paths
        return DiagramResult(
            nodes=list(self.graph.nodes.values()),
            edges=self.graph.edges,
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
        )

    def _assign_layers(self):
        """Step 2: Assign nodes to columns (longest path from roots).
        TODO: Topological sort with longest path layering could be more efficient..."""
        changed = True
        while changed:
            changed = False
            for edge in self.graph.edges:
                if edge.source_id not in self.graph.nodes or edge.target_id not in self.graph.nodes:
                    continue  # Ignore edges with missing nodes

                source = self.graph.nodes[edge.source_id]
                target = self.graph.nodes[edge.target_id]

                if target.layer <= source.layer:
                    target.layer = source.layer + 1
                    changed = True

    def _build_layer_map(self):
        """Helper to group nodes by layer."""
        self._layer_map = {}
        for node in self.graph.nodes.values():
            self._layer_map.setdefault(node.layer, []).append(node)

    def _order_vertices(self):
        """Step 3: Order vertices within layers to minimize edge crossings (barycenter heuristic)."""
        if not self._layer_map:
            return  # No nodes to order

        max_layer = max(self._layer_map.keys())
        # Build adjacency by direction
        parents = {}   # node_id -> [source_ids]
        children = {}  # node_id -> [target_ids]
        for e in self.graph.edges:
            if e.source_id in self.graph.nodes and e.target_id in self.graph.nodes:
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

                def bary_up(n, _pos=prev_pos):
                    hits = [_pos[pid] for pid in parents.get(n.id, []) if pid in _pos]
                    return sum(hits) / len(hits) if hits else float('inf')

                self._layer_map[l].sort(key=bary_up)

            # Backward pass: Sort layer by average position of children in next layer
            for l in range(max_layer - 1, -1, -1):
                nodes = self._layer_map.get(l, [])
                if len(nodes) <= 1:
                    continue
                next_pos = {n.id: i for i, n in enumerate(self._layer_map.get(l + 1, []))}

                def bary_down(n, _pos=next_pos):
                    hits = [_pos[cid] for cid in children.get(n.id, []) if cid in _pos]
                    return sum(hits) / len(hits) if hits else float('inf')

                self._layer_map[l].sort(key=bary_down)

    def _assign_coordinates(self):
        """Step 4: Assign (x,y) coordinates to nodes based on layer and order."""
        if not self.graph.nodes:
            return

        # 1. Find the tallest column to act as the vertical anchor
        max_nodes_in_layer = max(len(nodes) for nodes in self._layer_map.values())
        padding = self.padding
        max_x = 0.0
        max_y = 0.0

        # 2. Calculate where this layer should start for centering
        if self.direction == "LR":
            tallest = max_nodes_in_layer * self.row_height
            for layer_idx, nodes in self._layer_map.items():
                col_height = len(nodes) * self.row_height
                start_y = padding + (tallest - col_height) / 2
                for i, node in enumerate(nodes):
                    node.x = padding + layer_idx * self.column_width
                    node.y = start_y + i * self.row_height
                    max_x = max(max_x, node.x + node.width)
                    max_y = max(max_y, node.y + node.height)
        else:  # Top-Bottom
            widest = max_nodes_in_layer * self.column_width
            for layer_idx, nodes in self._layer_map.items():
                row_width = len(nodes) * self.column_width
                start_x = padding + (widest - row_width) / 2
                for i, node in enumerate(nodes):
                    node.x = start_x + i * self.column_width
                    node.y = padding + layer_idx * self.row_height
                    max_x = max(max_x, node.x + node.width)
                    max_y = max(max_y, node.y + node.height)

        self.canvas_width = max_x + padding
        self.canvas_height = max_y + padding

    def _route_edges(self):
        """Calculates the Bezier curve paths connection the nodes."""
        valid_edges = [
            e for e in self.graph.edges
            if e.source_id in self.graph.nodes and e.target_id in self.graph.nodes
        ]
        # TODO: Draw stubs to/from missing nodes

        # Group edges by source and target node
        outgoing = {}
        incoming = {}
        for edge in valid_edges:
            outgoing.setdefault(edge.source_id, []).append(edge)
            incoming.setdefault(edge.target_id, []).append(edge)

        # Sort by perpendicular position to minimize crossings
        is_lr = self.direction == "LR"
        perp = (lambda n: n.y) if is_lr else (lambda n: n.x)

        for edges in outgoing.values():
            edges.sort(key=lambda e: perp(self.graph.nodes[e.target_id]))
        for edges in incoming.values():
            edges.sort(key=lambda e: perp(self.graph.nodes[e.source_id]))

        for edge in valid_edges:
            source = self.graph.nodes[edge.source_id]
            target = self.graph.nodes[edge.target_id]

            out_edges = outgoing[edge.source_id]
            in_edges = incoming[edge.target_id]
            out_slot = out_edges.index(edge)
            in_slot = in_edges.index(edge)

            if is_lr:
                # Distribute along the right side of source, left side of target
                sx = source.x + source.width
                sy = source.y + source.height * (out_slot + 1) / (len(out_edges) + 1)
                ex = target.x - self.arrow_margin  # give space for arrowhead
                ey = target.y + target.height * (in_slot + 1) / (len(in_edges) + 1)
                ctrl_dx = abs(ex - sx) * 0.5
                nudge = self._compute_nudge(source, target, sy, ey, self._layer_map)

                edge.path_d = (
                    f"M {sx},{sy} "
                    f"C {sx + ctrl_dx},{sy + nudge} "
                    f"{ex - ctrl_dx},{ey + nudge} "
                    f"{ex},{ey}"
                )
            else:
                # Distribute along the bottom of the source, top side of target
                sx = source.x + source.width * (out_slot + 1) / (len(out_edges) + 1)
                sy = source.y + source.height
                ex = target.x + target.width * (in_slot + 1) / (len(in_edges) + 1)
                ey = target.y - self.arrow_margin  # give space for arrowhead
                ctrl_dy = abs(ey - sy) * 0.5
                nudge = self._compute_nudge(source, target, sx, ex, self._layer_map)

                edge.path_d = (
                    f"M {sx},{sy} "
                    f"C {sx + nudge},{sy + ctrl_dy} "
                    f"{ex + nudge},{ey - ctrl_dy} "
                    f"{ex},{ey}"
                )

    def _compute_nudge(self, source, target, start_perp, end_perp, layer_map):
        """
        Perpendicular offset for bezier controls to dodge intermediate nodes.

        Walks each layer between source and target. At each layer, interpolates 
        where the straight line would cross, then checks if an node overlaps its
        position. Collects the worst-case clearance needed in both directions, 
        then picks the less disruptive one.
        """
        span = target.layer - source.layer
        if span <= 1:
            return 0.0  # No intermediate layers, no nudge needed

        is_lr = self.direction == "LR"
        padding = 20
        push_neg = 0.0  # maximum up/left shift
        push_pos = 0.0  # maximum down/right shift
        obstructed = False

        for layer_idx in range(source.layer + 1, target.layer):
            t = (layer_idx - source.layer) / span
            line_pos = start_perp + t * (end_perp - start_perp)

            for node in layer_map.get(layer_idx, []):
                lo = node.y if is_lr else node.x
                hi = (node.y + node.height) if is_lr else (node.x + node.width)

                if lo - padding < line_pos < hi + padding:
                    obstructed = True
                    push_neg = min(push_neg, (lo - padding) - line_pos)
                    push_pos = max(push_pos, (hi + padding) - line_pos)

        if not obstructed:
            return 0.0  # no nudge needed

        # Pick the smaller nudge direction
        raw = push_neg if abs(push_neg) < abs(push_pos) else push_pos
        # Cubic Bezier at 0.5 receives ~75% of the nudge, can play with this
        return raw / 0.75
