"""
graph_model.py: Graph-based building representation for evacuation simulation.

Represents buildings as graphs where:
- Nodes (vertices): rooms, hallways, stairwells, exits
- Edges: doors, corridors, connections (weighted by distance/time)
- Dynamic properties: hazards, occupants, priorities
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto
import yaml
import math


class NodeType(Enum):
    """Types of nodes in the building graph."""
    CORRIDOR = auto()
    STAIRCASE = auto()
    SMALL_ROOM = auto()
    CONNECTED_SMALL_ROOM = auto()
    CONNECTED_LARGE_ROOM = auto()  # hub
    LARGE_CENTRAL_ROOM = auto()
    LARGE_SIDE_ROOM = auto()
    EXIT = auto()
    CHECKPOINT = auto()


class EdgeType(Enum):
    """Types of edges (connections) between nodes."""
    OPEN_AREA = auto()          # traversable, gives vision
    HALLWAY = auto()            # traversable, gives vision
    DESKS = auto()              # traversable, gives vision, slows
    TALL_SHELVES = auto()       # non-traversable, blocks vision
    OPEN_DOOR = auto()          # traversable, blocks vision
    CLOSED_DOOR = auto()        # non-traversable, blocks vision, can be opened
    LOCKED_DOOR = auto()        # non-traversable, blocks vision, needs breaking
    WALL = auto()               # non-traversable/no link
    RADIO_CAMERA = auto()       # non-traversable, gives vision
    STAIRS = auto()             # traversable, slower


class HazardType(Enum):
    """Types of hazards that can affect nodes."""
    NONE = auto()
    SMOKE = auto()
    FIRE = auto()
    HEAT = auto()
    BIOHAZARD = auto()
    EXPLOSIVE = auto()
    CHEMICAL = auto()
    RADIOACTIVE = auto()


@dataclass
class Evacuee:
    """Represents an evacuee in the building."""
    id: int
    age_group: str  # infant, children, adolescent, young_adult, adult, senior, elderly
    health: str     # fit, healthy, unfit, illness, requiring_assistance, mobility_disability, incapacitated
    node: int       # current node location
    evacuating: bool = False
    tagged: bool = False
    outside: bool = False
    
    @property
    def speed_multiplier(self) -> float:
        """Calculate speed multiplier based on age and health."""
        age_speeds = {
            "infant": 0.0,
            "children": 0.6,
            "adolescent": 1.1,
            "young_adult": 1.0,
            "adult": 0.9,
            "senior": 0.7,
            "elderly": 0.4,
        }
        health_speeds = {
            "fit": 1.4,
            "healthy": 1.0,
            "unfit": 0.8,
            "illness": 0.7,
            "requiring_assistance": 0.0,
            "mobility_disability": 0.3,
            "incapacitated": 0.0,
        }
        return age_speeds.get(self.age_group, 1.0) * health_speeds.get(self.health, 1.0)
    
    @property
    def needs_assistance(self) -> bool:
        """Check if evacuee needs responder assistance."""
        return self.speed_multiplier == 0.0 or self.health in ["requiring_assistance", "incapacitated"]


@dataclass
class Node:
    """Represents a node (room/area) in the building graph."""
    id: int
    node_type: NodeType
    floor: int = 0
    name: str = ""
    
    # Geometric properties
    x: float = 0.0
    y: float = 0.0
    area: float = 10.0  # square meters
    
    # Priority and sector
    room_priority: int = 3      # 1 (highest) to 5 (lowest)
    sector_id: int = 0
    sector_priority: int = 3
    
    # Hazards and occupants
    hazard: HazardType = HazardType.NONE
    hazard_severity: float = 0.0  # 0.0 to 1.0
    evacuees: List[Evacuee] = field(default_factory=list)
    
    # HASO: Dynamic properties
    visibility: float = 1.0      # 0.0 (no visibility) to 1.0 (full visibility)
    occupancy_probability: float = 0.0  # Probability of occupants present
    fire_spread_rate: float = 0.0  # Fire propagation coefficient k
    smoke_decay_rate: float = 0.05  # Visibility decay γ
    
    # Search/clearance properties
    search_time: float = 5.0     # base time to search this node (seconds)
    cleared: bool = False
    cleared_by: Optional[int] = None  # agent ID
    verified_by: Optional[int] = None  # HASO: dual verification
    clearance_timestamp: float = 0.0  # When was it cleared
    
    # HASO: Zone assignment
    zone_id: int = -1  # Which zone this node belongs to
    
    # Fog of war: 0=unknown-unknown, 1=known-unknown, 2=unknown-known, 3=known-known
    fog_state: int = 0
    
    def get_effective_search_time(self, agent_speed_modifier: float = 1.0) -> float:
        """Calculate effective search time based on area, hazards, and agent speed."""
        base = self.search_time * (self.area / 10.0)  # scale by area
        hazard_penalty = 1.0 + self.hazard_severity * 2.0  # up to 3x slower
        
        # Prevent division by zero
        if agent_speed_modifier <= 0:
            agent_speed_modifier = 0.1  # fallback to very slow
        
        return base * hazard_penalty / agent_speed_modifier


@dataclass
class Edge:
    """Represents an edge (connection) between two nodes."""
    src: int
    dst: int
    edge_type: EdgeType = EdgeType.HALLWAY
    
    # Physical properties
    length: float = 5.0         # meters
    width: float = 1.5          # meters (for flow calculations)
    
    # Traversal properties
    traversable: bool = True
    gives_vision: bool = True
    
    # Time costs
    open_time: float = 0.0      # time to open if closed (seconds)
    break_time: float = 0.0     # time to break through if locked (seconds)
    
    # State
    is_open: bool = True
    
    def get_traversal_time(self, base_speed: float = 1.5, hazard_modifier: float = 1.0) -> float:
        """
        Calculate time to traverse this edge.
        
        Args:
            base_speed: base walking speed in m/s (default 1.5 m/s)
            hazard_modifier: speed multiplier from hazards (< 1.0 = slower)
        
        Returns:
            Time in seconds to traverse this edge
        """
        if not self.traversable or not self.is_open:
            return float('inf')
        
        # Base traversal time
        speed = base_speed * hazard_modifier
        
        # Prevent division by zero
        if speed <= 0:
            return float('inf')
        
        # Edge type modifiers
        if self.edge_type == EdgeType.DESKS:
            speed *= 0.7
        elif self.edge_type == EdgeType.STAIRS:
            speed *= 0.5
        elif self.edge_type == EdgeType.OPEN_DOOR:
            # Add small overhead for door passage
            return self.length / speed + 0.5
        
        return self.length / speed
    
    @property
    def can_open(self) -> bool:
        """Check if edge can be opened (for closed/locked doors)."""
        return self.edge_type in [EdgeType.CLOSED_DOOR, EdgeType.LOCKED_DOOR]


class Graph:
    """
    Building graph representation using adjacency list.
    
    Stores nodes and edges with efficient neighbor lookup.
    Supports dynamic updates for hazards, fog-of-war, and clearance.
    """
    
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[Tuple[int, int], Edge] = {}
        self._adjacency: Dict[int, List[int]] = {}
        
        # Building metadata
        self.building_type: str = "generic"
        self.num_floors: int = 1
        self.exits: List[int] = []
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = []
        if node.node_type == NodeType.EXIT:
            if node.id not in self.exits:
                self.exits.append(node.id)
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph (bidirectional by default)."""
        self.edges[(edge.src, edge.dst)] = edge
        
        # Add reverse edge with same properties
        reverse = Edge(
            src=edge.dst,
            dst=edge.src,
            edge_type=edge.edge_type,
            length=edge.length,
            width=edge.width,
            traversable=edge.traversable,
            gives_vision=edge.gives_vision,
            open_time=edge.open_time,
            break_time=edge.break_time,
            is_open=edge.is_open,
        )
        self.edges[(edge.dst, edge.src)] = reverse
        
        # Update adjacency
        if edge.src not in self._adjacency:
            self._adjacency[edge.src] = []
        if edge.dst not in self._adjacency[edge.src]:
            self._adjacency[edge.src].append(edge.dst)
        
        if edge.dst not in self._adjacency:
            self._adjacency[edge.dst] = []
        if edge.src not in self._adjacency[edge.dst]:
            self._adjacency[edge.dst].append(edge.src)
    
    def neighbors(self, node_id: int) -> List[int]:
        """Get list of neighbor node IDs."""
        return self._adjacency.get(node_id, [])
    
    def get_edge(self, src: int, dst: int) -> Optional[Edge]:
        """Get edge between two nodes."""
        return self.edges.get((src, dst))
    
    def get_node(self, node_id: int) -> Optional[Node]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def distance(self, n1: int, n2: int) -> float:
        """Euclidean distance between two nodes."""
        node1 = self.nodes.get(n1)
        node2 = self.nodes.get(n2)
        if not node1 or not node2:
            return float('inf')
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        return math.sqrt(dx * dx + dy * dy)
    
    def get_cleared_count(self) -> Tuple[int, int]:
        """Returns (cleared_count, total_rooms)."""
        total = sum(1 for n in self.nodes.values() 
                   if n.node_type not in [NodeType.CORRIDOR, NodeType.EXIT, NodeType.CHECKPOINT])
        cleared = sum(1 for n in self.nodes.values() 
                     if n.cleared and n.node_type not in [NodeType.CORRIDOR, NodeType.EXIT, NodeType.CHECKPOINT])
        return cleared, total


def load_map_yaml(path: str) -> Graph:
    """
    Load a building map from a YAML file.
    
    Expected YAML structure:
    ```yaml
    building:
      type: "office"
      num_floors: 3
    
    nodes:
      - id: 0
        type: "EXIT"
        floor: 0
        x: 0
        y: 0
        name: "Main Entrance"
      - id: 1
        type: "CORRIDOR"
        floor: 0
        x: 5
        y: 0
        # ... more properties
    
    edges:
      - src: 0
        dst: 1
        type: "OPEN_DOOR"
        length: 3.0
        width: 1.2
    
    evacuees:  # optional
      - id: 0
        age_group: "adult"
        health: "healthy"
        node: 5
    ```
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    graph = Graph()
    
    # Load building metadata
    if 'building' in data:
        graph.building_type = data['building'].get('type', 'generic')
        graph.num_floors = data['building'].get('num_floors', 1)
    
    # Load nodes
    for node_data in data.get('nodes', []):
        node = Node(
            id=node_data['id'],
            node_type=NodeType[node_data.get('type', 'SMALL_ROOM').upper()],
            floor=node_data.get('floor', 0),
            name=node_data.get('name', f"Node_{node_data['id']}"),
            x=node_data.get('x', 0.0),
            y=node_data.get('y', 0.0),
            area=node_data.get('area', 10.0),
            room_priority=node_data.get('room_priority', 3),
            sector_id=node_data.get('sector_id', 0),
            sector_priority=node_data.get('sector_priority', 3),
            hazard=HazardType[node_data.get('hazard', 'NONE').upper()],
            hazard_severity=node_data.get('hazard_severity', 0.0),
            search_time=node_data.get('search_time', 5.0),
        )
        graph.add_node(node)
    
    # Load edges
    for edge_data in data.get('edges', []):
        edge = Edge(
            src=edge_data['src'],
            dst=edge_data['dst'],
            edge_type=EdgeType[edge_data.get('type', 'HALLWAY').upper()],
            length=edge_data.get('length', 5.0),
            width=edge_data.get('width', 1.5),
            traversable=edge_data.get('traversable', True),
            gives_vision=edge_data.get('gives_vision', True),
            open_time=edge_data.get('open_time', 0.0),
            break_time=edge_data.get('break_time', 0.0),
            is_open=edge_data.get('is_open', True),
        )
        graph.add_edge(edge)
    
    # Load evacuees
    for evac_data in data.get('evacuees', []):
        evacuee = Evacuee(
            id=evac_data['id'],
            age_group=evac_data.get('age_group', 'adult'),
            health=evac_data.get('health', 'healthy'),
            node=evac_data['node'],
        )
        node = graph.get_node(evacuee.node)
        if node:
            node.evacuees.append(evacuee)
    
    return graph

