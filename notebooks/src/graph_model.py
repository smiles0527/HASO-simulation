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
    """
    Types of edges (connections) between nodes with physical properties.
    
    Properties encoded:
    - Traversability: Can agents pass through?
    - Vision: Does it block or provide line of sight?
    - Speed: Does it slow movement?
    - Openable: Can it be opened (doors)?
    - Breakable: Can it be broken through?
    """
    OPEN_AREA = auto()          # traversable, gives vision
    HALLWAY = auto()            # traversable, gives vision
    DESKS = auto()              # traversable, gives vision, slows 40%
    TALL_SHELVES = auto()       # non-traversable, blocks vision
    OPEN_DOOR = auto()          # traversable, blocks vision
    CLOSED_DOOR = auto()        # non-traversable initially, blocks vision, can be opened (2s)
    LOCKED_DOOR = auto()        # non-traversable, blocks vision, can be broken (10s)
    WALL = auto()               # non-traversable/no link
    RADIO_CAMERA = auto()       # non-traversable, provides vision remotely
    STAIRS = auto()             # traversable, slower


# Edge type property definitions
EDGE_TYPE_PROPERTIES = {
    EdgeType.OPEN_AREA: {
        'base_traversable': True,
        'blocks_vision': False,
        'speed_modifier': 1.0,
        'can_open': False,
        'can_break': False,
        'open_time': 0.0,
        'break_time': 0.0,
        'provides_remote_vision': False,
    },
    EdgeType.HALLWAY: {
        'base_traversable': True,
        'blocks_vision': False,
        'speed_modifier': 1.0,
        'can_open': False,
        'can_break': False,
        'open_time': 0.0,
        'break_time': 0.0,
        'provides_remote_vision': False,
    },
    EdgeType.DESKS: {
        'base_traversable': True,
        'blocks_vision': False,
        'speed_modifier': 0.6,  # 40% slower navigating around desks
        'can_open': False,
        'can_break': False,
        'open_time': 0.0,
        'break_time': 0.0,
        'provides_remote_vision': False,
    },
    EdgeType.TALL_SHELVES: {
        'base_traversable': False,
        'blocks_vision': True,
        'speed_modifier': 0.0,
        'can_open': False,
        'can_break': False,
        'open_time': 0.0,
        'break_time': 0.0,
        'provides_remote_vision': False,
    },
    EdgeType.OPEN_DOOR: {
        'base_traversable': True,
        'blocks_vision': True,  # Can't see through doorway even when open
        'speed_modifier': 0.9,  # Slight slowdown passing through
        'can_open': False,  # Already open
        'can_break': False,
        'open_time': 0.0,
        'break_time': 0.0,
        'provides_remote_vision': False,
    },
    EdgeType.CLOSED_DOOR: {
        'base_traversable': False,  # Until opened
        'blocks_vision': True,
        'speed_modifier': 0.9,  # Once opened
        'can_open': True,
        'can_break': True,
        'open_time': 2.0,  # 2 seconds to open
        'break_time': 5.0,  # 5 seconds to break if needed
        'provides_remote_vision': False,
    },
    EdgeType.LOCKED_DOOR: {
        'base_traversable': False,
        'blocks_vision': True,
        'speed_modifier': 0.9,  # Once broken
        'can_open': False,  # Cannot be opened normally
        'can_break': True,
        'open_time': 0.0,
        'break_time': 10.0,  # 10 seconds to break through
        'provides_remote_vision': False,
    },
    EdgeType.WALL: {
        'base_traversable': False,
        'blocks_vision': True,
        'speed_modifier': 0.0,
        'can_open': False,
        'can_break': False,
        'open_time': 0.0,
        'break_time': 0.0,
        'provides_remote_vision': False,
    },
    EdgeType.RADIO_CAMERA: {
        'base_traversable': False,  # Can't walk through camera/radio
        'blocks_vision': False,  # Does not block vision
        'speed_modifier': 0.0,
        'can_open': False,
        'can_break': False,
        'open_time': 0.0,
        'break_time': 0.0,
        'provides_remote_vision': True,  # Provides vision to connected nodes
    },
    EdgeType.STAIRS: {
        'base_traversable': True,
        'blocks_vision': False,
        'speed_modifier': 0.7,  # 30% slower on stairs
        'can_open': False,
        'can_break': False,
        'open_time': 0.0,
        'break_time': 0.0,
        'provides_remote_vision': False,
    },
}


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
    """
    Represents an edge (connection) between two nodes with physical obstacle properties.
    
    Supports dynamic state changes (opening doors, breaking through obstacles).
    """
    src: int
    dst: int
    edge_type: EdgeType = EdgeType.HALLWAY
    
    # Physical properties
    length: float = 5.0         # meters
    width: float = 1.5          # meters (for flow calculations)
    
    # Traversal properties (dynamically updated from edge_type)
    traversable: bool = True
    gives_vision: bool = True
    
    # Dynamic state
    is_open: bool = True        # For doors: currently open?
    is_broken: bool = False     # Has it been broken through?
    opened_by: Optional[int] = None  # Agent ID who opened/broke it
    opened_at: float = 0.0      # Timestamp when opened/broken
    
    def __post_init__(self):
        """Initialize properties from edge_type."""
        props = self.get_properties()
        # Set initial state from edge type
        if self.edge_type in [EdgeType.CLOSED_DOOR, EdgeType.LOCKED_DOOR]:
            self.is_open = False
            self.traversable = False
        else:
            self.traversable = props['base_traversable']
        self.gives_vision = not props['blocks_vision']
    
    def get_properties(self) -> Dict[str, any]:
        """Get properties of this edge's type."""
        return EDGE_TYPE_PROPERTIES.get(self.edge_type, EDGE_TYPE_PROPERTIES[EdgeType.HALLWAY])
    
    def can_traverse(self) -> bool:
        """Check if edge can currently be traversed."""
        props = self.get_properties()
        
        # Check base traversability
        if not props['base_traversable']:
            # For doors, check if opened or broken
            if self.edge_type in [EdgeType.CLOSED_DOOR, EdgeType.LOCKED_DOOR]:
                return self.is_open or self.is_broken
            return False
        
        return True
    
    def can_open_edge(self) -> bool:
        """Check if this edge can be opened."""
        props = self.get_properties()
        return props['can_open'] and not self.is_open and not self.is_broken
    
    def can_break_edge(self) -> bool:
        """Check if this edge can be broken through."""
        props = self.get_properties()
        return props['can_break'] and not self.is_broken
    
    def get_open_time(self) -> float:
        """Time required to open (e.g., a door)."""
        props = self.get_properties()
        return props['open_time']
    
    def get_break_time(self) -> float:
        """Time required to break through."""
        props = self.get_properties()
        return props['break_time']
    
    def open_edge(self, agent_id: int, current_time: float) -> bool:
        """
        Attempt to open this edge (e.g., open a door).
        
        Returns:
            True if successfully opened, False otherwise
        """
        if self.can_open_edge():
            self.is_open = True
            self.traversable = True
            self.opened_by = agent_id
            self.opened_at = current_time
            return True
        return False
    
    def break_edge(self, agent_id: int, current_time: float) -> bool:
        """
        Attempt to break through this edge.
        
        Returns:
            True if successfully broken, False otherwise
        """
        if self.can_break_edge():
            self.is_broken = True
            self.traversable = True
            self.is_open = True  # Effectively open once broken
            self.opened_by = agent_id
            self.opened_at = current_time
            return True
        return False
    
    def get_traversal_time(self, base_speed: float = 1.5, hazard_modifier: float = 1.0) -> float:
        """
        Calculate time to traverse this edge.
        
        Args:
            base_speed: base walking speed in m/s (default 1.5 m/s)
            hazard_modifier: speed multiplier from hazards (< 1.0 = slower)
        
        Returns:
            Time in seconds to traverse this edge
        """
        if not self.can_traverse():
            return float('inf')
        
        # Get obstacle speed modifier
        props = self.get_properties()
        obstacle_modifier = props['speed_modifier']
        
        # Calculate effective speed
        speed = base_speed * hazard_modifier * obstacle_modifier
        
        # Prevent division by zero
        if speed <= 0:
            return float('inf')
        
        return self.length / speed
    
    @property
    def can_open(self) -> bool:
        """Check if edge can be opened (for closed/locked doors)."""
        return self.can_open_edge()


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
            # traversable and gives_vision are set from edge_type in __post_init__
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

