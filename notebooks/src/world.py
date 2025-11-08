"""
world.py: Discrete-event simulation engine for evacuation sweeps.

Manages the simulation state, event scheduling, and fog-of-war system.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import heapq
import math

from .graph_model import Graph, Node, NodeType, HazardType
from .agents import Agent, Role, Status


@dataclass
class Event:
    """Discrete event for simulation scheduler."""
    time: float
    event_id: int
    fn: Callable
    args: tuple
    
    def __lt__(self, other: Event) -> bool:
        """Priority queue ordering by time, then event_id."""
        if self.time != other.time:
            return self.time < other.time
        return self.event_id < other.event_id


class FogOfWar:
    """
    Fog of war system tracking knowledge state of each node.
    
    States:
    0 = Unknown-unknown: don't know exists, must explore via adjacent
    1 = Known-unknown: know exists, don't know hazards/people
    2 = Unknown-known: don't know exists, but know hazards/people (via radio/camera)
    3 = Known-known: have vision, know everything
    """
    
    def __init__(self, graph: Graph):
        self.graph = graph
        # fog_state[node_id] = 0, 1, 2, or 3
        self.fog_state: Dict[int, int] = {nid: 0 for nid in graph.nodes}
    
    def init_known_nodes(self, node_ids: List[int]) -> None:
        """Initialize certain nodes as known (e.g., starting position, corridors)."""
        for nid in node_ids:
            if nid in self.fog_state:
                self.fog_state[nid] = 1  # known-unknown
    
    def reveal_node(self, node_id: int, full: bool = True) -> None:
        """
        Reveal a node's existence and optionally full information.
        
        Args:
            node_id: node to reveal
            full: if True, reveal full info (state 3), else just existence (state 1)
        """
        if node_id in self.fog_state:
            if full:
                self.fog_state[node_id] = 3  # known-known
                node = self.graph.get_node(node_id)
                if node:
                    node.fog_state = 3
            else:
                if self.fog_state[node_id] == 0:
                    self.fog_state[node_id] = 1  # known-unknown
                    node = self.graph.get_node(node_id)
                    if node:
                        node.fog_state = 1
    
    def get_visible_neighbors(self, node_id: int, vision_range: int = 1) -> List[int]:
        """
        Get neighbors visible from a node within vision range.
        
        Args:
            node_id: current node
            vision_range: how many hops to see
        
        Returns:
            List of node IDs that can be seen
        """
        visible = set()
        frontier = {node_id}
        
        for _ in range(vision_range):
            next_frontier = set()
            for nid in frontier:
                neighbors = self.graph.neighbors(nid)
                for neighbor in neighbors:
                    edge = self.graph.get_edge(nid, neighbor)
                    # Check if edge gives vision
                    if edge and edge.gives_vision and edge.is_open:
                        visible.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        
        return list(visible)
    
    def update_from_agent(self, agent: Agent) -> None:
        """Update fog of war based on agent's current position and vision."""
        if not agent.has_vision:
            return
        
        # Agent's current node is fully known
        self.reveal_node(agent.node, full=True)
        
        # Reveal visible neighbors
        vision_range = agent.get_effective_vision()
        visible = self.get_visible_neighbors(agent.node, vision_range)
        
        for nid in visible:
            self.reveal_node(nid, full=True)
        
        # Reveal existence of neighbors of visible nodes (known-unknown)
        for nid in visible:
            for neighbor in self.graph.neighbors(nid):
                if self.fog_state.get(neighbor, 0) == 0:
                    self.reveal_node(neighbor, full=False)
    
    def is_known(self, node_id: int) -> bool:
        """Check if node existence is known (state >= 1)."""
        return self.fog_state.get(node_id, 0) >= 1
    
    def is_fully_known(self, node_id: int) -> bool:
        """Check if node is fully known (state == 3)."""
        return self.fog_state.get(node_id, 0) == 3
    
    def get_known_nodes(self) -> List[int]:
        """Get all nodes that are at least partially known."""
        return [nid for nid, state in self.fog_state.items() if state >= 1]
    
    def get_fully_known_nodes(self) -> List[int]:
        """Get all fully known nodes."""
        return [nid for nid, state in self.fog_state.items() if state == 3]


class World:
    """
    Main simulation world managing graph, agents, and discrete event scheduling.
    
    Uses a priority queue for event scheduling with time-ordered execution.
    Tracks simulation state, fog-of-war, and clearance progress.
    """
    
    def __init__(self, G: Graph, agents: List[Agent]):
        self.G = G
        self.agents = agents
        
        # Event scheduler
        self._event_queue: List[Event] = []
        self._event_counter = 0
        self.time = 0.0
        
        # Fog of war
        self.fog = FogOfWar(G)
        
        # Clearance tracking
        self.cleared: Dict[int, bool] = {nid: False for nid in G.nodes}
        
        # Weights for scoring/optimization
        self.weights = {
            "room_priority": 1.0,
            "sector_priority": 1.0,
            "distance": 0.5,
            "hazard_penalty": 2.0,
            "unchecked_penalty": 3.0,
        }
        
        # History tracking (for visualization)
        self.history: List[Dict[str, Any]] = []
        self.record_interval = 5.0  # record state every 5 seconds
        self._next_record_time = 0.0
        
        # Policies (attached externally)
        self.policies: Dict[Role, Callable] = {}
    
    def init_fog(self, known_nodes: List[int]) -> None:
        """Initialize fog of war with known starting nodes."""
        self.fog.init_known_nodes(known_nodes)
        
        # Update fog from all agents' initial positions
        for agent in self.agents:
            self.fog.update_from_agent(agent)
    
    def schedule(self, dt: float, fn: Callable, *args) -> None:
        """
        Schedule an event to occur at time + dt.
        
        Args:
            dt: time delay from now
            fn: function to call
            *args: arguments to pass to function
        """
        event = Event(
            time=self.time + dt,
            event_id=self._event_counter,
            fn=fn,
            args=args,
        )
        self._event_counter += 1
        heapq.heappush(self._event_queue, event)
    
    def run(self, tmax: float = 1200.0) -> None:
        """
        Run simulation until tmax or all rooms cleared.
        
        Args:
            tmax: maximum simulation time in seconds
        """
        while self._event_queue and self.time < tmax:
            # Get next event
            event = heapq.heappop(self._event_queue)
            self.time = event.time
            
            # Execute event
            try:
                event.fn(*event.args)
            except Exception as e:
                print(f"[World] Error in event {event.event_id} at t={self.time:.2f}: {e}")
            
            # Record history periodically
            if self.time >= self._next_record_time:
                self._record_state()
                self._next_record_time = self.time + self.record_interval
            
            # Check termination condition
            if self.all_rooms_cleared():
                print(f"[World] All rooms cleared at t={self.time:.2f}s")
                break
        
        # Final state record
        self._record_state()
        
        print(f"[World] Simulation ended at t={self.time:.2f}s")
        cleared, total = self.G.get_cleared_count()
        print(f"[World] Cleared {cleared}/{total} rooms")
    
    def _record_state(self) -> None:
        """Record current state for history/visualization."""
        state = {
            "time": self.time,
            "agents": [
                {
                    "id": a.id,
                    "role": a.role.name,
                    "node": a.node,
                    "status": a.status.name,
                    "action": a.action.name,
                    "rooms_cleared": a.rooms_cleared,
                }
                for a in self.agents
            ],
            "cleared_nodes": [nid for nid, cleared in self.cleared.items() if cleared],
            "fog_known": len(self.fog.get_known_nodes()),
            "fog_fully_known": len(self.fog.get_fully_known_nodes()),
        }
        self.history.append(state)
    
    def all_rooms_cleared(self) -> bool:
        """Check if all rooms (non-corridor nodes) are cleared."""
        for node in self.G.nodes.values():
            if node.node_type not in [NodeType.CORRIDOR, NodeType.EXIT, NodeType.CHECKPOINT]:
                if not node.cleared:
                    return False
        return True
    
    def get_agent(self, agent_id: int) -> Optional[Agent]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def mark_cleared(self, node_id: int, agent_id: int) -> None:
        """Mark a node as cleared by an agent."""
        node = self.G.get_node(node_id)
        if node:
            node.cleared = True
            node.cleared_by = agent_id
            self.cleared[node_id] = True
    
    def get_hazard_modifier(self, node_id: int) -> float:
        """
        Get speed modifier for hazards at a node.
        
        Returns:
            Multiplier for speed (< 1.0 = slower, 1.0 = no effect)
        """
        node = self.G.get_node(node_id)
        if not node or node.hazard == HazardType.NONE:
            return 1.0
        
        # Hazard severity affects speed
        severity = node.hazard_severity
        
        if node.hazard == HazardType.SMOKE:
            return 1.0 - 0.4 * severity  # up to 40% slower
        elif node.hazard == HazardType.FIRE:
            return 1.0 - 0.7 * severity  # up to 70% slower
        elif node.hazard == HazardType.HEAT:
            return 1.0 - 0.3 * severity
        else:
            return 1.0 - 0.5 * severity
    
    def shortest_path_known(self, start: int, goal: int) -> Optional[List[int]]:
        """
        A* shortest path considering only known and traversable nodes/edges.
        
        Args:
            start: starting node ID
            goal: goal node ID
        
        Returns:
            Path as list of node IDs, or None if no path exists
        """
        if start == goal:
            return [start]
        
        # Check if goal is known
        if not self.fog.is_known(goal):
            return None
        
        # A* search
        from heapq import heappush, heappop
        
        open_set = []
        heappush(open_set, (0.0, start))
        
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.G.distance(start, goal)}
        
        while open_set:
            _, current = heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            
            for neighbor in self.G.neighbors(current):
                # Only consider known nodes
                if not self.fog.is_known(neighbor):
                    continue
                
                edge = self.G.get_edge(current, neighbor)
                if not edge or not edge.traversable or not edge.is_open:
                    continue
                
                # Calculate tentative g_score
                hazard_mod = self.get_hazard_modifier(neighbor)
                edge_cost = edge.get_traversal_time(base_speed=1.5, hazard_modifier=hazard_mod)
                
                tentative_g = g_score[current] + edge_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.G.distance(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def find_nearest_uncleared_room(self, from_node: int, max_candidates: int = 10) -> Optional[int]:
        """
        Find nearest uncleared room from a given node.
        
        Args:
            from_node: starting node ID
            max_candidates: maximum number of candidates to consider
        
        Returns:
            Node ID of nearest uncleared room, or None
        """
        uncleared = [
            nid for nid, node in self.G.nodes.items()
            if not node.cleared
            and node.node_type not in [NodeType.CORRIDOR, NodeType.EXIT, NodeType.CHECKPOINT]
            and self.fog.is_known(nid)
        ]
        
        if not uncleared:
            return None
        
        # Sort by distance (Euclidean heuristic)
        candidates = sorted(uncleared, key=lambda n: self.G.distance(from_node, n))
        
        # Try to find path to nearest candidates
        for candidate in candidates[:max_candidates]:
            path = self.shortest_path_known(from_node, candidate)
            if path:
                return candidate
        
        return None
    
    def get_priority_score(self, node_id: int, agent: Agent) -> float:
        """
        Calculate priority score for a node from an agent's perspective.
        
        Higher score = higher priority
        
        Considers:
        - Room priority (1 = highest, 5 = lowest)
        - Sector priority
        - Distance
        - Hazards
        - Whether it's been checked
        """
        node = self.G.get_node(node_id)
        if not node:
            return 0.0
        
        score = 0.0
        
        # Room priority (invert so 1 = high score)
        room_priority_score = (6 - node.room_priority) * self.weights["room_priority"]
        score += room_priority_score
        
        # Sector priority
        sector_priority_score = (6 - node.sector_priority) * self.weights["sector_priority"]
        score += sector_priority_score
        
        # Distance penalty (closer = higher score)
        dist = self.G.distance(agent.node, node_id)
        distance_penalty = dist * self.weights["distance"]
        score -= distance_penalty
        
        # Hazard penalty
        if node.hazard != HazardType.NONE:
            hazard_penalty = node.hazard_severity * self.weights["hazard_penalty"]
            score -= hazard_penalty
        
        # Unchecked rooms get bonus
        if not node.cleared:
            score += self.weights["unchecked_penalty"]
        
        return score

