"""
world.py: Discrete-event simulation engine for evacuation sweeps.

Manages the simulation state, event scheduling, and fog-of-war system.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import heapq
import math

from .graph_model import Graph, Node, NodeType, HazardType
from .agents import Agent, Role, Status

try:
    from .task_allocator import optimize_zone_assignment_ilp
except ImportError:  # pragma: no cover
    optimize_zone_assignment_ilp = None


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
        
        # HASO: Zone assignments
        self.zones: Dict[int, List[int]] = {}  # zone_id -> node_ids
        self.agent_zones: Dict[int, int] = {}  # agent_id -> zone_id
        
        # HASO: Weights for cost function C(a_i)
        self.weights = {
            "room_priority": 1.0,
            "sector_priority": 1.0,
            "distance": 0.5,
            "hazard_penalty": 2.0,  # β in HASO
            "visibility_penalty": 1.5,  # λ in HASO
            "unchecked_penalty": 3.0,
        }
        
        # HASO: Hazard propagation parameters
        self.hazard_update_interval = 10.0  # Update hazards every 10 seconds
        self._next_hazard_update = 10.0
        self.fire_spread_probability = 0.15  # Probability of fire spreading to neighbor
        
        # History tracking (for visualization)
        self.history: List[Dict[str, Any]] = []
        self.record_interval = 5.0  # record state every 5 seconds
        self._next_record_time = 0.0
        
        # Policies (attached externally)
        self.policies: Dict[Role, Callable] = {}
        
        # Visualization helper: track active agent movements between nodes
        self.movement_tracker: Dict[int, Dict[str, Any]] = {}

        # Evacuee state tracking
        self.evac_states: Dict[int, Dict[str, Any]] = {}
        self._total_evacuees: int = 0
        self._initialize_evacuation_states()
    
    def init_fog(self, known_nodes: List[int]) -> None:
        """Initialize fog of war with known starting nodes."""
        self.fog.init_known_nodes(known_nodes)
        
        # Update fog from all agents' initial positions
        for agent in self.agents:
            self.fog.update_from_agent(agent)

    # ------------------------------------------------------------------
    # Evacuee initialization & utilities
    # ------------------------------------------------------------------

    def _initialize_evacuation_states(self) -> None:
        """Pre-compute evacuee metadata and occupancy probabilities."""
        self.evac_states.clear()
        self._total_evacuees = 0

        for node in self.G.nodes.values():
            evacuees = list(getattr(node, "evacuees", []))
            self._total_evacuees += len(evacuees)
            self._update_node_occupancy(node.id)

            for evac in evacuees:
                evac.evacuating = False
                evac.outside = False
                state = {
                    "evac": evac,
                    "path": self._compute_exit_path(evac.node),
                    "index": 0,
                    "moving": False,
                    "assisted": not evac.needs_assistance,
                    "position": (node.x, node.y),
                    "started": False,
                    "waiting_for_path": False,
                    "segment_start_time": None,
                    "segment_end_time": None,
                    "start_pos": (node.x, node.y),
                    "end_pos": (node.x, node.y),
                    "departing_from": node.id,
                }

                if not state["path"] or len(state["path"]) <= 1:
                    state["waiting_for_path"] = True

                self.evac_states[evac.id] = state

    def _compute_exit_path(self, start_id: int) -> Optional[List[int]]:
        """Find a traversable path from start node to the nearest exit."""
        exits = list(self.G.exits)
        if not exits:
            return None

        if start_id in exits:
            return [start_id]

        from heapq import heappush, heappop

        open_set: List[Tuple[float, int]] = []
        heappush(open_set, (0.0, start_id))

        came_from: Dict[int, int] = {}
        g_score: Dict[int, float] = {start_id: 0.0}

        exit_set = set(exits)

        def heuristic(node_id: int) -> float:
            return min(self.G.distance(node_id, exit_id) for exit_id in exits)

        while open_set:
            _, current = heappop(open_set)

            if current in exit_set:
                return self._reconstruct_path(came_from, current)

            for neighbor in self.G.neighbors(current):
                edge = self.G.get_edge(current, neighbor)
                if not edge or not edge.can_traverse():
                    continue

                hazard_mod = self.get_hazard_modifier(neighbor)
                travel_time = edge.get_traversal_time(
                    base_speed=1.2,
                    hazard_modifier=hazard_mod,
                )

                if travel_time == float("inf"):
                    continue

                tentative_g = g_score[current] + travel_time

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heappush(open_set, (tentative_g + heuristic(neighbor), neighbor))

        return None

    @staticmethod
    def _reconstruct_path(came_from: Dict[int, int], current: int) -> List[int]:
        """Reconstruct path from parent map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _update_node_occupancy(self, node_id: int) -> None:
        """Update occupancy probability based on current evacuees."""
        node = self.G.get_node(node_id)
        if not node:
            return
        evacuees = getattr(node, "evacuees", [])
        capacity = max(1.0, node.area / 2.0)
        node.occupancy_probability = min(1.0, len(evacuees) / capacity)

    def _maybe_start_evacuees(self, node_id: int) -> None:
        """Check evacuees at a node and start their evacuation if possible."""
        node = self.G.get_node(node_id)
        if not node:
            return

        for evac in list(getattr(node, "evacuees", [])):
            state = self.evac_states.get(evac.id)
            if not state or state["started"]:
                continue

            if evac.needs_assistance and not state.get("assisted", False):
                continue

            path = self._compute_exit_path(evac.node)
            if not path or len(path) <= 1:
                state["waiting_for_path"] = True
                continue

            state["path"] = path
            state["waiting_for_path"] = False
            self._start_evacuation(evac.id)

    def _start_evacuation(self, evac_id: int) -> None:
        """Kick off evacuee movement along its path."""
        state = self.evac_states.get(evac_id)
        if not state or state["started"]:
            return

        path = state.get("path")
        if not path or len(path) <= 1:
            state["waiting_for_path"] = True
            return

        state["started"] = True
        state["moving"] = True
        evac = state["evac"]
        evac.evacuating = True
        self._schedule_next_evac_segment(evac_id)

    def _schedule_next_evac_segment(self, evac_id: int) -> None:
        """Schedule the next leg of evacuee movement."""
        state = self.evac_states.get(evac_id)
        if not state:
            return

        path = state.get("path") or []
        idx = state.get("index", 0)
        if idx >= len(path) - 1:
            self._complete_evac(evac_id)
            return

        current_node_id = path[idx]
        next_node_id = path[idx + 1]

        current_node = self.G.get_node(current_node_id)
        next_node = self.G.get_node(next_node_id)
        edge = self.G.get_edge(current_node_id, next_node_id) if current_node and next_node else None

        if not current_node or not next_node or not edge or not edge.can_traverse():
            # Door or path is unavailable – re-evaluate later
            state["started"] = False
            state["moving"] = False
            state["waiting_for_path"] = True
            if current_node and state["evac"] not in current_node.evacuees:
                current_node.evacuees.append(state["evac"])
                self._update_node_occupancy(current_node.id)
            return

        evac = state["evac"]
        speed_multiplier = evac.speed_multiplier or 0.3
        if evac.needs_assistance and not state.get("assisted", False):
            # Should not happen, but guard
            speed_multiplier = 0.0

        effective_speed = max(0.4, 1.1 * speed_multiplier)
        hazard_mod = self.get_hazard_modifier(next_node_id)
        travel_time = edge.get_traversal_time(base_speed=effective_speed, hazard_modifier=hazard_mod)

        if travel_time == float("inf"):
            state["started"] = False
            state["moving"] = False
            state["waiting_for_path"] = True
            if current_node and state["evac"] not in current_node.evacuees:
                current_node.evacuees.append(state["evac"])
                self._update_node_occupancy(current_node.id)
            return

        travel_time = max(0.1, float(travel_time))

        if current_node and state["evac"] in current_node.evacuees:
            current_node.evacuees.remove(state["evac"])
            self._update_node_occupancy(current_node.id)

        state["moving"] = True
        state["departing_from"] = current_node_id
        state["segment_start_time"] = self.time
        state["segment_end_time"] = self.time + travel_time
        state["start_pos"] = (current_node.x, current_node.y)
        state["end_pos"] = (next_node.x, next_node.y)

        self.schedule(travel_time, self._evac_arrive, evac_id, next_node_id)

    def _evac_arrive(self, evac_id: int, next_node_id: int) -> None:
        """Handle evacuee arrival at the next node."""
        state = self.evac_states.get(evac_id)
        if not state:
            return

        evac = state["evac"]
        path = state.get("path") or []
        idx = state.get("index", 0)

        if idx >= len(path) - 1:
            self._complete_evac(evac_id)
            return

        current_node_id = path[idx]
        next_node = self.G.get_node(next_node_id)
        current_node = self.G.get_node(current_node_id)

        state["index"] = idx + 1
        state["segment_start_time"] = None
        state["segment_end_time"] = None
        state["start_pos"] = (next_node.x, next_node.y) if next_node else state["start_pos"]
        state["end_pos"] = state["start_pos"]
        state["position"] = (next_node.x, next_node.y) if next_node else state.get("position", (0.0, 0.0))

        evac.node = next_node_id

        if next_node:
            if next_node.node_type == NodeType.EXIT:
                self._complete_evac(evac_id)
            else:
                if evac not in next_node.evacuees:
                    next_node.evacuees.append(evac)
                self._update_node_occupancy(next_node.id)
                # Continue to the following segment
                self._schedule_next_evac_segment(evac_id)

        if current_node:
            self._update_node_occupancy(current_node.id)

    def _complete_evac(self, evac_id: int) -> None:
        """Mark evacuee as safely outside."""
        state = self.evac_states.get(evac_id)
        if not state:
            return

        evac = state["evac"]
        evac.outside = True
        evac.evacuating = False
        state["moving"] = False
        state["segment_start_time"] = None
        state["segment_end_time"] = None
        exit_node = self.G.get_node(evac.node)
        if exit_node:
            state["position"] = (exit_node.x, exit_node.y)
            self._update_node_occupancy(exit_node.id)

    def get_evacuee_states(self) -> List[Dict[str, Any]]:
        """Expose evacuee states for visualization."""
        states = []
        for evac_id, state in self.evac_states.items():
            evac = state["evac"]
            position = state.get("position", (0.0, 0.0))

            # Interpolate position if in transit
            start_time = state.get("segment_start_time")
            end_time = state.get("segment_end_time")
            if state.get("moving") and start_time is not None and end_time is not None and end_time > start_time:
                t = (self.time - start_time) / (end_time - start_time)
                t = max(0.0, min(1.0, t))
                sx, sy = state.get("start_pos", position)
                ex, ey = state.get("end_pos", position)
                position = (sx + (ex - sx) * t, sy + (ey - sy) * t)

            states.append(
                {
                    "id": evac_id,
                    "x": position[0],
                    "y": position[1],
                    "outside": evac.outside,
                    "moving": state.get("moving", False),
                    "node": evac.node,
                }
            )
        return states

    def get_evacuee_summary(self) -> Dict[str, int]:
        """Return aggregate evacuee statistics."""
        safe = sum(1 for state in self.evac_states.values() if state["evac"].outside)
        moving = sum(1 for state in self.evac_states.values() if state.get("moving", False))
        return {
            "total": self._total_evacuees,
            "safe": safe,
            "moving": moving,
        }

    def mark_evacuee_assisted(self, evacuee_id: int) -> None:
        """Mark that an evacuee has received assistance and can start moving."""
        state = self.evac_states.get(evacuee_id)
        if not state:
            return
        state["assisted"] = True
        evac = state["evac"]
        evac.evacuating = True
        self._maybe_start_evacuees(evac.node)

    def notify_edge_open(self, src: int, dst: int) -> None:
        """Notify evac system that connectivity changed between nodes."""
        self._maybe_start_evacuees(src)
        self._maybe_start_evacuees(dst)

    # ------------------------------------------------------------------

    
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
    
    def register_agent_movement(self, agent_id: int, start_node: int, end_node: int, travel_time: float) -> None:
        """Register an agent moving between two nodes for visualization."""
        self.movement_tracker[agent_id] = {
            "start_node": start_node,
            "end_node": end_node,
            "start_time": self.time,
            "end_time": self.time + float(max(0.0, travel_time)),
            "travel_time": float(max(0.01, travel_time)),
        }
    
    def finish_agent_movement(self, agent_id: int) -> None:
        """Clear movement tracking when agent arrives."""
        self.movement_tracker.pop(agent_id, None)
    
    def step(self, dt: float) -> None:
        """
        Advance simulation by a time step (for real-time visualization).
        
        Args:
            dt: Time delta to advance (seconds)
        """
        target_time = self.time + dt
        
        # Process events up to target time
        while self._event_queue and self._event_queue[0].time <= target_time:
            event = heapq.heappop(self._event_queue)
            self.time = event.time
            
            # Execute event
            try:
                event.fn(*event.args)
            except Exception as e:
                print(f"[World] Error in event {event.event_id} at t={self.time:.2f}: {e}")
            
            # HASO: Update hazards periodically
            if self.time >= self._next_hazard_update:
                self.update_hazards(self.hazard_update_interval)
                self._next_hazard_update = self.time + self.hazard_update_interval
            
            # Record history periodically
            if self.time >= self._next_record_time:
                self._record_state()
                self._next_record_time = self.time + self.record_interval
        
        # Advance time to target even if no events
        if self.time < target_time:
            self.time = target_time
    
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
            
            # HASO: Update hazards periodically
            if self.time >= self._next_hazard_update:
                self.update_hazards(self.hazard_update_interval)
                self._next_hazard_update = self.time + self.hazard_update_interval
            
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
    
    def find_nearest_uncleared_room(
        self,
        from_node: int,
        max_candidates: int = 10,
        allowed_nodes: Optional[Set[int]] = None,
    ) -> Optional[int]:
        """
        Find nearest uncleared room from a given node.
        
        Args:
            from_node: starting node ID
            max_candidates: maximum number of candidates to consider
            allowed_nodes: optional set of node IDs to restrict search
        
        Returns:
            Node ID of nearest uncleared room, or None
        """
        uncleared = [
            nid for nid, node in self.G.nodes.items()
            if not node.cleared
            and node.node_type not in [NodeType.CORRIDOR, NodeType.EXIT, NodeType.CHECKPOINT]
            and self.fog.is_known(nid)
            and (allowed_nodes is None or nid in allowed_nodes)
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
    
    # ========== HASO METHODS ==========
    
    def update_hazards(self, dt: float) -> None:
        """
        HASO Step 4: Dynamic hazard propagation.
        
        Updates fire spread and smoke visibility using:
        - h_i(t+1) = h_i(t) + Δt⋅k (fire spread)
        - v_i(t+1) = v_i(t) - γ (smoke visibility decay)
        
        Args:
            dt: Time delta since last update (seconds)
        """
        import random
        
        for node in self.G.nodes.values():
            # Fire spread to neighbors
            if node.hazard == HazardType.FIRE and node.hazard_severity > 0.3:
                for neighbor_id in self.G.neighbors(node.id):
                    neighbor = self.G.get_node(neighbor_id)
                    if neighbor and neighbor.hazard != HazardType.FIRE:
                        # Probability-based fire spread
                        if random.random() < self.fire_spread_probability * node.hazard_severity:
                            neighbor.hazard = HazardType.FIRE
                            neighbor.hazard_severity = 0.2
                            neighbor.fire_spread_rate = 0.05
            
            # Fire intensity growth
            if node.hazard == HazardType.FIRE:
                node.hazard_severity = min(1.0, node.hazard_severity + dt * node.fire_spread_rate / 100.0)
                # Create smoke in adjacent rooms
                for neighbor_id in self.G.neighbors(node.id):
                    neighbor = self.G.get_node(neighbor_id)
                    if neighbor and neighbor.hazard == HazardType.NONE:
                        neighbor.hazard = HazardType.SMOKE
                        neighbor.hazard_severity = 0.1
            
            # Smoke visibility decay
            if node.hazard == HazardType.SMOKE:
                node.visibility = max(0.0, node.visibility - node.smoke_decay_rate * dt / 10.0)
                node.hazard_severity = min(1.0, node.hazard_severity + dt * 0.001)
            
            # Fire reduces visibility
            if node.hazard == HazardType.FIRE:
                node.visibility = max(0.1, 1.0 - node.hazard_severity * 0.8)
    
    def shortest_path_haso(
        self,
        start: int,
        goal: int,
        beta: Optional[float] = None,
        lambda_: Optional[float] = None,
    ) -> Optional[List[int]]:
        """
        HASO Step 4: A* pathfinding with dynamic hazard and visibility costs.
        
        Minimizes cost function:
        C(a_i) = Σ[t(e_ij) + β⋅h_j(t) + λ⋅(1-v_j(t))]
        
        Args:
            start: Starting node ID
            goal: Goal node ID
            beta: Hazard penalty weight (default from self.weights)
            lambda_: Visibility penalty weight (default from self.weights)
        
        Returns:
            Path as list of node IDs, or None if no path exists
        """
        if start == goal:
            return [start]
        
        # Use weights from world if not provided
        if beta is None:
            beta = self.weights["hazard_penalty"]
        if lambda_ is None:
            lambda_ = self.weights["visibility_penalty"]
        
        # Check if goal is known
        if not self.fog.is_known(goal):
            return None
        
        # A* search with HASO cost function
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
                if not edge:
                    continue
                
                # HASO cost function
                node = self.G.get_node(neighbor)
                if node:
                    # Base traversal time
                    hazard_mod = self.get_hazard_modifier(neighbor)
                    prep_cost = 0.0
                    traversable = edge.traversable and edge.is_open
                    if not traversable:
                        if edge.can_open_edge():
                            prep_cost = edge.get_open_time()
                            traversable = True
                        elif edge.can_break_edge():
                            prep_cost = edge.get_break_time()
                            traversable = True
                        else:
                            traversable = False
                    if not traversable:
                        continue
                    props = edge.get_properties()
                    speed_modifier = props.get('speed_modifier', 1.0) or 1.0
                    effective_speed = 1.5 * hazard_mod * speed_modifier
                    if effective_speed <= 0:
                        continue
                    travel_time = edge.length / effective_speed
                    edge_cost = prep_cost + travel_time
                    
                    # Add hazard penalty: β⋅h_j(t)
                    hazard_cost = beta * node.hazard_severity
                    
                    # Add visibility penalty: λ⋅(1-v_j(t))
                    visibility_cost = lambda_ * (1.0 - node.visibility)
                    
                    # Total cost
                    total_cost = edge_cost + hazard_cost + visibility_cost
                    
                    tentative_g = g_score[current] + total_cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.G.distance(neighbor, goal)
                        heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def init_zones(self, num_zones: Optional[int] = None) -> None:
        """
        HASO Step 3: Initialize zone partitioning.
        
        Args:
            num_zones: Number of zones (defaults to number of agents)
        """
        from .zone_optimizer import partition_building_zones, assign_responders_to_zones
        
        if num_zones is None:
            num_zones = len(self.agents)
        
        # Partition building into zones
        self.zones = partition_building_zones(self.G, num_zones)
        
        # Assign agents to zones (prefer ILP optimization when available)
        assignment: Dict[int, int] = {}
        if optimize_zone_assignment_ilp is not None:
            try:
                assignment = optimize_zone_assignment_ilp(self.zones, self.agents, self.G)
            except Exception as exc:  # pragma: no cover
                print(f"[HASO] ILP assignment failed ({exc}); falling back to greedy assignment.")
                assignment = {}

        if not assignment:
            assignment = assign_responders_to_zones(self.zones, self.agents, self.G)

        self.agent_zones = assignment
        
        print(f"[HASO] Partitioned building into {len(self.zones)} zones")
        for agent in self.agents:
            if agent.id in self.agent_zones:
                zone_size = len(self.zones.get(self.agent_zones[agent.id], []))
                print(f"[HASO] Agent {agent.id} ({agent.role.name}) -> Zone {self.agent_zones[agent.id]} ({zone_size} rooms)")
    
    def reallocate_failed_zone(self, failed_agent_id: int) -> None:
        """
        HASO Step 3: Dynamic zone reallocation when agent fails.
        
        Args:
            failed_agent_id: ID of agent that failed or slowed down
        """
        from .zone_optimizer import reallocate_zone_dynamic
        
        self.agent_zones = reallocate_zone_dynamic(
            failed_agent_id,
            self.agents,
            self.zones,
            self.G
        )
        
        print(f"[HASO] Reallocated zones after Agent {failed_agent_id} failure")

