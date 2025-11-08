"""
agents.py: Responder agents with role-based behavior and state machines.

Defines responder types, status, and action enums for discrete-event simulation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


class Role(Enum):
    """
    Responder roles with different responsibilities.
    
    - SCOUT: First enters, fast moving, checks room status, tags evacuees, signals priorities
    - SECURER: Second enters, slower moving, secures hazards, assists evacuees, clears rooms
    - CHECKPOINTER: Stays at checkpoints to secure areas, assists in any duty
    - EVACUATOR: Double checks rooms are clear, last to leave, closes areas off
    """
    SCOUT = auto()
    SECURER = auto()
    CHECKPOINTER = auto()
    EVACUATOR = auto()


class Status(Enum):
    """
    Responder status affecting movement and capabilities.
    
    Status affects speed and vision:
    - NORMAL (0): Fast movement, full vision
    - SLOWED (1): Reduced movement and vision
    - PROGRESSING (3): Staying on node, performing action
    - IMMOBILIZED (4): Stopped, has vision
    - INCAPACITATED (5): Stopped, no vision, can be moved by others
    - DEAD (6): Stopped, no vision, cannot be moved
    - EXITED (7): Left the simulation
    """
    NORMAL = 0
    SLOWED = 1
    PROGRESSING = 3
    IMMOBILIZED = 4
    INCAPACITATED = 5
    DEAD = 6
    EXITED = 7


class Action(Enum):
    """
    Actions responders can perform, each with associated status codes.
    
    Format: ACTION (status_code)
    """
    SCOUTING = 1          # S=1
    MOVING = 0            # S=0
    SECURING = 3          # S=3
    CHECKING = 3          # S=3
    SIGNALLING = 4        # S=4
    ASSISTING = 2         # S=2
    CHECKPOINTING = 4     # S=4
    LIGHTLY_INJURED = 2   # S=2
    SEVERELY_INJURED = 4  # S=4
    INCAPACITATED = 5     # S=5
    DEAD = 6              # S=6
    EXITED_SITE = 7       # S=7
    
    def to_status(self) -> Status:
        """Convert action to corresponding status."""
        status_map = {
            0: Status.NORMAL,
            1: Status.SLOWED,
            2: Status.SLOWED,
            3: Status.PROGRESSING,
            4: Status.IMMOBILIZED,
            5: Status.INCAPACITATED,
            6: Status.DEAD,
            7: Status.EXITED,
        }
        return status_map.get(self.value, Status.NORMAL)


@dataclass
class Agent:
    """
    Represents a responder agent in the evacuation simulation.
    
    Agents have:
    - Role-based behavior (scout, securer, etc.)
    - State machine (status, action)
    - Movement and pathfinding
    - Task queue and decision making
    """
    id: int
    role: Role
    node: int  # current node ID
    
    # Movement and sweep strategy
    sweep_mode: str = "right"  # "right", "left", "corridor"
    personal_priority: int = 3  # 1-5, affects task allocation
    
    # State
    status: Status = Status.NORMAL
    action: Action = Action.MOVING
    
    # Performance characteristics
    base_speed: float = 1.5     # m/s
    vision_range: int = 1       # how many edges away can see
    
    # HASO: Agent parameters
    communication_reliability: float = 0.95  # c_r
    vision_radius: float = 5.0  # r_v in meters
    priority_heuristic: float = 1.0  # π weighting factor
    assigned_zone: int = -1  # Zone assignment from macro-level planner
    
    # Task and path
    target_node: Optional[int] = None
    path: List[int] = field(default_factory=list)
    task_queue: List[Tuple[str, int]] = field(default_factory=list)  # [(task_type, node_id), ...]
    
    # Tracking
    visited_nodes: set = field(default_factory=set)
    rooms_cleared: int = 0
    evacuees_assisted: int = 0
    distance_traveled: float = 0.0
    
    # Communication and signaling
    signals: List[Tuple[int, str]] = field(default_factory=list)  # [(node_id, signal_type), ...]
    
    # Logging
    log: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize visited nodes with starting position."""
        self.visited_nodes.add(self.node)
    
    @property
    def is_active(self) -> bool:
        """Check if agent is active (can perform actions)."""
        return self.status in [Status.NORMAL, Status.SLOWED, Status.PROGRESSING]
    
    @property
    def can_move(self) -> bool:
        """Check if agent can move."""
        return self.status in [Status.NORMAL, Status.SLOWED]
    
    @property
    def has_vision(self) -> bool:
        """Check if agent has vision."""
        return self.status not in [Status.INCAPACITATED, Status.DEAD, Status.EXITED]
    
    def get_effective_speed(self, hazard_modifier: float = 1.0) -> float:
        """
        Calculate effective movement speed based on status and hazards.
        
        Args:
            hazard_modifier: speed multiplier from environmental hazards
        
        Returns:
            Effective speed in m/s
        """
        speed = self.base_speed
        
        # Status modifiers
        if self.status == Status.SLOWED:
            speed *= 0.6
        elif self.status == Status.PROGRESSING:
            speed = 0.0  # not moving
        elif self.status in [Status.IMMOBILIZED, Status.INCAPACITATED, Status.DEAD]:
            speed = 0.0
        
        # Role modifiers
        if self.role == Role.SCOUT:
            speed *= 1.3  # scouts are faster
        elif self.role == Role.SECURER:
            speed *= 0.9  # securers carry more equipment
        elif self.role == Role.CHECKPOINTER:
            speed *= 1.0  # normal speed
        elif self.role == Role.EVACUATOR:
            speed *= 0.95  # slightly slower, more thorough
        
        # Apply hazard modifier
        speed *= hazard_modifier
        
        return max(0.0, speed)
    
    def get_effective_vision(self) -> int:
        """Get effective vision range based on status."""
        if not self.has_vision:
            return 0
        
        base = self.vision_range
        
        # Status affects vision
        if self.status == Status.SLOWED:
            base = max(1, base - 1)
        elif self.status == Status.IMMOBILIZED:
            base = max(1, base - 1)
        
        return base
    
    def add_task(self, task_type: str, node_id: int, priority: bool = False) -> None:
        """
        Add a task to the agent's queue.
        
        Args:
            task_type: "scout", "secure", "check", "assist", "signal"
            node_id: target node for the task
            priority: if True, add to front of queue
        """
        task = (task_type, node_id)
        if priority:
            self.task_queue.insert(0, task)
        else:
            self.task_queue.append(task)
    
    def pop_task(self) -> Optional[Tuple[str, int]]:
        """Get and remove the next task from queue."""
        if self.task_queue:
            return self.task_queue.pop(0)
        return None
    
    def has_tasks(self) -> bool:
        """Check if agent has pending tasks."""
        return len(self.task_queue) > 0
    
    def add_signal(self, node_id: int, signal_type: str) -> None:
        """
        Add a signal for other responders.
        
        Args:
            node_id: location of signal
            signal_type: "priority", "hazard", "evacuees", "cleared", "checkpoint"
        """
        self.signals.append((node_id, signal_type))
        self.log_action(f"Signaled {signal_type} at node {node_id}")
    
    def log_action(self, message: str) -> None:
        """Add a log entry for this agent."""
        self.log.append(f"[Agent {self.id}/{self.role.name}] {message}")
    
    def set_action(self, action: Action) -> None:
        """Set current action and update status accordingly."""
        self.action = action
        self.status = action.to_status()
    
    def move_to_node(self, node_id: int) -> None:
        """Update agent position to new node."""
        self.node = node_id
        self.visited_nodes.add(node_id)
        self.log_action(f"Moved to node {node_id}")
    
    def clear_room(self, node_id: int) -> None:
        """Mark that this agent cleared a room."""
        self.rooms_cleared += 1
        self.log_action(f"Cleared room at node {node_id}")
    
    def assist_evacuee(self, evacuee_id: int) -> None:
        """Mark that this agent assisted an evacuee."""
        self.evacuees_assisted += 1
        self.log_action(f"Assisted evacuee {evacuee_id}")


class SweepMode:
    """
    Sweep strategies for building traversal.
    
    - Right-hand: Follow right wall
    - Left-hand: Follow left wall
    - Corridor: Prioritize corridors, then attached rooms
    """
    
    @staticmethod
    def right_hand_order(current: int, neighbors: List[int], prev: Optional[int] = None) -> List[int]:
        """
        Order neighbors for right-hand wall following.
        
        Simple approximation: reverse order of neighbors
        (In practice, would use geometric angles)
        """
        if prev is not None and prev in neighbors:
            # Put the "right" neighbor first (opposite of where we came from)
            idx = neighbors.index(prev)
            ordered = neighbors[idx+1:] + neighbors[:idx]
            return list(reversed(ordered))
        return list(reversed(neighbors))
    
    @staticmethod
    def left_hand_order(current: int, neighbors: List[int], prev: Optional[int] = None) -> List[int]:
        """Order neighbors for left-hand wall following."""
        if prev is not None and prev in neighbors:
            idx = neighbors.index(prev)
            ordered = neighbors[idx+1:] + neighbors[:idx]
            return ordered
        return neighbors
    
    @staticmethod
    def corridor_priority(neighbors: List[int], graph, current: int) -> List[int]:
        """
        Order neighbors prioritizing corridors and checkpoints.
        
        Args:
            neighbors: list of neighbor node IDs
            graph: Graph object to check node types
            current: current node ID
        
        Returns:
            Ordered list with corridors first, then other nodes
        """
        from .graph_model import NodeType
        
        corridors = []
        rooms = []
        
        for n in neighbors:
            node = graph.get_node(n)
            if node:
                if node.node_type in [NodeType.CORRIDOR, NodeType.CHECKPOINT]:
                    corridors.append(n)
                else:
                    rooms.append(n)
        
        # Return corridors first, then rooms
        return corridors + rooms


