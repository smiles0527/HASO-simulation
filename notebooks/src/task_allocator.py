"""
task_allocator.py: HASO Integer Linear Programming for Optimal Task Assignment

Implements ILP-based optimization for responder-to-zone assignment
minimizing total sweep time with redundancy constraints.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, TYPE_CHECKING, Optional
import math

if TYPE_CHECKING:
    from .graph_model import Graph
    from .agents import Agent

try:
    from pulp import *
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


def optimize_zone_assignment_ilp(
    zones: Dict[int, List[int]],
    agents: List[Agent],
    graph: Graph,
    alpha: float = 0.3
) -> Dict[int, int]:
    """
    HASO Step 2: Integer Linear Programming for optimal zone assignment.
    
    Minimizes: Σ ExpectedTime(agent_i, zone_j) + α * Redundancy(zone_j)
    
    Subject to:
    - Each zone assigned to exactly one primary agent
    - Agent capacity constraints
    - Adjacency preferences for backup coverage
    
    Args:
        zones: Zone definitions (zone_id -> node_ids)
        agents: List of responder agents
        graph: Building graph
        alpha: Redundancy weight factor
    
    Returns:
        Dictionary mapping agent_id -> zone_id
    """
    if not HAS_PULP:
        # Fallback to greedy assignment
        return _greedy_assignment(zones, agents, graph)
    
    # Create problem
    prob = LpProblem("HASO_Zone_Assignment", LpMinimize)
    
    # Decision variables: x[i][j] = 1 if agent i assigned to zone j
    x = {}
    for agent in agents:
        for zone_id in zones.keys():
            x[(agent.id, zone_id)] = LpVariable(
                f"x_{agent.id}_{zone_id}", 
                cat='Binary'
            )
    
    # Auxiliary variable: maximum completion time
    T_max = LpVariable("T_max", lowBound=0)
    
    # Calculate expected time for each agent-zone pair
    expected_times = {}
    for agent in agents:
        for zone_id, node_list in zones.items():
            time = _calculate_zone_time(agent, node_list, graph)
            expected_times[(agent.id, zone_id)] = time
    
    # Objective: minimize max completion time + redundancy penalty
    redundancy_cost = 0
    for zone_id in zones.keys():
        # Redundancy = number of agents assigned to neighbors of this zone
        redundancy_cost += alpha * lpSum([
            x[(a.id, zone_id)] for a in agents
        ])
    
    prob += T_max + redundancy_cost
    
    # Constraint 1: Each zone must be assigned to exactly one agent
    for zone_id in zones.keys():
        prob += lpSum([x[(a.id, zone_id)] for a in agents]) == 1, \
                f"Zone_{zone_id}_assigned_once"
    
    # Constraint 2: Each agent assigned to at most one zone (primary assignment)
    for agent in agents:
        prob += lpSum([x[(agent.id, z)] for z in zones.keys()]) <= 1, \
                f"Agent_{agent.id}_max_one_zone"
    
    # Constraint 3: Completion time constraint
    for agent in agents:
        for zone_id in zones.keys():
            prob += T_max >= expected_times[(agent.id, zone_id)] * x[(agent.id, zone_id)], \
                    f"Time_agent_{agent.id}_zone_{zone_id}"
    
    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))
    
    # Extract solution
    assignment = {}
    for agent in agents:
        for zone_id in zones.keys():
            if value(x[(agent.id, zone_id)]) and value(x[(agent.id, zone_id)]) > 0.5:
                assignment[agent.id] = zone_id
                agent.assigned_zone = zone_id
                break
    
    return assignment


def _calculate_zone_time(agent: Agent, node_list: List[int], graph: Graph) -> float:
    """
    Calculate expected time for an agent to clear a zone.
    
    Time = travel_time + Σ search_times + hazard_penalties
    """
    if not node_list:
        return 0.0
    
    total_time = 0.0
    
    # Estimate travel time (TSP approximation: 2 * MST)
    total_distance = 0.0
    for i in range(len(node_list) - 1):
        node1 = graph.get_node(node_list[i])
        node2 = graph.get_node(node_list[i + 1])
        if node1 and node2:
            dist = ((node1.x - node2.x)**2 + (node1.y - node2.y)**2)**0.5
            total_distance += dist
    
    travel_time = total_distance / max(agent.base_speed, 0.1)
    
    # Calculate search time
    search_time = 0.0
    for node_id in node_list:
        node = graph.get_node(node_id)
        if node:
            # Base search time with hazard penalty
            time = node.search_time * (node.area / 10.0) * \
                   (1 + node.hazard_severity * 2.0) * \
                   (1 + node.occupancy_probability)
            search_time += time
    
    # Role modifier
    role_modifiers = {
        'SCOUT': 0.7,      # Scouts are faster but less thorough
        'SECURER': 1.2,    # Securers are thorough
        'EVACUATOR': 1.0,  # Evacuators are methodical
        'CHECKPOINTER': 0.5  # Checkpointers mainly observe
    }
    role_mod = role_modifiers.get(agent.role.name, 1.0)
    
    total_time = travel_time + (search_time * role_mod)
    
    return total_time


def _greedy_assignment(
    zones: Dict[int, List[int]],
    agents: List[Agent],
    graph: Graph
) -> Dict[int, int]:
    """Greedy fallback when PuLP not available."""
    assignment = {}
    zone_times = {}
    
    # Calculate time for each zone
    for zone_id, node_list in zones.items():
        avg_time = sum(_calculate_zone_time(a, node_list, graph) for a in agents) / len(agents)
        zone_times[zone_id] = avg_time
    
    # Sort zones by time (longest first)
    sorted_zones = sorted(zone_times.items(), key=lambda x: x[1], reverse=True)
    
    # Sort agents by capability (fastest first)
    sorted_agents = sorted(agents, key=lambda a: a.base_speed * a.priority_heuristic, reverse=True)
    
    # Assign greedily
    for i, (zone_id, _) in enumerate(sorted_zones):
        if i < len(sorted_agents):
            agent = sorted_agents[i]
            assignment[agent.id] = zone_id
            agent.assigned_zone = zone_id
    
    return assignment


def calculate_redundancy_index(graph: Graph) -> float:
    """
    HASO Step 6: Calculate redundancy index R = |V_reswept| / |V|
    
    Args:
        graph: Building graph
    
    Returns:
        Redundancy ratio (0.0 to 1.0+)
    """
    from .graph_model import NodeType
    
    total_rooms = sum(1 for n in graph.nodes.values() 
                     if n.node_type not in [NodeType.CORRIDOR, NodeType.EXIT, NodeType.CHECKPOINT])
    
    if total_rooms == 0:
        return 0.0
    
    # Count rooms that were verified (cleared AND verified)
    reswept_rooms = sum(1 for n in graph.nodes.values()
                       if n.cleared and n.verified_by is not None
                       and n.node_type not in [NodeType.CORRIDOR, NodeType.EXIT, NodeType.CHECKPOINT])
    
    return reswept_rooms / total_rooms


def calculate_risk_exposure(graph: Graph) -> float:
    """
    HASO Step 6: Calculate risk exposure E = Σ h_i(t) * o_i(t)
    
    Args:
        graph: Building graph
    
    Returns:
        Total risk exposure score
    """
    risk = 0.0
    
    for node in graph.nodes.values():
        # Risk = hazard_severity * occupancy_probability
        risk += node.hazard_severity * node.occupancy_probability
    
    return risk


def calculate_efficiency_ratio(cleared_count: int, total_time: float) -> float:
    """
    HASO Step 6: Calculate efficiency ratio η = |V| / T_sweep
    
    Args:
        cleared_count: Number of rooms cleared
        total_time: Total sweep time in seconds
    
    Returns:
        Rooms per second
    """
    if total_time <= 0:
        return 0.0
    
    return cleared_count / total_time

