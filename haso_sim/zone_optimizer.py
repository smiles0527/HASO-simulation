"""
zone_optimizer.py: HASO Macro-Level Partitioning and Zone Assignment

Implements hierarchical graph partitioning and dynamic zone reallocation
for optimal responder assignment using community detection algorithms.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set, TYPE_CHECKING
import random
from collections import defaultdict

if TYPE_CHECKING:
    from .graph_model import Graph
    from .agents import Agent

try:
    import networkx as nx
    from networkx.algorithms import community
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def partition_building_zones(graph: Graph, num_zones: int, balance_weight: float = 0.5) -> Dict[int, List[int]]:
    """
    HASO Step 3: Hierarchical partitioning of building into zones.
    
    Uses community detection to divide the building graph into balanced zones,
    minimizing inter-zone edges while balancing room importance.
    
    Args:
        graph: Building graph
        num_zones: Number of zones to create (typically = number of responders)
        balance_weight: Weight for balancing zone sizes (0.0 to 1.0)
    
    Returns:
        Dictionary mapping zone_id -> list of node_ids in that zone
    """
    if not HAS_NETWORKX:
        # Fallback: simple spatial partitioning
        return _spatial_partition(graph, num_zones)
    
    # Convert to NetworkX graph
    G_nx = nx.Graph()
    
    # Add nodes with room importance weights
    for node_id, node in graph.nodes.items():
        importance = (6 - node.room_priority) * node.area  # Higher priority = higher weight
        G_nx.add_node(node_id, weight=importance, pos=(node.x, node.y))
    
    # Add edges with traversal time weights
    for (src, dst), edge in graph.edges.items():
        if src < dst:  # Only add once (undirected)
            weight = edge.length / max(0.1, edge.traversable * 1.0)
            G_nx.add_edge(src, dst, weight=weight)
    
    # Use Louvain community detection for initial partitioning
    try:
        communities = community.greedy_modularity_communities(G_nx, weight='weight')
    except:
        # Fallback if greedy_modularity fails
        communities = _simple_partition(G_nx, num_zones)
    
    # Convert communities to zones
    zones = {}
    for i, comm in enumerate(list(communities)[:num_zones]):
        zones[i] = list(comm)
    
    # If we have fewer communities than zones, split largest
    while len(zones) < num_zones:
        largest_zone_id = max(zones.keys(), key=lambda k: len(zones[k]))
        largest_zone = zones[largest_zone_id]
        
        # Split largest zone in half
        mid = len(largest_zone) // 2
        zones[len(zones)] = largest_zone[mid:]
        zones[largest_zone_id] = largest_zone[:mid]
    
    # Balance zones if needed
    if balance_weight > 0:
        zones = _balance_zones(zones, graph, balance_weight)
    
    # Update graph nodes with zone assignments
    for zone_id, node_list in zones.items():
        for node_id in node_list:
            node = graph.get_node(node_id)
            if node:
                node.zone_id = zone_id
    
    return zones


def assign_responders_to_zones(
    zones: Dict[int, List[int]], 
    agents: List[Agent], 
    graph: Graph
) -> Dict[int, int]:
    """
    HASO Step 2: Assign responders to zones minimizing expected completion time.
    
    Uses greedy assignment with priority heuristics.
    
    Args:
        zones: Zone assignments (zone_id -> node_ids)
        agents: List of responder agents
        graph: Building graph
    
    Returns:
        Dictionary mapping agent_id -> zone_id
    """
    assignment = {}
    zone_loads = {z: 0.0 for z in zones.keys()}
    
    # Calculate zone workloads
    zone_workload = {}
    for zone_id, node_list in zones.items():
        workload = 0.0
        for node_id in node_list:
            node = graph.get_node(node_id)
            if node:
                # Workload = search time + hazard penalty + occupancy factor
                workload += (node.search_time * node.area / 10.0) * \
                           (1 + node.hazard_severity * 2) * \
                           (1 + node.occupancy_probability)
        zone_workload[zone_id] = workload
    
    # Sort agents by priority (Scouts first, then Securers, etc.)
    role_priority = {'SCOUT': 4, 'SECURER': 3, 'EVACUATOR': 2, 'CHECKPOINTER': 1}
    sorted_agents = sorted(agents, 
                          key=lambda a: role_priority.get(a.role.name, 0) * a.priority_heuristic,
                          reverse=True)
    
    # Greedy assignment: assign each agent to zone with highest workload/speed ratio
    for agent in sorted_agents:
        # Find zone with maximum workload / agent_efficiency
        best_zone = None
        best_score = -float('inf')
        
        for zone_id in zones.keys():
            if zone_id in zone_loads:
                # Score = workload / (current_load + agent_speed)
                efficiency = agent.base_speed * agent.priority_heuristic
                score = zone_workload[zone_id] / (zone_loads[zone_id] + efficiency)
                
                if score > best_score:
                    best_score = score
                    best_zone = zone_id
        
        if best_zone is not None:
            assignment[agent.id] = best_zone
            agent.assigned_zone = best_zone
            zone_loads[best_zone] += agent.base_speed * agent.priority_heuristic
    
    return assignment


def reallocate_zone_dynamic(
    failed_agent_id: int,
    agents: List[Agent],
    zones: Dict[int, List[int]],
    graph: Graph
) -> Dict[int, int]:
    """
    HASO Step 3: Dynamic zone reallocation when responder fails or slows down.
    
    Reassigns the failed agent's zone to nearby active agents.
    
    Args:
        failed_agent_id: ID of agent that failed/slowed
        agents: All agents
        zones: Current zone assignments
        graph: Building graph
    
    Returns:
        Updated zone assignments
    """
    failed_agent = next((a for a in agents if a.id == failed_agent_id), None)
    if not failed_agent or failed_agent.assigned_zone == -1:
        return {a.id: a.assigned_zone for a in agents if a.assigned_zone != -1}
    
    failed_zone = failed_agent.assigned_zone
    failed_agent.assigned_zone = -1
    
    # Find nearest active agent to take over the zone
    active_agents = [a for a in agents if a.is_active and a.id != failed_agent_id]
    
    if not active_agents:
        return {a.id: a.assigned_zone for a in agents if a.assigned_zone != -1}
    
    # Find agent closest to the failed zone
    failed_zone_nodes = zones.get(failed_zone, [])
    if not failed_zone_nodes:
        return {a.id: a.assigned_zone for a in agents if a.assigned_zone != -1}
    
    # Calculate centroid of failed zone
    zone_x = sum(graph.get_node(n).x for n in failed_zone_nodes if graph.get_node(n)) / len(failed_zone_nodes)
    zone_y = sum(graph.get_node(n).y for n in failed_zone_nodes if graph.get_node(n)) / len(failed_zone_nodes)
    
    # Find closest agent
    closest_agent = min(active_agents, 
                       key=lambda a: ((graph.get_node(a.node).x - zone_x)**2 + 
                                     (graph.get_node(a.node).y - zone_y)**2)**0.5)
    
    # Merge failed zone into closest agent's zone
    if closest_agent.assigned_zone in zones:
        zones[closest_agent.assigned_zone].extend(failed_zone_nodes)
    else:
        closest_agent.assigned_zone = failed_zone
    
    return {a.id: a.assigned_zone for a in agents if a.assigned_zone != -1}


def _spatial_partition(graph: Graph, num_zones: int) -> Dict[int, List[int]]:
    """Simple spatial partitioning fallback (k-means like)."""
    zones = {i: [] for i in range(num_zones)}
    
    # Get all nodes with positions
    nodes = [(nid, n.x, n.y) for nid, n in graph.nodes.items()]
    
    if not nodes:
        return zones
    
    # Simple grid-based partitioning
    min_x = min(x for _, x, _ in nodes)
    max_x = max(x for _, x, _ in nodes)
    min_y = min(y for _, _, y in nodes)
    max_y = max(y for _, _, y in nodes)
    
    for node_id, x, y in nodes:
        # Determine zone based on position
        zone_x = int((x - min_x) / ((max_x - min_x + 1) / num_zones))
        zone_id = min(zone_x, num_zones - 1)
        zones[zone_id].append(node_id)
    
    return zones


def _simple_partition(G_nx, num_zones: int) -> List[Set[int]]:
    """Simple partition for fallback."""
    nodes = list(G_nx.nodes())
    chunk_size = len(nodes) // num_zones + 1
    
    communities = []
    for i in range(num_zones):
        start = i * chunk_size
        end = start + chunk_size
        communities.append(set(nodes[start:end]))
    
    return communities


def _balance_zones(zones: Dict[int, List[int]], graph: Graph, alpha: float) -> Dict[int, List[int]]:
    """Balance zone sizes and workloads."""
    # Calculate zone weights
    zone_weights = {}
    for zone_id, node_list in zones.items():
        weight = sum(
            graph.get_node(nid).area * (6 - graph.get_node(nid).room_priority)
            for nid in node_list if graph.get_node(nid)
        )
        zone_weights[zone_id] = weight
    
    avg_weight = sum(zone_weights.values()) / len(zone_weights) if zone_weights else 0
    
    # Move nodes from heavy zones to light zones
    max_iterations = 10
    for _ in range(max_iterations):
        heavy_zones = [z for z, w in zone_weights.items() if w > avg_weight * (1 + alpha)]
        light_zones = [z for z, w in zone_weights.items() if w < avg_weight * (1 - alpha)]
        
        if not heavy_zones or not light_zones:
            break
        
        # Move one node from heaviest to lightest
        heaviest = max(heavy_zones, key=lambda z: zone_weights[z])
        lightest = min(light_zones, key=lambda z: zone_weights[z])
        
        if zones[heaviest]:
            node_id = zones[heaviest].pop()
            zones[lightest].append(node_id)
            
            # Update weights
            node = graph.get_node(node_id)
            if node:
                node_weight = node.area * (6 - node.room_priority)
                zone_weights[heaviest] -= node_weight
                zone_weights[lightest] += node_weight
    
    return zones

