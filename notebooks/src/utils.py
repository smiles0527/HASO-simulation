"""
utils.py: Utility functions for configuration, logging, and analysis.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import yaml
import json
from pathlib import Path


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        path: Path to config file (.yaml or .json)
    
    Returns:
        Configuration dictionary, or empty dict if path is None
    
    Example config structure:
    ```yaml
    agents:
      - id: 0
        role: "SCOUT"
        node: 0
        sweep_mode: "right"
        personal_priority: 4
      - id: 1
        role: "SECURER"
        node: 0
        sweep_mode: "right"
        personal_priority: 3
    
    weights:
      room_priority: 1.0
      sector_priority: 1.0
      distance: 0.5
      hazard_penalty: 2.0
    
    known_nodes: [0, 1, 2]  # initially known corridor/entry nodes
    
    simulation:
      tmax: 1200
      record_interval: 5.0
      seed: 42
    ```
    """
    if path is None:
        return {}
    
    path_obj = Path(path)
    
    if not path_obj.exists():
        print(f"[Utils] Config file not found: {path}")
        return {}
    
    with open(path_obj, 'r') as f:
        if path_obj.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif path_obj.suffix == '.json':
            config = json.load(f)
        else:
            print(f"[Utils] Unsupported config format: {path_obj.suffix}")
            return {}
    
    return config if config else {}


def save_results(data: Dict[str, Any], output_path: str) -> None:
    """
    Save simulation results to file.
    
    Args:
        data: Dictionary of results to save
        output_path: Path to output file (.yaml or .json)
    """
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path_obj, 'w') as f:
        if path_obj.suffix in ['.yaml', '.yml']:
            yaml.dump(data, f, default_flow_style=False)
        elif path_obj.suffix == '.json':
            json.dump(data, f, indent=2)
        else:
            print(f"[Utils] Unsupported output format: {path_obj.suffix}")


def analyze_agent_performance(agent) -> Dict[str, Any]:
    """
    Analyze performance metrics for a single agent.
    
    Args:
        agent: Agent object
    
    Returns:
        Dictionary of performance metrics
    """
    return {
        "agent_id": agent.id,
        "role": agent.role.name,
        "rooms_cleared": agent.rooms_cleared,
        "evacuees_assisted": agent.evacuees_assisted,
        "distance_traveled": round(agent.distance_traveled, 2),
        "nodes_visited": len(agent.visited_nodes),
        "tasks_completed": len([log for log in agent.log if "Cleared" in log or "Assisted" in log]),
        "final_status": agent.status.name,
    }


def analyze_simulation_results(world) -> Dict[str, Any]:
    """
    Analyze overall simulation results.
    
    Args:
        world: World object after simulation
    
    Returns:
        Dictionary of analysis results
    """
    cleared, total = world.G.get_cleared_count()
    
    agent_stats = [analyze_agent_performance(a) for a in world.agents]
    
    total_distance = sum(a.distance_traveled for a in world.agents)
    total_rooms = sum(a.rooms_cleared for a in world.agents)
    total_evacuees = sum(a.evacuees_assisted for a in world.agents)
    
    return {
        "simulation_time": round(world.time, 2),
        "total_rooms": total,
        "rooms_cleared": cleared,
        "clearance_rate": round(cleared / total * 100, 2) if total > 0 else 0,
        "total_distance_traveled": round(total_distance, 2),
        "total_rooms_cleared": total_rooms,
        "total_evacuees_assisted": total_evacuees,
        "agent_performance": agent_stats,
        "efficiency_metric": round(cleared / (world.time / 60.0), 2) if world.time > 0 else 0,  # rooms/minute
    }


def generate_summary_report(world) -> str:
    """
    Generate a human-readable summary report of the simulation.
    
    Args:
        world: World object after simulation
    
    Returns:
        Formatted string report
    """
    analysis = analyze_simulation_results(world)
    
    report = []
    report.append("=" * 60)
    report.append("EMERGENCY EVACUATION SWEEP - SIMULATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    report.append(f"Building Type: {world.G.building_type}")
    report.append(f"Total Simulation Time: {analysis['simulation_time']:.1f} seconds ({analysis['simulation_time']/60:.1f} minutes)")
    report.append("")
    
    report.append("CLEARANCE RESULTS:")
    report.append(f"  Rooms Cleared: {analysis['rooms_cleared']} / {analysis['total_rooms']} ({analysis['clearance_rate']:.1f}%)")
    report.append(f"  Evacuees Assisted: {analysis['total_evacuees_assisted']}")
    report.append(f"  Efficiency: {analysis['efficiency_metric']:.2f} rooms/minute")
    report.append("")
    
    report.append("RESPONDER PERFORMANCE:")
    for agent_stat in analysis['agent_performance']:
        report.append(f"  Agent {agent_stat['agent_id']} ({agent_stat['role']}):")
        report.append(f"    - Rooms Cleared: {agent_stat['rooms_cleared']}")
        report.append(f"    - Evacuees Assisted: {agent_stat['evacuees_assisted']}")
        report.append(f"    - Distance Traveled: {agent_stat['distance_traveled']:.1f}m")
        report.append(f"    - Nodes Visited: {agent_stat['nodes_visited']}")
        report.append(f"    - Final Status: {agent_stat['final_status']}")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def validate_map(graph) -> list:
    """
    Validate a building map for common issues.
    
    Args:
        graph: Graph object to validate
    
    Returns:
        List of validation warnings/errors
    """
    issues = []
    
    # Check for exits
    if not graph.exits:
        issues.append("ERROR: No exit nodes defined")
    
    # Check for isolated nodes
    for node_id, node in graph.nodes.items():
        if not graph.neighbors(node_id):
            issues.append(f"WARNING: Node {node_id} ({node.name}) has no connections")
    
    # Check for bidirectional edges
    for (src, dst), edge in graph.edges.items():
        reverse = graph.get_edge(dst, src)
        if not reverse:
            issues.append(f"WARNING: Edge {src}->{dst} is not bidirectional")
    
    # Check for unreachable rooms from exits
    if graph.exits:
        from collections import deque
        visited = set()
        queue = deque([graph.exits[0]])
        visited.add(graph.exits[0])
        
        while queue:
            current = queue.popleft()
            for neighbor in graph.neighbors(current):
                edge = graph.get_edge(current, neighbor)
                if neighbor not in visited and edge and edge.traversable:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        unreachable = set(graph.nodes.keys()) - visited
        if unreachable:
            issues.append(f"WARNING: {len(unreachable)} nodes unreachable from exits: {list(unreachable)[:5]}")
    
    return issues


def calculate_theoretical_minimum_time(graph) -> float:
    """
    Calculate theoretical minimum time to clear all rooms.
    
    This is a lower bound assuming perfect coordination and no delays.
    
    Args:
        graph: Graph object
    
    Returns:
        Estimated minimum time in seconds
    """
    from .graph_model import NodeType
    
    # Count rooms that need clearing
    rooms = [n for n in graph.nodes.values() 
             if n.node_type not in [NodeType.CORRIDOR, NodeType.EXIT, NodeType.CHECKPOINT]]
    
    if not rooms:
        return 0.0
    
    # Sum of search times
    total_search_time = sum(r.get_effective_search_time() for r in rooms)
    
    # Estimate average travel between rooms (heuristic)
    avg_room_distance = 10.0  # meters
    avg_travel_time = avg_room_distance / 1.5  # 1.5 m/s walking speed
    total_travel_time = len(rooms) * avg_travel_time
    
    return total_search_time + total_travel_time


def create_building_summary(graph) -> Dict[str, Any]:
    """
    Create a summary of building characteristics.
    
    Args:
        graph: Graph object
    
    Returns:
        Dictionary with building statistics
    """
    from .graph_model import NodeType, HazardType
    
    node_types = {}
    for node in graph.nodes.values():
        node_types[node.node_type.name] = node_types.get(node.node_type.name, 0) + 1
    
    hazard_counts = {}
    for node in graph.nodes.values():
        if node.hazard != HazardType.NONE:
            hazard_counts[node.hazard.name] = hazard_counts.get(node.hazard.name, 0) + 1
    
    total_evacuees = sum(len(node.evacuees) for node in graph.nodes.values())
    
    return {
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges) // 2,  # bidirectional
        "exits": len(graph.exits),
        "node_types": node_types,
        "hazards": hazard_counts,
        "total_evacuees": total_evacuees,
        "building_type": graph.building_type,
        "num_floors": graph.num_floors,
    }

