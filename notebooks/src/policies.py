"""
policies.py: Role-based decision policies for responder agents.

Each responder type (Scout, Securer, Checkpointer, Evacuator) has a unique
behavior policy that determines their actions during the sweep.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List
import random

if TYPE_CHECKING:
    from .world import World
    from .agents import Agent

from .agents import Role, Status, Action, SweepMode
from .graph_model import NodeType, HazardType


def tick_policy(world: World, agent: Agent) -> None:
    """
    Main policy tick for an agent - delegates to role-specific policy.
    
    This is the entry point called by the scheduler for each agent.
    """
    if not agent.is_active:
        # Agent is not active, don't schedule next tick
        return
    
    # Get role-specific policy
    policy = world.policies.get(agent.role)
    if policy:
        policy(world, agent)
    
    # Schedule next tick (default 1 second intervals)
    world.schedule(1.0, tick_policy, world, agent)


def scout_policy(world: World, agent: Agent) -> None:
    """
    Scout policy: Fast exploration, room status checking, evacuee tagging.
    
    Scouts prioritize:
    1. Exploring unknown areas
    2. Checking room status quickly
    3. Tagging evacuees for assistance
    4. Signaling priorities to other responders
    """
    if agent.is_busy(world.time):
        return
    agent.set_action(Action.SCOUTING)
    
    # Update fog of war from current position
    world.fog.update_from_agent(agent)
    
    # If currently at a room, do quick scout
    current_node = world.G.get_node(agent.node)
    if current_node:
        if current_node.node_type not in [NodeType.CORRIDOR, NodeType.EXIT, NodeType.CHECKPOINT]:
            if not current_node.cleared:
                # Quick scout: tag evacuees, assess hazards, signal
                if current_node.evacuees:
                    for evac in current_node.evacuees:
                        if not evac.tagged:
                            evac.tagged = True
                            agent.add_signal(agent.node, "evacuees")
                
                if current_node.hazard != HazardType.NONE and current_node.hazard_severity > 0.15:
                    agent.add_signal(agent.node, "hazard")
                
                if current_node.room_priority <= 2:
                    agent.add_signal(agent.node, "priority")
                
                # Scouts do light marking (not full clear)
                agent.log_action(f"Scouted node {agent.node}")
    
    # Decide next move: explore unknowns or move to high-priority areas
    next_target = _find_scout_target(world, agent)
    
    if next_target is not None:
        # Move towards target
        _move_agent_to(world, agent, next_target)
    else:
        # No target found, explore any unknown areas
        _explore_random(world, agent)


def securer_policy(world: World, agent: Agent) -> None:
    """
    Securer policy: Secure hazards, assist evacuees, clear rooms thoroughly.
    
    Securers prioritize:
    1. Responding to scout signals (hazards, evacuees)
    2. Securing hazardous areas
    3. Assisting evacuees
    4. Full room clearance
    """
    if agent.is_busy(world.time):
        return
    # Check if at a room that needs securing
    current_node = world.G.get_node(agent.node)
    if current_node:
        if current_node.node_type not in [NodeType.CORRIDOR, NodeType.EXIT, NodeType.CHECKPOINT]:
            if not current_node.cleared:
                # Perform securing actions
                agent.set_action(Action.SECURING)
                
                # Handle hazards
                if current_node.hazard != HazardType.NONE and current_node.hazard_severity > 0.15:
                    secure_time = _secure_hazard(world, agent, current_node)
                    agent.set_busy_until(world.time + secure_time)
                    world.schedule(secure_time, _finish_securing, world, agent, agent.node)
                    return
                
                # Assist evacuees
                for evac in current_node.evacuees:
                    if evac.needs_assistance and not evac.evacuating:
                        agent.set_action(Action.ASSISTING)
                        assist_time = 5.0  # base assist time
                        evac.evacuating = True
                        agent.assist_evacuee(evac.id)
                        agent.set_busy_until(world.time + assist_time)
                        world.schedule(assist_time, _finish_assisting, world, agent, evac.id)
                        return
                
                # Clear the room
                clear_time = current_node.get_effective_search_time(agent.get_effective_speed())
                agent.set_busy_until(world.time + clear_time)
                world.schedule(clear_time, _finish_clearing, world, agent, agent.node)
                return
    
    # Not at a room or room already cleared, find next target
    agent.set_action(Action.MOVING)
    next_target = _find_securer_target(world, agent)
    
    if next_target is not None:
        _move_agent_to(world, agent, next_target)
    else:
        # No target, explore or wait
        _explore_random(world, agent)


def checkpointer_policy(world: World, agent: Agent) -> None:
    """
    Checkpointer policy: Maintain checkpoints, assist as needed.
    
    Checkpointers:
    1. Position at strategic checkpoints
    2. Monitor cleared areas
    3. Assist other responders on request
    4. Ensure no backflow of evacuees
    """
    if agent.is_busy(world.time):
        return
    agent.set_action(Action.CHECKPOINTING)
    
    # Update fog from position
    world.fog.update_from_agent(agent)
    
    current_node = world.G.get_node(agent.node)
    
    # If not at a checkpoint, find one
    if current_node and current_node.node_type != NodeType.CHECKPOINT:
        checkpoint = _find_checkpoint(world, agent)
        if checkpoint is not None:
            agent.set_action(Action.MOVING)
            _move_agent_to(world, agent, checkpoint)
            return
    
    # At checkpoint, monitor and assist if needed
    # Check for nearby evacuees needing help
    for neighbor_id in world.G.neighbors(agent.node):
        neighbor_node = world.G.get_node(neighbor_id)
        if neighbor_node:
            for evac in neighbor_node.evacuees:
                if evac.needs_assistance and not evac.evacuating:
                    # Move to assist
                    agent.set_action(Action.MOVING)
                    _move_agent_to(world, agent, neighbor_id)
                    return
    
    # Otherwise, stay at checkpoint
    agent.log_action(f"Holding checkpoint at node {agent.node}")


def evacuator_policy(world: World, agent: Agent) -> None:
    """
    Evacuator policy: Double-check clearance, final sweep, close areas.
    
    Evacuators:
    1. Wait for scouts/securers to do initial sweep
    2. Re-check rooms marked as cleared
    3. Perform final verification
    4. Close off cleared areas
    """
    if agent.is_busy(world.time):
        return
    agent.set_action(Action.CHECKING)
    
    # Update fog
    world.fog.update_from_agent(agent)
    
    current_node = world.G.get_node(agent.node)
    if current_node:
        # If at a cleared room, double-check it
        if current_node.cleared and current_node.node_type not in [NodeType.CORRIDOR, NodeType.EXIT]:
            if getattr(current_node, 'verified_by', None) == agent.id:
                pass
            else:
                # Perform verification check
                check_time = current_node.get_effective_search_time(agent.get_effective_speed()) * 0.7
                agent.set_busy_until(world.time + check_time)
                world.schedule(check_time, _finish_checking, world, agent, agent.node)
                agent.log_action(f"Double-checking node {agent.node}")
                return
    
    # Find next cleared room to verify
    next_target = _find_evacuator_target(world, agent)
    
    if next_target is not None:
        agent.set_action(Action.MOVING)
        _move_agent_to(world, agent, next_target)
    else:
        # All rooms checked or no rooms to check yet
        # Move towards uncleared areas to follow the sweep
        uncleared = world.find_nearest_uncleared_room(agent.node)
        if uncleared:
            agent.set_action(Action.MOVING)
            _move_agent_to(world, agent, uncleared)
        else:
            # All done, head to exit
            if world.G.exits:
                exit_node = world.G.exits[0]
                agent.set_action(Action.MOVING)
                _move_agent_to(world, agent, exit_node)


# ==================== Helper Functions ====================

def _find_scout_target(world: World, agent: Agent) -> Optional[int]:
    """Find next target for scout (prioritize unexplored areas)."""
    # Get known but not fully explored nodes
    known_nodes = world.fog.get_known_nodes()
    unexplored = [
        nid for nid in known_nodes
        if not world.fog.is_fully_known(nid)
    ]
    
    if unexplored:
        # Pick closest unexplored
        unexplored.sort(key=lambda n: world.G.distance(agent.node, n))
        return unexplored[0]
    
    # All known nodes explored, find uncleared rooms
    uncleared = world.find_nearest_uncleared_room(agent.node)
    if uncleared:
        return uncleared
    
    # Use sweep mode to explore systematically
    neighbors = world.G.neighbors(agent.node)
    if agent.sweep_mode == "right":
        ordered = SweepMode.right_hand_order(agent.node, neighbors)
    elif agent.sweep_mode == "left":
        ordered = SweepMode.left_hand_order(agent.node, neighbors)
    else:  # corridor
        ordered = SweepMode.corridor_priority(neighbors, world.G, agent.node)
    
    # Pick first unvisited or least visited
    for n in ordered:
        if n not in agent.visited_nodes:
            return n
    
    # All visited, pick first
    return ordered[0] if ordered else None


def _find_securer_target(world: World, agent: Agent) -> Optional[int]:
    """Find next target for securer (prioritize signaled rooms and uncleared)."""
    # Check for signals from scouts
    for other in world.agents:
        if other.role == Role.SCOUT and other.signals:
            # Get most recent signal
            signal_node, signal_type = other.signals[-1]
            if signal_type in ["evacuees", "hazard", "priority"]:
                node = world.G.get_node(signal_node)
                if node and not node.cleared:
                    return signal_node
    
    # Find nearest uncleared room
    return world.find_nearest_uncleared_room(agent.node)


def _find_evacuator_target(world: World, agent: Agent) -> Optional[int]:
    """Find next target for evacuator (cleared but not double-checked rooms)."""
    # Find cleared rooms that haven't been double-checked by this agent
    cleared_rooms = [
        nid for nid, node in world.G.nodes.items()
        if node.cleared
        and node.cleared_by != agent.id
        and node.node_type not in [NodeType.CORRIDOR, NodeType.EXIT, NodeType.CHECKPOINT]
        and nid not in agent.visited_nodes
    ]
    
    if cleared_rooms:
        # Sort by distance
        cleared_rooms.sort(key=lambda n: world.G.distance(agent.node, n))
        return cleared_rooms[0]
    
    return None


def _find_checkpoint(world: World, agent: Agent) -> Optional[int]:
    """Find nearest checkpoint node."""
    checkpoints = [
        nid for nid, node in world.G.nodes.items()
        if node.node_type == NodeType.CHECKPOINT
    ]
    
    if checkpoints:
        checkpoints.sort(key=lambda n: world.G.distance(agent.node, n))
        return checkpoints[0]
    
    # No explicit checkpoints, use corridor intersections
    corridors = [
        nid for nid, node in world.G.nodes.items()
        if node.node_type == NodeType.CORRIDOR
        and len(world.G.neighbors(nid)) >= 3  # intersection
    ]
    
    if corridors:
        corridors.sort(key=lambda n: world.G.distance(agent.node, n))
        return corridors[0]
    
    return None


def _move_agent_to(world: World, agent: Agent, target: int) -> None:
    """
    Move agent one step towards target along shortest path.
    
    Schedules arrival at next node.
    """
    if agent.node == target:
        return

    if agent.is_busy(world.time):
        return

    # Check if agent can move
    if not agent.can_move:
        agent.log_action(f"Cannot move (status: {agent.status.name})")
        return
    
    # Find path using HASO if available, otherwise fallback to basic
    try:
        path = world.shortest_path_haso(agent.node, target)
    except:
        path = world.shortest_path_known(agent.node, target)
    
    if path and len(path) >= 2:
        next_node = path[1]  # path[0] is current node
        agent.target_node = target
        agent.path = path
        
        # Calculate travel time
        edge = world.G.get_edge(agent.node, next_node)
        if edge and edge.traversable:
            hazard_mod = world.get_hazard_modifier(next_node)
            travel_time = edge.get_traversal_time(
                base_speed=agent.get_effective_speed(),
                hazard_modifier=hazard_mod
            )
            
            # Don't schedule if travel time is infinite
            if travel_time == float('inf'):
                agent.log_action(f"Cannot reach {next_node} (infinite travel time)")
                return
            
            # Track distance
            agent.distance_traveled += edge.length
            
            # Schedule arrival and register movement for visualization
            agent.set_busy_until(world.time + travel_time)
            world.schedule(travel_time, _arrive_at_node, world, agent, next_node)
            world.register_agent_movement(agent.id, agent.node, next_node, travel_time)
            agent.log_action(f"Moving to node {next_node} (ETA: {travel_time:.1f}s)")
        elif edge and (edge.can_open_edge() or edge.can_break_edge()):
            if edge.can_open_edge():
                prep_time = edge.get_open_time()
                action_type = "open"
                action_desc = "Opening"
            else:
                prep_time = edge.get_break_time()
                action_type = "break"
                action_desc = "Breaching"
            agent.set_busy_until(world.time + prep_time)
            agent.log_action(f"{action_desc} edge to node {next_node} (ETA: {prep_time:.1f}s)")
            world.schedule(
                prep_time,
                _complete_edge_preparation,
                world,
                agent,
                edge.src,
                edge.dst,
                target,
                action_type
            )
        else:
            agent.log_action(f"Cannot traverse edge to {next_node}")
    else:
        agent.log_action(f"No path to target {target}")


def _explore_random(world: World, agent: Agent) -> None:
    """Explore random neighbor when no specific target."""
    neighbors = world.G.neighbors(agent.node)
    known_neighbors = [n for n in neighbors if world.fog.is_known(n)]
    
    if known_neighbors:
        target = random.choice(known_neighbors)
        _move_agent_to(world, agent, target)
    else:
        agent.log_action("No known neighbors to explore")


def _secure_hazard(world: World, agent: Agent, node) -> float:
    """Calculate time to secure hazard at node."""
    base_time = 10.0
    
    if node.hazard == HazardType.FIRE:
        return base_time * 3.0 * node.hazard_severity
    elif node.hazard == HazardType.SMOKE:
        return base_time * 1.5 * node.hazard_severity
    elif node.hazard == HazardType.CHEMICAL:
        return base_time * 2.0 * node.hazard_severity
    else:
        return base_time * node.hazard_severity


# ==================== Event Completion Callbacks ====================

def _arrive_at_node(world: World, agent: Agent, node_id: int) -> None:
    """Callback when agent arrives at a node."""
    world.finish_agent_movement(agent.id)
    agent.clear_busy(world.time)
    agent.move_to_node(node_id)
    world.fog.update_from_agent(agent)


def _finish_clearing(world: World, agent: Agent, node_id: int) -> None:
    """Callback when agent finishes clearing a room."""
    world.mark_cleared(node_id, agent.id)
    agent.clear_room(node_id)
    agent.log_action(f"Cleared node {node_id}")
    agent.clear_busy(world.time)
    agent.set_action(Action.MOVING)


def _finish_checking(world: World, agent: Agent, node_id: int) -> None:
    """Callback when evacuator finishes double-checking a room."""
    node = world.G.get_node(node_id)
    if node:
        node.verified_by = agent.id
    agent.log_action(f"Double-checked node {node_id} - verified clear")
    agent.clear_busy(world.time)
    agent.set_action(Action.MOVING)


def _finish_securing(world: World, agent: Agent, node_id: int) -> None:
    """Callback when securer finishes securing a hazard."""
    node = world.G.get_node(node_id)
    if node:
        # Reduce hazard severity significantly each pass
        node.hazard_severity *= 0.3
        if node.hazard_severity < 0.1:
            node.hazard_severity = 0.0
            node.hazard = HazardType.NONE
            agent.log_action(f"Neutralized hazard at node {node_id}")
        else:
            agent.log_action(f"Secured hazard at node {node_id}")
    agent.clear_busy(world.time)
    agent.set_action(Action.MOVING)


def _finish_assisting(world: World, agent: Agent, evacuee_id: int) -> None:
    """Callback when agent finishes assisting an evacuee."""
    agent.log_action(f"Finished assisting evacuee {evacuee_id}")
    agent.clear_busy(world.time)
    agent.set_action(Action.MOVING)


def _complete_edge_preparation(
    world: World,
    agent: Agent,
    src: int,
    dst: int,
    target: int,
    action_type: str,
) -> None:
    """Helper to open or breach an edge before resuming movement."""
    edge = world.G.get_edge(src, dst)
    if edge:
        if action_type == "open":
            edge.open_edge(agent.id, world.time)
            agent.log_action(f"Opened edge to node {dst}")
        else:
            edge.break_edge(agent.id, world.time)
            agent.log_action(f"Breached edge to node {dst}")
    agent.clear_busy(world.time)
    _move_agent_to(world, agent, target)


def make_default_policies(world: World) -> dict:
    """
    Create default policy mappings for all roles.
    
    Returns:
        Dict mapping Role -> policy function
    """
    return {
        Role.SCOUT: scout_policy,
        Role.SECURER: securer_policy,
        Role.CHECKPOINTER: checkpointer_policy,
        Role.EVACUATOR: evacuator_policy,
    }

