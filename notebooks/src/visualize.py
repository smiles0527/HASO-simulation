"""
visualize.py: Visualization and animation tools for evacuation simulation.

Provides functions to create static plots and animated GIFs of the simulation.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

if TYPE_CHECKING:
    from .world import World
    from .graph_model import Graph

from .graph_model import NodeType, HazardType
from .agents import Role, Status


# Color schemes
ROLE_COLORS = {
    Role.SCOUT: '#4CAF50',        # Green
    Role.SECURER: '#2196F3',      # Blue
    Role.CHECKPOINTER: '#FF9800', # Orange
    Role.EVACUATOR: '#9C27B0',    # Purple
}

STATUS_COLORS = {
    Status.NORMAL: '#4CAF50',
    Status.SLOWED: '#FFEB3B',
    Status.PROGRESSING: '#FF9800',
    Status.IMMOBILIZED: '#F44336',
    Status.INCAPACITATED: '#9E9E9E',
    Status.DEAD: '#000000',
    Status.EXITED: '#E0E0E0',
}

NODE_TYPE_SHAPES = {
    NodeType.CORRIDOR: 'o',
    NodeType.STAIRCASE: '^',
    NodeType.SMALL_ROOM: 's',
    NodeType.CONNECTED_SMALL_ROOM: 's',
    NodeType.CONNECTED_LARGE_ROOM: 'D',
    NodeType.LARGE_CENTRAL_ROOM: 'D',
    NodeType.LARGE_SIDE_ROOM: 'D',
    NodeType.EXIT: '*',
    NodeType.CHECKPOINT: 'p',
}


def plot_building_layout(graph: Graph, ax=None, show_labels: bool = True) -> Any:
    """
    Plot static building layout as a graph.
    
    Args:
        graph: Graph object representing the building
        ax: Matplotlib axes (creates new if None)
        show_labels: Whether to show node labels
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw edges
    for (src, dst), edge in graph.edges.items():
        # Only draw each edge once (src < dst)
        if src < dst:
            node_src = graph.get_node(src)
            node_dst = graph.get_node(dst)
            
            if node_src and node_dst:
                # Edge style based on type
                linestyle = '-' if edge.traversable else ':'
                linewidth = 1.0
                color = '#CCCCCC'
                
                if not edge.gives_vision:
                    linestyle = '--'
                    color = '#999999'
                
                ax.plot([node_src.x, node_dst.x], 
                       [node_src.y, node_dst.y],
                       linestyle=linestyle, 
                       linewidth=linewidth,
                       color=color,
                       zorder=1)
    
    # Draw nodes
    for node_id, node in graph.nodes.items():
        # Node color based on type and state
        if node.node_type == NodeType.EXIT:
            color = '#4CAF50'
            size = 300
        elif node.cleared:
            color = '#81C784'
            size = 150
        elif node.hazard != HazardType.NONE:
            color = '#F44336'
            size = 150
        else:
            color = '#E0E0E0'
            size = 100
        
        # Node shape
        marker = NODE_TYPE_SHAPES.get(node.node_type, 'o')
        
        ax.scatter(node.x, node.y, 
                  c=color, 
                  s=size, 
                  marker=marker,
                  edgecolors='black',
                  linewidths=1.5,
                  zorder=2)
        
        # Labels
        if show_labels:
            label = f"{node_id}"
            if node.name and node.name != f"Node_{node_id}":
                label = f"{node_id}\n{node.name}"
            
            ax.text(node.x, node.y - 1.5, label,
                   ha='center', va='top',
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', 
                           edgecolor='none',
                           alpha=0.7))
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Building Layout')
    
    return ax


def plot_fog_of_war(graph: Graph, fog, ax=None) -> Any:
    """
    Visualize fog of war state across the building.
    
    Args:
        graph: Graph object
        fog: FogOfWar object
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw edges lightly
    for (src, dst), edge in graph.edges.items():
        if src < dst:
            node_src = graph.get_node(src)
            node_dst = graph.get_node(dst)
            if node_src and node_dst:
                ax.plot([node_src.x, node_dst.x], 
                       [node_src.y, node_dst.y],
                       linestyle='-', 
                       linewidth=0.5,
                       color='#CCCCCC',
                       zorder=1)
    
    # Color nodes by fog state
    fog_colors = {
        0: '#424242',  # Unknown-unknown (dark gray)
        1: '#FFA726',  # Known-unknown (orange)
        2: '#42A5F5',  # Unknown-known (blue)
        3: '#66BB6A',  # Known-known (green)
    }
    
    for node_id, node in graph.nodes.items():
        fog_state = fog.fog_state.get(node_id, 0)
        color = fog_colors.get(fog_state, '#000000')
        
        ax.scatter(node.x, node.y,
                  c=color,
                  s=150,
                  marker='o',
                  edgecolors='black',
                  linewidths=1,
                  zorder=2)
        
        ax.text(node.x, node.y, str(node_id),
               ha='center', va='center',
               fontsize=8,
               color='white',
               weight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=fog_colors[0], label='Unknown-unknown'),
        Patch(facecolor=fog_colors[1], label='Known-unknown'),
        Patch(facecolor=fog_colors[2], label='Unknown-known'),
        Patch(facecolor=fog_colors[3], label='Known-known'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Fog of War State')
    
    return ax


def plot_clearance_progress(world: World, ax=None) -> Any:
    """
    Plot clearance progress over time.
    
    Args:
        world: World object with history
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if not world.history:
        ax.text(0.5, 0.5, 'No history data available',
               ha='center', va='center',
               transform=ax.transAxes)
        return ax
    
    times = [h['time'] for h in world.history]
    cleared_counts = [len(h['cleared_nodes']) for h in world.history]
    fog_known = [h['fog_known'] for h in world.history]
    
    ax.plot(times, cleared_counts, 
           label='Rooms Cleared', 
           linewidth=2, 
           marker='o',
           markersize=4)
    
    ax.plot(times, fog_known, 
           label='Nodes Discovered', 
           linewidth=2, 
           marker='s',
           markersize=4,
           alpha=0.7)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Count')
    ax.set_title('Clearance Progress Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_agent_paths(world: World, ax=None) -> Any:
    """
    Plot paths taken by each agent.
    
    Args:
        world: World object
        ax: Matplotlib axes
    
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw building lightly
    plot_building_layout(world.G, ax=ax, show_labels=False)
    
    # Draw agent paths
    for agent in world.agents:
        visited = list(agent.visited_nodes)
        if len(visited) < 2:
            continue
        
        # Get coordinates
        xs = [world.G.get_node(n).x for n in visited if world.G.get_node(n)]
        ys = [world.G.get_node(n).y for n in visited if world.G.get_node(n)]
        
        # Plot path
        color = ROLE_COLORS.get(agent.role, '#000000')
        ax.plot(xs, ys,
               linewidth=2,
               alpha=0.6,
               color=color,
               label=f"Agent {agent.id} ({agent.role.name})")
        
        # Mark start and end
        if xs:
            ax.scatter(xs[0], ys[0], 
                      c=color, 
                      s=200, 
                      marker='o',
                      edgecolors='black',
                      linewidths=2,
                      zorder=10)
            ax.scatter(xs[-1], ys[-1], 
                      c=color, 
                      s=200, 
                      marker='X',
                      edgecolors='black',
                      linewidths=2,
                      zorder=10)
    
    ax.legend()
    ax.set_title('Agent Movement Paths')
    
    return ax


def animate_run(world: World, history: Optional[List[Dict]], out_path: Optional[str] = None) -> None:
    """
    Create an animated visualization of the simulation.
    
    Args:
        world: World object
        history: History data (list of state snapshots)
        out_path: Path to save animation (if None, displays instead)
    """
    if not history or len(history) < 2:
        print("[Visualize] Not enough history data for animation")
        return
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    def update(frame_idx):
        ax.clear()
        
        frame = history[frame_idx]
        time = frame['time']
        
        # Draw building
        plot_building_layout(world.G, ax=ax, show_labels=True)
        
        # Highlight cleared nodes
        for node_id in frame['cleared_nodes']:
            node = world.G.get_node(node_id)
            if node:
                circle = plt.Circle((node.x, node.y), 1.5, 
                                   color='#81C784', 
                                   alpha=0.3,
                                   zorder=3)
                ax.add_patch(circle)
        
        # Draw agents
        for agent_data in frame['agents']:
            node = world.G.get_node(agent_data['node'])
            if node:
                role = Role[agent_data['role']]
                status = Status[agent_data['status']]
                
                color = ROLE_COLORS.get(role, '#000000')
                
                # Agent marker
                ax.scatter(node.x, node.y,
                          c=color,
                          s=400,
                          marker='o',
                          edgecolors='black',
                          linewidths=2,
                          zorder=5,
                          alpha=0.8)
                
                # Agent label
                ax.text(node.x, node.y,
                       f"A{agent_data['id']}",
                       ha='center', va='center',
                       fontsize=10,
                       color='white',
                       weight='bold',
                       zorder=6)
        
        ax.set_title(f"Evacuation Sweep - Time: {time:.1f}s\n"
                    f"Cleared: {len(frame['cleared_nodes'])} rooms | "
                    f"Discovered: {frame['fog_known']} nodes")
    
    anim = FuncAnimation(fig, update, frames=len(history), interval=200, repeat=True)
    
    if out_path:
        print(f"[Visualize] Saving animation to {out_path}...")
        writer = PillowWriter(fps=5)
        anim.save(out_path, writer=writer)
        print(f"[Visualize] Animation saved!")
    else:
        plt.show()
    
    plt.close()


def create_summary_dashboard(world: World, save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive dashboard summarizing the simulation.
    
    Args:
        world: World object after simulation
        save_path: Path to save figure (if None, displays instead)
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Building layout with clearance
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    plot_building_layout(world.G, ax=ax1, show_labels=True)
    ax1.set_title('Building Layout & Clearance Status', fontsize=14, weight='bold')
    
    # 2. Agent paths
    ax2 = fig.add_subplot(gs[0, 2])
    plot_agent_paths(world, ax=ax2)
    ax2.set_title('Agent Paths', fontsize=12, weight='bold')
    
    # 3. Clearance progress
    ax3 = fig.add_subplot(gs[1, 2])
    plot_clearance_progress(world, ax=ax3)
    
    # 4. Fog of war
    ax4 = fig.add_subplot(gs[2, 0])
    plot_fog_of_war(world.G, world.fog, ax=ax4)
    
    # 5. Agent performance bar chart
    ax5 = fig.add_subplot(gs[2, 1])
    agent_ids = [a.id for a in world.agents]
    rooms_cleared = [a.rooms_cleared for a in world.agents]
    colors = [ROLE_COLORS.get(a.role, '#000000') for a in world.agents]
    
    bars = ax5.bar(agent_ids, rooms_cleared, color=colors, edgecolor='black', linewidth=1.5)
    ax5.set_xlabel('Agent ID')
    ax5.set_ylabel('Rooms Cleared')
    ax5.set_title('Agent Performance', fontsize=12, weight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Statistics text
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    cleared, total = world.G.get_cleared_count()
    stats_text = f"""
SIMULATION SUMMARY
{'='*30}
Time: {world.time:.1f}s ({world.time/60:.1f} min)

Clearance: {cleared}/{total} rooms
Rate: {(cleared/total*100):.1f}%

Total Distance: {sum(a.distance_traveled for a in world.agents):.1f}m
Total Evacuees: {sum(a.evacuees_assisted for a in world.agents)}

Agents Active: {sum(1 for a in world.agents if a.is_active)}
"""
    
    ax6.text(0.1, 0.9, stats_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Emergency Evacuation Sweep - Simulation Dashboard',
                fontsize=16, weight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualize] Dashboard saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

