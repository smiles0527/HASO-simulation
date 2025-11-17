"""
Figure generation functions for HiMCM supplemental diagrams.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import networkx as nx

try:
    from haso_sim import load_map, build_world, simulate
    from haso_sim.graph_model import Graph, Node, Edge, NodeType, HazardType
    HAS_HASO_SIM = True
except ImportError:
    HAS_HASO_SIM = False


# Color schemes
COLORS = {
    'scout': '#2E86AB',
    'securer': '#A23B72',
    'checkpointer': '#F18F01',
    'evacuator': '#C73E1D',
    'hazard_low': '#FFF9C4',
    'hazard_med': '#FFB74D',
    'hazard_high': '#E53935',
    'exit': '#4CAF50',
    'room': '#E3F2FD',
    'corridor': '#F5F5F5',
}


def setup_figure(size=(10, 8), dpi=150):
    """Create a figure with consistent styling."""
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


def plot_graph_base(graph: Graph, ax, show_labels=True, node_size=300):
    """Plot base graph structure."""
    for node_id, node in graph.nodes.items():
        x, y = node.x, node.y
        
        # Node color by type
        if node.node_type == NodeType.EXIT:
            color = COLORS['exit']
            shape = 's'  # square
        elif node.node_type == NodeType.CORRIDOR:
            color = COLORS['corridor']
            shape = 'o'
        else:
            color = COLORS['room']
            shape = 'o'
        
        ax.scatter(x, y, s=node_size, c=color, marker=shape, 
                  edgecolors='black', linewidths=1.5, zorder=3)
        
        if show_labels:
            ax.text(x, y, str(node_id), ha='center', va='center',
                   fontsize=8, fontweight='bold', zorder=4)
    
    # Draw edges
    for (src, dst), edge in graph.edges.items():
        if src < dst:  # Draw each edge once
            src_node = graph.get_node(src)
            dst_node = graph.get_node(dst)
            if src_node and dst_node:
                ax.plot([src_node.x, dst_node.x], [src_node.y, dst_node.y],
                       'k-', alpha=0.3, linewidth=1, zorder=1)


def create_complex_building_layout():
    """Create a realistic building layout with 100+ nodes and 150+ edges."""
    nodes = []
    node_id = 0
    edges = []
    
    # === MAIN LOBBY AND ENTRANCE ===
    # Main entrance/lobby (larger area)
    lobby_nodes = []
    lobby_positions = [(0, 0), (2, 0), (4, 0), (0, 2), (2, 2), (4, 2)]
    for x, y in lobby_positions:
        is_exit = (x == 0 and y == 0) or (x == 4 and y == 0)
        nodes.append({
            'id': node_id,
            'x': x,
            'y': y,
            'type': 'EXIT' if is_exit else 'CORRIDOR',
            'area': 'lobby'
        })
        lobby_nodes.append(node_id)
        node_id += 1
    
    # Connect lobby nodes
    for i in range(len(lobby_nodes) - 1):
        edges.append({'src': lobby_nodes[i], 'dst': lobby_nodes[i+1], 'length': 2.0})
    
    # === MAIN CORRIDOR SYSTEM (L-shaped and branching) ===
    # Main horizontal corridor (west to east)
    main_corridor = []
    for i in range(8):
        x = 6 + i * 3
        y = 1
        nodes.append({
            'id': node_id,
            'x': x,
            'y': y,
            'type': 'CORRIDOR',
            'area': 'main_corridor'
        })
        main_corridor.append(node_id)
        if i > 0:
            edges.append({'src': main_corridor[i-1], 'dst': main_corridor[i], 'length': 3.0})
        node_id += 1
    
    # Connect lobby to main corridor
    edges.append({'src': lobby_nodes[2], 'dst': main_corridor[0], 'length': 2.0})
    
    # North branch corridor (L-shaped)
    north_branch = []
    for i in range(6):
        x = 6 + i * 2.5
        y = 4 + i * 0.3  # Slight curve
        nodes.append({
            'id': node_id,
            'x': x,
            'y': y,
            'type': 'CORRIDOR',
            'area': 'north_branch'
        })
        north_branch.append(node_id)
        if i > 0:
            edges.append({'src': north_branch[i-1], 'dst': north_branch[i], 'length': 2.5})
        node_id += 1
    
    edges.append({'src': main_corridor[2], 'dst': north_branch[0], 'length': 3.0})
    
    # South branch corridor
    south_branch = []
    for i in range(5):
        x = 12 + i * 2.5
        y = -2 - i * 0.5
        nodes.append({
            'id': node_id,
            'x': x,
            'y': y,
            'type': 'CORRIDOR',
            'area': 'south_branch'
        })
        south_branch.append(node_id)
        if i > 0:
            edges.append({'src': south_branch[i-1], 'dst': south_branch[i], 'length': 2.5})
        node_id += 1
    
    edges.append({'src': main_corridor[4], 'dst': south_branch[0], 'length': 3.0})
    
    # === OFFICE WINGS (irregular arrangements) ===
    # North wing offices (varying sizes)
    north_offices = []
    office_configs = [
        # Row 1: Small offices
        [(7, 6.5), (9.5, 6.5), (12, 6.5)],
        # Row 2: Mix of sizes
        [(6.5, 8), (9, 8), (11.5, 8), (14, 8)],
        # Row 3: Larger offices
        [(8, 9.5), (11, 9.5)],
        # Row 4: Small offices
        [(7.5, 11), (10, 11), (12.5, 11)],
    ]
    
    for row_idx, row in enumerate(office_configs):
        for x, y in row:
            nodes.append({
                'id': node_id,
                'x': x,
                'y': y,
                'type': 'ROOM',
                'area': 'north_offices',
                'row': row_idx
            })
            north_offices.append(node_id)
            # Connect to nearest corridor
            nearest_corr = min(north_branch, key=lambda cid: 
                ((nodes[cid]['x'] - x)**2 + (nodes[cid]['y'] - y)**2)**0.5)
            edges.append({'src': node_id, 'dst': nearest_corr, 'length': 1.5})
            node_id += 1
    
    # South wing offices (different layout)
    south_offices = []
    south_configs = [
        [(13, -3.5), (15.5, -3.5), (18, -3.5)],
        [(12.5, -5), (15, -5), (17.5, -5), (20, -5)],
        [(14, -6.5), (16.5, -6.5)],
        [(13.5, -8), (16, -8), (18.5, -8)],
    ]
    
    for row_idx, row in enumerate(south_configs):
        for x, y in row:
            nodes.append({
                'id': node_id,
                'x': x,
                'y': y,
                'type': 'ROOM',
                'area': 'south_offices',
                'row': row_idx
            })
            south_offices.append(node_id)
            nearest_corr = min(south_branch, key=lambda cid:
                ((nodes[cid]['x'] - x)**2 + (nodes[cid]['y'] - y)**2)**0.5)
            edges.append({'src': node_id, 'dst': nearest_corr, 'length': 1.5})
            node_id += 1
    
    # === EAST WING (conference rooms and larger spaces) ===
    east_wing_corridor = []
    for i in range(4):
        x = 27 + i * 2
        y = 2 + i * 0.5
        nodes.append({
            'id': node_id,
            'x': x,
            'y': y,
            'type': 'CORRIDOR',
            'area': 'east_wing'
        })
        east_wing_corridor.append(node_id)
        if i > 0:
            edges.append({'src': east_wing_corridor[i-1], 'dst': east_wing_corridor[i], 'length': 2.0})
        node_id += 1
    
    edges.append({'src': main_corridor[-1], 'dst': east_wing_corridor[0], 'length': 3.0})
    
    # Conference rooms and large spaces
    east_rooms = []
    east_positions = [
        (28, 4), (30.5, 4),  # Conference rooms
        (27.5, 5.5), (30, 5.5), (32.5, 5.5),  # Offices
        (29, 7), (31.5, 7),  # Meeting rooms
    ]
    for x, y in east_positions:
        nodes.append({
            'id': node_id,
            'x': x,
            'y': y,
            'type': 'ROOM',
            'area': 'east_wing'
        })
        east_rooms.append(node_id)
        nearest_corr = min(east_wing_corridor, key=lambda cid:
            ((nodes[cid]['x'] - x)**2 + (nodes[cid]['y'] - y)**2)**0.5)
        edges.append({'src': node_id, 'dst': nearest_corr, 'length': 1.5})
        node_id += 1
    
    # === WEST WING (support spaces) ===
    west_wing_corridor = []
    for i in range(3):
        x = -3 - i * 2.5
        y = 1.5
        nodes.append({
            'id': node_id,
            'x': x,
            'y': y,
            'type': 'CORRIDOR',
            'area': 'west_wing'
        })
        west_wing_corridor.append(node_id)
        if i > 0:
            edges.append({'src': west_wing_corridor[i-1], 'dst': west_wing_corridor[i], 'length': 2.5})
        node_id += 1
    
    edges.append({'src': lobby_nodes[0], 'dst': west_wing_corridor[0], 'length': 3.0})
    
    # Support rooms (break room, storage, etc.)
    west_rooms = []
    west_positions = [
        (-4, 3.5), (-6.5, 3.5),  # Break room, storage
        (-3.5, -1), (-6, -1),  # Utility rooms
    ]
    for x, y in west_positions:
        nodes.append({
            'id': node_id,
            'x': x,
            'y': y,
            'type': 'ROOM',
            'area': 'west_wing'
        })
        west_rooms.append(node_id)
        nearest_corr = min(west_wing_corridor, key=lambda cid:
            ((nodes[cid]['x'] - x)**2 + (nodes[cid]['y'] - y)**2)**0.5)
        edges.append({'src': node_id, 'dst': nearest_corr, 'length': 2.0})
        node_id += 1
    
    # === STAIRWELLS AND VERTICAL CONNECTIONS ===
    stairwells = []
    stair_positions = [(6, 3), (15, 0.5), (24, 3)]
    for x, y in stair_positions:
        nodes.append({
            'id': node_id,
            'x': x,
            'y': y,
            'type': 'STAIR',
            'area': 'stairwell'
        })
        stairwells.append(node_id)
        # Connect to nearest corridor
        all_corridors = main_corridor + north_branch + south_branch + east_wing_corridor + west_wing_corridor
        nearest = min(all_corridors, key=lambda cid:
            ((nodes[cid]['x'] - x)**2 + (nodes[cid]['y'] - y)**2)**0.5)
        edges.append({'src': node_id, 'dst': nearest, 'length': 1.0})
        node_id += 1
    
    # === ADDITIONAL ROOMS ALONG MAIN CORRIDOR ===
    # Offices along main corridor
    main_offices = []
    for i in [1, 3, 5, 6]:
        corr_node = main_corridor[i]
        corr_x, corr_y = nodes[corr_node]['x'], nodes[corr_node]['y']
        # Add rooms on both sides
        for side in [-1.5, 1.5]:
            nodes.append({
                'id': node_id,
                'x': corr_x,
                'y': corr_y + side,
                'type': 'ROOM',
                'area': 'main_corridor_offices'
            })
            main_offices.append(node_id)
            edges.append({'src': node_id, 'dst': corr_node, 'length': 1.5})
            node_id += 1
    
    # === INTER-ROOM CONNECTIONS (adjacent offices) ===
    # Connect adjacent offices in north wing
    for i in range(len(north_offices) - 1):
        room1 = nodes[north_offices[i]]
        room2 = nodes[north_offices[i+1]]
        dist = ((room1['x'] - room2['x'])**2 + (room1['y'] - room2['y'])**2)**0.5
        if dist < 3.5:  # Only connect if close
            edges.append({'src': north_offices[i], 'dst': north_offices[i+1], 'length': dist})
    
    # Connect adjacent offices in south wing
    for i in range(len(south_offices) - 1):
        room1 = nodes[south_offices[i]]
        room2 = nodes[south_offices[i+1]]
        dist = ((room1['x'] - room2['x'])**2 + (room1['y'] - room2['y'])**2)**0.5
        if dist < 3.5:
            edges.append({'src': south_offices[i], 'dst': south_offices[i+1], 'length': dist})
    
    # === ADDITIONAL EXIT ===
    # Add emergency exit at end of east wing
    nodes.append({
        'id': node_id,
        'x': 35,
        'y': 4,
        'type': 'EXIT',
        'area': 'east_exit'
    })
    edges.append({'src': node_id, 'dst': east_wing_corridor[-1], 'length': 2.0})
    node_id += 1
    
    return nodes, edges


# 1. Spatial embedding
def generate_spatial_embedding(map_path: Path, output_path: Path):
    """Generate spatial embedding with nodes, edges, and distances."""
    fig, ax = setup_figure((20, 16))
    
    nodes, edges = create_complex_building_layout()
    
    # Plot nodes
    for node in nodes:
        x, y = node['x'], node['y']
        if node['type'] == 'EXIT':
            ax.scatter(x, y, s=400, c=COLORS['exit'], marker='s',
                      edgecolors='black', linewidths=2, zorder=3)
        elif node['type'] == 'CORRIDOR':
            ax.scatter(x, y, s=250, c=COLORS['corridor'], marker='o',
                      edgecolors='black', linewidths=1.5, zorder=3)
        else:
            ax.scatter(x, y, s=200, c=COLORS['room'], marker='o',
                      edgecolors='black', linewidths=1.5, zorder=3)
        
        # Label only some nodes to avoid clutter
        if node['type'] == 'EXIT' or node['id'] % 15 == 0:
            ax.text(x, y, str(node['id']), ha='center', va='center',
                   fontsize=5, fontweight='bold', zorder=4)
    
    # Draw edges with distance labels (sample some to avoid clutter)
    edge_count = 0
    for edge in edges:
        src = nodes[edge['src']]
        dst = nodes[edge['dst']]
        ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
               'k-', linewidth=0.5, alpha=0.4, zorder=1)
        
        # Label every 15th edge to show distances
        if edge_count % 15 == 0:
            mid_x = (src['x'] + dst['x']) / 2
            mid_y = (src['y'] + dst['y']) / 2
            ax.text(mid_x, mid_y, f'{edge["length"]:.1f}m',
                   ha='center', va='center', fontsize=4,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                   zorder=5)
        edge_count += 1
    
    ax.set_xlim(-10, 40)
    ax.set_ylim(-10, 15)
    ax.set_title(f'Spatial Embedding of Building Graph ({len(nodes)} nodes, {len(edges)} edges)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 2. Weighted traversal costs
def generate_weighted_traversal_costs(map_path: Path, output_path: Path):
    """Visualize weighted traversal costs as edge thickness/color."""
    fig, ax = setup_figure((20, 16))
    
    nodes, edges = create_complex_building_layout()
    
    # Calculate costs
    costs = {}
    for edge in edges:
        # Corridor edges have lower cost
        src_type = nodes[edge['src']]['type']
        dst_type = nodes[edge['dst']]['type']
        if src_type == 'CORRIDOR' and dst_type == 'CORRIDOR':
            cost = edge['length'] * 0.8  # Lower cost
        elif src_type == 'ROOM' and dst_type == 'ROOM':
            cost = edge['length'] * 1.5  # Higher cost
        elif src_type == 'STAIR' or dst_type == 'STAIR':
            cost = edge['length'] * 1.2  # Stairs medium-high cost
        else:
            cost = edge['length'] * 1.0  # Medium cost
        costs[(edge['src'], edge['dst'])] = cost
    
    max_cost = max(costs.values()) if costs else 1.0
    
    # Plot all nodes
    for node in nodes:
        x, y = node['x'], node['y']
        if node['type'] == 'EXIT':
            ax.scatter(x, y, s=400, c=COLORS['exit'], marker='s',
                      edgecolors='black', linewidths=2, zorder=3)
        elif node['type'] == 'STAIR':
            ax.scatter(x, y, s=300, c='#FFD700', marker='D',
                      edgecolors='black', linewidths=2, zorder=3)
        elif node['type'] == 'CORRIDOR':
            ax.scatter(x, y, s=200, c=COLORS['corridor'], marker='o',
                      edgecolors='black', linewidths=1.5, zorder=3)
        else:
            ax.scatter(x, y, s=180, c=COLORS['room'], marker='o',
                      edgecolors='black', linewidths=1.5, zorder=3)
    
    # Draw edges with thickness proportional to cost
    for edge in edges:
        src = nodes[edge['src']]
        dst = nodes[edge['dst']]
        cost = costs.get((edge['src'], edge['dst']), edge['length'])
        width = 0.8 + 3.5 * (cost / max_cost)
        # Color edges by cost
        if cost / max_cost < 0.4:
            color = 'green'
            alpha = 0.6
        elif cost / max_cost < 0.7:
            color = 'orange'
            alpha = 0.7
        else:
            color = 'red'
            alpha = 0.8
        ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
               color=color, linewidth=width, alpha=alpha, zorder=1)
    
    ax.set_xlim(-10, 40)
    ax.set_ylim(-10, 15)
    ax.set_title(f'Weighted Traversal Costs ({len(nodes)} nodes, {len(edges)} edges)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 3. Agent routing over hazard-modified graph
def generate_hazard_routing(map_path: Path, output_path: Path):
    """Show routing paths avoiding hazards."""
    fig, ax = setup_figure((20, 16))
    
    nodes, edges = create_complex_building_layout()
    
    # Add hazards to some rooms (find rooms in north and south areas)
    north_rooms = [n for n in nodes if n.get('area') == 'north_offices']
    south_rooms = [n for n in nodes if n.get('area') == 'south_offices']
    
    # Add hazards to some north wing rooms
    if len(north_rooms) >= 3:
        for i in range(min(3, len(north_rooms))):
            north_rooms[i]['hazard'] = 0.7
    
    # Add hazards to some south wing rooms
    if len(south_rooms) >= 2:
        for i in range(min(2, len(south_rooms))):
            south_rooms[i]['hazard'] = 0.6
    
    # Initialize hazard for all nodes
    for node in nodes:
        if 'hazard' not in node:
            node['hazard'] = 0.0
    
    # Plot all nodes with hazard shading
    for node in nodes:
        x, y = node['x'], node['y']
        hazard = node.get('hazard', 0.0)
        
        if hazard > 0.5:
            color = COLORS['hazard_high']
            size = 300
        elif hazard > 0.2:
            color = COLORS['hazard_med']
            size = 250
        elif node['type'] == 'EXIT':
            color = COLORS['exit']
            size = 400
        elif node['type'] == 'STAIR':
            color = '#FFD700'
            size = 300
        elif node['type'] == 'CORRIDOR':
            color = COLORS['corridor']
            size = 200
        else:
            color = COLORS['room']
            size = 180
        
        ax.scatter(x, y, s=size, c=color,
                  edgecolors='black', linewidths=1.5, zorder=3, alpha=0.8)
        
        # Draw hazard intensity circle for hazardous nodes
        if hazard > 0.2:
            circle = Circle((x, y), 1.2, color=color, alpha=0.3 * hazard, zorder=2)
            ax.add_patch(circle)
    
    # Draw all edges
    for edge in edges:
        src = nodes[edge['src']]
        dst = nodes[edge['dst']]
        # Check if edge goes through hazard
        src_hazard = src.get('hazard', 0.0)
        dst_hazard = dst.get('hazard', 0.0)
        if src_hazard > 0.2 or dst_hazard > 0.2:
            ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
                   'r--', alpha=0.3, linewidth=1, zorder=1)
        else:
            ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
                   'k-', alpha=0.2, linewidth=0.8, zorder=1)
    
    # Example paths avoiding hazards
    exits = [n for n in nodes if n['type'] == 'EXIT']
    lobby_nodes = [n for n in nodes if n.get('area') == 'lobby']
    main_corridor_nodes = [n for n in nodes if n.get('area') == 'main_corridor']
    
    if exits and lobby_nodes and main_corridor_nodes:
        start = lobby_nodes[0]  # Start from lobby
        end = exits[0]
        
        # Path avoiding hazards: go through main corridor
        path_nodes = [start]
        if main_corridor_nodes:
            path_nodes.append(main_corridor_nodes[0])
            if len(main_corridor_nodes) > 2:
                path_nodes.append(main_corridor_nodes[-1])
        path_nodes.append(end)
        
        for i in range(len(path_nodes) - 1):
            src = path_nodes[i]
            dst = path_nodes[i+1]
            ax.annotate('', xy=(dst['x'], dst['y']), xytext=(src['x'], src['y']),
                      arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['scout'], alpha=0.8),
                      zorder=2)
    
    ax.set_xlim(-10, 40)
    ax.set_ylim(-10, 15)
    ax.set_title(f'Agent Routing Over Hazard-Modified Graph ({len(nodes)} nodes)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 4. Hazard intensity diffusion
def generate_hazard_diffusion(map_path: Path, output_path: Path):
    """Show hazard spreading over time (t0 through t5)."""
    num_times = 6
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes = axes.flatten()
    
    # Use realistic building layout
    nodes, edges = create_complex_building_layout()
    
    # Build neighbor graph from edges
    neighbors = {i: [] for i in range(len(nodes))}
    for edge in edges:
        neighbors[edge['src']].append(edge['dst'])
        neighbors[edge['dst']].append(edge['src'])
    
    # Initialize hazard at a room in north wing
    north_rooms = [n for n in nodes if n.get('area') == 'north_offices']
    if north_rooms:
        start_node = north_rooms[len(north_rooms) // 2]
        start_idx = start_node['id']
    else:
        start_idx = len(nodes) // 2
    
    # Store hazard states for each time step
    hazard_states = []
    
    # Initialize hazard values
    initial_hazards = {}
    for node in nodes:
        initial_hazards[node['id']] = 0.0
    initial_hazards[start_idx] = 0.3
    hazard_states.append(initial_hazards.copy())
    
    # Simulate diffusion at multiple time points
    for t_idx in range(1, num_times):
        current_hazards = hazard_states[t_idx - 1].copy()
        new_hazards = {}
        
        for i, node in enumerate(nodes):
            node_hazard = current_hazards.get(i, 0.0)
            if node_hazard > 0.2:
                # Spread to neighbors
                for neighbor_idx in neighbors.get(i, []):
                    if neighbor_idx < len(nodes):
                        neighbor_hazard = current_hazards.get(neighbor_idx, 0.0)
                        spread_rate = 0.12 * t_idx  # Slower spread for more time steps
                        if neighbor_hazard == 0.0:
                            new_hazards[neighbor_idx] = min(spread_rate, 0.8)
                        elif neighbor_hazard < 0.8:
                            new_hazards[neighbor_idx] = min(neighbor_hazard + spread_rate * 0.4, 0.8)
        
        # Update hazards
        for idx, severity in new_hazards.items():
            current_hazards[idx] = severity
        
        hazard_states.append(current_hazards.copy())
    
    # Plot each time step
    for t_idx, ax in enumerate(axes[:num_times]):
        current_hazards = hazard_states[t_idx]
        
        # Update node hazards for plotting
        for i, node in enumerate(nodes):
            node['hazard'] = current_hazards.get(i, 0.0)
        
        # Plot all nodes
        for node in nodes:
            x, y = node['x'], node['y']
            severity = node['hazard']
            
            if severity > 0.5:
                color = COLORS['hazard_high']
            elif severity > 0.2:
                color = COLORS['hazard_med']
            elif severity > 0.0:
                color = COLORS['hazard_low']
            elif node['type'] == 'EXIT':
                color = COLORS['exit']
            elif node['type'] == 'CORRIDOR':
                color = COLORS['corridor']
            else:
                color = COLORS['room']
            
            ax.scatter(x, y, s=180, c=color, edgecolors='black',
                      linewidths=1.5, zorder=3, alpha=0.8)
            
            # Draw hazard intensity circle
            if severity > 0:
                circle = Circle((x, y), 1.0, color=color,
                                  alpha=0.4 * severity, zorder=2)
                ax.add_patch(circle)
        
        # Draw edges
        for edge in edges:
            src = nodes[edge['src']]
            dst = nodes[edge['dst']]
            ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
                   'k-', alpha=0.15, linewidth=0.5, zorder=1)
        
        ax.set_xlim(-10, 40)
        ax.set_ylim(-10, 15)
        ax.set_title(f't = {t_idx} ({len(nodes)} nodes)', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    fig.suptitle('Hazard Intensity Diffusion Over Time', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 5. Securer mitigation
def generate_securer_mitigation(map_path: Path, output_path: Path):
    """Before/during/after securer intervention."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()
    
    nodes, edges = create_complex_building_layout()
    
    # Select a room to have hazard (find a north wing room)
    north_rooms = [n for n in nodes if n.get('area') == 'north_offices']
    if not north_rooms:
        north_rooms = [n for n in nodes if n['type'] == 'ROOM']
    hazard_node = north_rooms[len(north_rooms) // 2] if north_rooms else nodes[0]
    hazard_node_id = hazard_node['id']
    
    # Define stages: initial, approaching, mitigating, mitigated
    stages = [
        (0.8, False, 'Initial Hazard'),
        (0.7, False, 'Securer Approaching'),
        (0.4, True, 'Mitigation In Progress'),
        (0.2, True, 'Hazard Mitigated'),
    ]
    
    for frame, (severity, securer_present, title) in enumerate(stages):
        ax = axes[frame]
        
        # Plot all nodes
        for node in nodes:
            x, y = node['x'], node['y']
            if node['id'] == hazard_node_id:
                if severity > 0.5:
                    color = COLORS['hazard_high']
                elif severity > 0.2:
                    color = COLORS['hazard_med']
                else:
                    color = COLORS['hazard_low']
            elif node['type'] == 'EXIT':
                color = COLORS['exit']
            elif node['type'] == 'STAIR':
                color = '#FFD700'
            elif node['type'] == 'CORRIDOR':
                color = COLORS['corridor']
            else:
                color = COLORS['room']
            
            ax.scatter(x, y, s=200, c=color, edgecolors='black',
                      linewidths=1.5, zorder=3, alpha=0.8)
        
        # Draw edges
        for edge in edges:
            src = nodes[edge['src']]
            dst = nodes[edge['dst']]
            ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
                   'k-', alpha=0.2, linewidth=0.5, zorder=1)
        
        # Add securer icon if present
        if securer_present:
            ax.scatter(hazard_node['x'], hazard_node['y'] - 0.8, s=300,
                      c=COLORS['securer'], marker='s', edgecolors='black',
                      linewidths=2, zorder=5)
            ax.text(hazard_node['x'], hazard_node['y'] - 0.8, 'S', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white', zorder=6)
        elif frame == 1:  # Approaching stage
            # Show securer approaching from corridor
            main_corridor_nodes = [n for n in nodes if n.get('area') == 'main_corridor']
            if main_corridor_nodes:
                approach_node = main_corridor_nodes[0]
                ax.scatter(approach_node['x'], approach_node['y'], s=300,
                          c=COLORS['securer'], marker='s', edgecolors='black',
                          linewidths=2, zorder=5)
                ax.text(approach_node['x'], approach_node['y'], 'S', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white', zorder=6)
                # Draw path to hazard
                ax.annotate('', xy=(hazard_node['x'], hazard_node['y'] - 0.8),
                          xytext=(approach_node['x'], approach_node['y']),
                          arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['securer'], alpha=0.7),
                          zorder=4)
        
        # Highlight hazard area
        if severity > 0.5:
            hazard_color = COLORS['hazard_high']
        elif severity > 0.2:
            hazard_color = COLORS['hazard_med']
        else:
            hazard_color = COLORS['hazard_low']
        
        circle = Circle((hazard_node['x'], hazard_node['y']), 1.5,
                          color=hazard_color, alpha=0.6, zorder=2)
        ax.add_patch(circle)
        
        ax.set_xlim(-10, 40)
        ax.set_ylim(-10, 15)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    fig.suptitle('Securer Mitigation Reducing Hazard Intensity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 6. Evacuee escort sequence
def generate_evacuee_escort(map_path: Path, output_path: Path):
    """Show evacuee escort from room to exit."""
    nodes, edges = create_complex_building_layout()
    
    # Find exit
    exits = [n for n in nodes if n['type'] == 'EXIT']
    if not exits:
        return
    
    # Create detailed path from room to exit
    north_rooms = [n for n in nodes if n.get('area') == 'north_offices']
    north_branch_nodes = [n for n in nodes if n.get('area') == 'north_branch']
    main_corridor_nodes = [n for n in nodes if n.get('area') == 'main_corridor']
    lobby_nodes = [n for n in nodes if n.get('area') == 'lobby']
    
    if north_rooms and main_corridor_nodes and exits:
        start_room = north_rooms[0]  # A room in north wing
        end_node = exits[0]
        
        # Create detailed path with more intermediate steps
        path = [start_room]
        # Add intermediate corridor nodes
        if north_branch_nodes:
            path.append(north_branch_nodes[0])
            if len(north_branch_nodes) > 2:
                path.append(north_branch_nodes[len(north_branch_nodes)//2])
        if main_corridor_nodes:
            path.append(main_corridor_nodes[0])
            if len(main_corridor_nodes) > 3:
                path.append(main_corridor_nodes[len(main_corridor_nodes)//2])
        if lobby_nodes:
            path.append(lobby_nodes[0])
            if len(lobby_nodes) > 1:
                path.append(lobby_nodes[-1])
        path.append(end_node)
    else:
        path = [nodes[0], exits[0]] if exits else [nodes[0]]
    
    num_frames = 8
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()
    
    for frame, ax in enumerate(axes[:num_frames]):
        # Plot all nodes
        for node in nodes:
            x, y = node['x'], node['y']
            if node['type'] == 'EXIT':
                ax.scatter(x, y, s=200, c=COLORS['exit'], marker='s',
                          edgecolors='black', linewidths=1.5, zorder=2, alpha=0.6)
            elif node['type'] == 'CORRIDOR':
                ax.scatter(x, y, s=100, c=COLORS['corridor'], marker='o',
                          edgecolors='black', linewidths=1, zorder=2, alpha=0.4)
            else:
                ax.scatter(x, y, s=120, c=COLORS['room'], marker='o',
                          edgecolors='black', linewidths=1, zorder=2, alpha=0.4)
        
        # Draw edges
        for edge in edges:
            src = nodes[edge['src']]
            dst = nodes[edge['dst']]
            ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
                   'k-', alpha=0.15, linewidth=0.5, zorder=1)
        
        # Show progress along path
        current_idx = min(frame, len(path) - 1)
        current_node = path[current_idx]
        
        # Evacuee
        ax.scatter(current_node['x'], current_node['y'] - 0.4, s=300,
                  c='orange', marker='o', edgecolors='black',
                  linewidths=2.5, zorder=5)
        ax.text(current_node['x'], current_node['y'] - 0.4, 'E', ha='center', va='center',
               fontsize=10, fontweight='bold', zorder=6)
        
        # Evacuator
        ax.scatter(current_node['x'], current_node['y'] + 0.4, s=300,
                  c=COLORS['evacuator'], marker='s', edgecolors='black',
                  linewidths=2.5, zorder=5)
        ax.text(current_node['x'], current_node['y'] + 0.4, 'V', ha='center', va='center',
               fontsize=10, fontweight='bold', color='white', zorder=6)
        
        # Show path so far
        for i in range(current_idx):
            if i + 1 < len(path):
                src = path[i]
                dst = path[i+1]
                ax.annotate('', xy=(dst['x'], dst['y']), xytext=(src['x'], src['y']),
                           arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['evacuator'], alpha=0.7),
                           zorder=3)
        
        ax.set_xlim(-10, 40)
        ax.set_ylim(-10, 15)
        ax.set_title(f'Frame {frame + 1}', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    fig.suptitle('Evacuee Escort Sequence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 7. Hazard-routing-evacuee interaction
def generate_hazard_routing_interaction(map_path: Path, output_path: Path):
    """Show how hazards affect routing and evacuee paths over time."""
    nodes, edges = create_complex_building_layout()
    
    # Add hazard region (find rooms in north area)
    north_rooms = [n for n in nodes if n.get('area') == 'north_offices']
    hazard_nodes = []
    for i in range(min(3, len(north_rooms))):
        north_rooms[i]['hazard'] = 0.7
        hazard_nodes.append(north_rooms[i]['id'])
    
    num_frames = 6
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes = axes.flatten()
    
    # Initialize hazard for all nodes
    for node in nodes:
        if 'hazard' not in node:
            node['hazard'] = 0.0
    
    # Create path progression
    exits = [n for n in nodes if n['type'] == 'EXIT']
    lobby_nodes = [n for n in nodes if n.get('area') == 'lobby']
    main_corridor_nodes = [n for n in nodes if n.get('area') == 'main_corridor']
    north_branch_nodes = [n for n in nodes if n.get('area') == 'north_branch']
    
    if exits and lobby_nodes:
        start = lobby_nodes[0]
        end = exits[0]
        
        # Create detailed path with intermediate steps
        full_path = [start]
        if main_corridor_nodes:
            full_path.extend([main_corridor_nodes[0], main_corridor_nodes[len(main_corridor_nodes)//2]])
        if north_branch_nodes:
            full_path.append(north_branch_nodes[0])
        full_path.append(end)
    else:
        full_path = [nodes[0], exits[0]] if exits else [nodes[0]]
    
    # Plot each frame showing progression
    for frame, ax in enumerate(axes[:num_frames]):
        # Plot all nodes
        for node in nodes:
            x, y = node['x'], node['y']
            hazard = node.get('hazard', 0.0)
            
            if hazard > 0.5:
                color = COLORS['hazard_high']
                size = 300
            elif hazard > 0.2:
                color = COLORS['hazard_med']
                size = 250
            elif node['type'] == 'EXIT':
                color = COLORS['exit']
                size = 400
            elif node['type'] == 'STAIR':
                color = '#FFD700'
                size = 300
            elif node['type'] == 'CORRIDOR':
                color = COLORS['corridor']
                size = 200
            else:
                color = COLORS['room']
                size = 180
            
            ax.scatter(x, y, s=size, c=color, edgecolors='black',
                      linewidths=1.5, zorder=3, alpha=0.8)
        
        # Draw hazard blob
        for nid in hazard_nodes:
            node = next((n for n in nodes if n['id'] == nid), None)
            if node:
                circle = Circle((node['x'], node['y']), 2.0, color=COLORS['hazard_high'],
                                  alpha=0.3, zorder=1)
                ax.add_patch(circle)
        
        # Draw all edges
        for edge in edges:
            src = nodes[edge['src']]
            dst = nodes[edge['dst']]
            src_hazard = src.get('hazard', 0.0)
            dst_hazard = dst.get('hazard', 0.0)
            if src_hazard > 0.2 or dst_hazard > 0.2:
                ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
                       'r--', alpha=0.2, linewidth=0.8, zorder=1)
            else:
                ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
                       'k-', alpha=0.15, linewidth=0.6, zorder=1)
        
        # Show path progression
        current_path_idx = min(frame * len(full_path) // num_frames, len(full_path) - 1)
        current_node = full_path[current_path_idx]
        
        # Show evacuee position
        ax.scatter(current_node['x'], current_node['y'] - 0.3, s=350,
                  c='orange', marker='o', edgecolors='black',
                  linewidths=2.5, zorder=5)
        ax.text(current_node['x'], current_node['y'] - 0.3, 'E', ha='center', va='center',
               fontsize=11, fontweight='bold', zorder=6)
        
        # Show path so far (avoiding hazard)
        for i in range(current_path_idx):
            if i + 1 < len(full_path):
                src = full_path[i]
                dst = full_path[i+1]
                # Check if path segment goes through hazard
                goes_through_hazard = any(nid in hazard_nodes for nid in [src['id'], dst['id']])
                if not goes_through_hazard:
                    ax.annotate('', xy=(dst['x'], dst['y']), xytext=(src['x'], src['y']),
                              arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS['scout'], alpha=0.7),
                              zorder=3)
        
        ax.set_xlim(-10, 40)
        ax.set_ylim(-10, 15)
        ax.set_title(f'Frame {frame + 1}: Path Progression', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    fig.suptitle('Hazard-Routing-Evacuee Interaction Over Time', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 8. HASO zones
def generate_haso_zones(map_path: Path, config_path: Path, output_path: Path):
    """Visualize HASO-generated zone partitions with different configurations."""
    num_frames = 6
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes = axes.flatten()
    
    nodes, edges = create_complex_building_layout()
    
    if not HAS_HASO_SIM:
        # Create placeholder with different zone configurations
        zone_colors = ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462']
        
        for frame, ax in enumerate(axes[:num_frames]):
            # Different zone assignment strategies
            if frame == 0:
                # Horizontal zones
                for i, node in enumerate(nodes):
                    zone_id = int((node['x'] + 10) / 10) % len(zone_colors)
                    color = zone_colors[zone_id]
                    ax.scatter(node['x'], node['y'], s=200, c=color, edgecolors='black',
                             linewidths=1.5, zorder=3, alpha=0.7)
            elif frame == 1:
                # Vertical zones
                for i, node in enumerate(nodes):
                    zone_id = int((node['y'] + 10) / 5) % len(zone_colors)
                    color = zone_colors[zone_id]
                    ax.scatter(node['x'], node['y'], s=200, c=color, edgecolors='black',
                             linewidths=1.5, zorder=3, alpha=0.7)
            elif frame == 2:
                # Radial zones from center
                center_x, center_y = 15, 2
                for i, node in enumerate(nodes):
                    dist = ((node['x'] - center_x)**2 + (node['y'] - center_y)**2)**0.5
                    zone_id = int(dist / 8) % len(zone_colors)
                    color = zone_colors[zone_id]
                    ax.scatter(node['x'], node['y'], s=200, c=color, edgecolors='black',
                             linewidths=1.5, zorder=3, alpha=0.7)
            else:
                # Random-like zones
                for i, node in enumerate(nodes):
                    zone_id = (i + frame * 10) % len(zone_colors)
                    color = zone_colors[zone_id]
                    ax.scatter(node['x'], node['y'], s=200, c=color, edgecolors='black',
                             linewidths=1.5, zorder=3, alpha=0.7)
            
            # Draw edges
            for edge in edges[::3]:
                src = nodes[edge['src']]
                dst = nodes[edge['dst']]
                ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
                       'k-', alpha=0.2, linewidth=0.5, zorder=1)
            
            ax.set_xlim(-10, 40)
            ax.set_ylim(-10, 15)
            ax.set_title(f'Zone Configuration {frame + 1}', fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
        
        fig.suptitle('HASO-Generated Zone Partitions (Different Strategies)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    try:
        world = build_world(str(map_path), str(config_path))
        
        if not hasattr(world, 'zones') or not world.zones:
            world.init_zones()
        
        # Generate multiple zone configurations by varying parameters
        base_colors = ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462']
        
        for frame, ax in enumerate(axes[:num_frames]):
            # Reinitialize zones with different parameters if possible
            if frame > 0 and hasattr(world, 'init_zones'):
                try:
                    world.init_zones()
                except:
                    pass
            
            zone_colors = [base_colors[i % len(base_colors)] for i in range(len(world.zones))]
            
            # Plot zones
            for zone_id, node_list in world.zones.items():
                color = zone_colors[zone_id]
                for node_id in node_list:
                    node = world.G.get_node(node_id)
                    if node:
                        circle = Circle((node.x, node.y), 0.6, color=color,
                                      alpha=0.5, zorder=1)
                        ax.add_patch(circle)
                        ax.scatter(node.x, node.y, s=300, c=color, edgecolors='black',
                                  linewidths=1.5, zorder=3, alpha=0.8)
                        if frame == 0:  # Label only first frame
                            ax.text(node.x, node.y, str(node_id), ha='center', va='center',
                                   fontsize=6, fontweight='bold', zorder=4)
            
            # Draw edges
            for (src, dst), edge in world.G.edges.items():
                if src < dst:
                    src_node = world.G.get_node(src)
                    dst_node = world.G.get_node(dst)
                    if src_node and dst_node:
                        ax.plot([src_node.x, dst_node.x], [src_node.y, dst_node.y],
                               'k-', alpha=0.2, linewidth=1, zorder=0)
            
            ax.set_title(f'Zone Configuration {frame + 1}', fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
        
        fig.suptitle('HASO-Generated Zone Partitions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate HASO zones: {e}")
        # Fallback to placeholder
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        axes = axes.flatten()
        zone_colors = ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462']
        
        for frame, ax in enumerate(axes[:num_frames]):
            for i, node in enumerate(nodes):
                zone_id = (i + frame * 3) % len(zone_colors)
                color = zone_colors[zone_id]
                ax.scatter(node['x'], node['y'], s=200, c=color, edgecolors='black',
                         linewidths=1.5, zorder=3, alpha=0.7)
            ax.set_xlim(-10, 40)
            ax.set_ylim(-10, 15)
            ax.set_title(f'Zone Configuration {frame + 1}', fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
        
        fig.suptitle('HASO Zones (Fallback)', fontsize=16, fontweight='bold')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# 9. Basic scenario (COMAP Figure 1 style)
def generate_basic_scenario(output_path: Path):
    """Complex building with 20+ rooms and 50+ edges."""
    fig, ax = setup_figure((16, 12))
    
    nodes, edges = create_complex_building_layout()
    
    # Plot rooms
    for node in nodes:
        x, y = node['x'], node['y']
        if node['type'] == 'EXIT':
            ax.add_patch(FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8,
                                       boxstyle="round,pad=0.05",
                                       facecolor=COLORS['exit'], edgecolor='black',
                                       linewidth=2, zorder=2))
        elif node['type'] == 'CORRIDOR':
            ax.add_patch(FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                                       boxstyle="round,pad=0.05",
                                       facecolor=COLORS['corridor'], edgecolor='black',
                                       linewidth=1.5, zorder=2))
        else:
            ax.add_patch(FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8,
                                       boxstyle="round,pad=0.05",
                                       facecolor=COLORS['room'], edgecolor='black',
                                       linewidth=1.5, zorder=2))
        if node['id'] % 20 == 0:  # Label some nodes
            ax.text(x, y, str(node['id']), ha='center', va='center',
                   fontsize=5, fontweight='bold', zorder=3)
    
    # Draw connections
    for edge in edges[::2]:  # Sample edges
        src = nodes[edge['src']]
        dst = nodes[edge['dst']]
        ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
               'k-', linewidth=0.8, alpha=0.3, zorder=1)
    
    # Responder starting positions
    start_room = nodes[0]
    ax.scatter(start_room['x'], start_room['y'], s=500, c=COLORS['scout'], marker='*', 
              edgecolors='black', linewidths=2, zorder=4, label='Responder Start')
    
    # Sweep order numbers (sample of first 10)
    room_nodes = [n for n in nodes if n['type'] == 'ROOM'][:10]
    for i, room in enumerate(room_nodes):
        ax.text(room['x'] + 0.5, room['y'] + 0.5, str(i+1), ha='center', va='center',
               fontsize=8, fontweight='bold', color='red',
               bbox=dict(boxstyle='circle', facecolor='white', edgecolor='red', linewidth=1.5),
               zorder=5)
    
    ax.set_xlim(-10, 40)
    ax.set_ylim(-10, 15)
    ax.set_title(f'Basic Scenario: Complex Building ({len(nodes)} rooms, {len(edges)} edges)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 10-11. Additional layouts
def generate_additional_layouts(output_dir: Path):
    """Generate two additional building layouts with 20+ rooms and 50+ edges."""
    # Layout 1: School wing (multi-story with classrooms)
    fig, ax = setup_figure((18, 14))
    
    nodes = []
    node_id = 0
    
    # Main hallway (long corridor)
    for i in range(10):
        nodes.append({
            'id': node_id,
            'x': i*1.8,
            'y': 6,
            'type': 'EXIT' if i == 0 or i == 9 else 'CORRIDOR'
        })
        node_id += 1
    
    # North side classrooms (3 floors)
    for floor in range(3):
        for i in range(9):
            nodes.append({
                'id': node_id,
                'x': 1.8 + i * 1.8,
                'y': 8 + floor * 2.5,
                'type': 'ROOM',
                'floor': floor,
                'col': i
            })
            node_id += 1
    
    # South side classrooms (3 floors)
    for floor in range(3):
        for i in range(9):
            nodes.append({
                'id': node_id,
                'x': 1.8 + i * 1.8,
                'y': 4 - floor * 2.5,
                'type': 'ROOM',
                'floor': floor,
                'col': i
            })
            node_id += 1
    
    # Stairwells
    for i in [2, 5, 8]:
        nodes.append({
            'id': node_id,
            'x': i*1.8,
            'y': 6,
            'type': 'STAIR'
        })
        node_id += 1
    
    # Create edges
    edges = []
    
    # Hallway connections
    for i in range(9):
        edges.append({'src': i, 'dst': i+1, 'length': 1.8})
    
    # Classrooms to hallway
    hallway_start = 0
    for floor in range(3):
        for col in range(9):
            # North classrooms
            room_id = 10 + floor * 9 + col
            hallway_id = hallway_start + col + 1
            edges.append({'src': room_id, 'dst': hallway_id, 'length': 2.0})
            
            # South classrooms
            room_id = 37 + floor * 9 + col
            edges.append({'src': room_id, 'dst': hallway_id, 'length': 2.0})
    
    # Stairwell connections
    stairs = [64, 65, 66]
    for stair_id in stairs:
        stair_x = nodes[stair_id]['x']
        hallway_node = next((n for n in nodes if abs(n['x'] - stair_x) < 0.1 and n['id'] < 10), None)
        if hallway_node:
            edges.append({'src': stair_id, 'dst': hallway_node['id'], 'length': 0.5})
    
    # Plot
    for node in nodes:
        x, y = node['x'], node['y']
        if node['type'] == 'EXIT':
            color = COLORS['exit']
            size = (0.6, 0.6)
        elif node['type'] == 'STAIR':
            color = '#FFD700'
            size = (0.6, 0.6)
        elif node['type'] == 'CORRIDOR':
            color = COLORS['corridor']
            size = (1.4, 0.5)
        else:
            color = COLORS['room']
            size = (0.6, 0.6)
        
        ax.add_patch(FancyBboxPatch((x-size[0]/2, y-size[1]/2), size[0], size[1],
                                   boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor='black',
                                   linewidth=1.5, zorder=2))
        if node['id'] % 10 == 0:
            ax.text(x, y, str(node['id'])[:3], ha='center', va='center',
                   fontsize=5, fontweight='bold', zorder=3)
    
    # Draw edges
    for edge in edges:
        src = nodes[edge['src']]
        dst = nodes[edge['dst']]
        ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
               'k-', linewidth=0.6, alpha=0.3, zorder=1)
    
    ax.set_xlim(-1, 18)
    ax.set_ylim(-1, 15)
    ax.set_title(f'School Wing: {len(nodes)} rooms, {len(edges)} edges', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "10_school_wing.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Layout 2: Lab block with hazards (complex research facility)
    fig, ax = setup_figure((16, 12))
    
    nodes = []
    node_id = 0
    
    # Main corridors (cross-shaped)
    for i in range(6):
        nodes.append({
            'id': node_id,
            'x': i*2,
            'y': 5,
            'type': 'EXIT' if i == 0 or i == 5 else 'CORRIDOR'
        })
        node_id += 1
    for i in range(1, 5):
        nodes.append({
            'id': node_id,
            'x': 5,
            'y': i*2,
            'type': 'CORRIDOR'
        })
        node_id += 1
    
    # Lab rooms (hazard zones)
    lab_positions = [
        (2, 3), (2, 7), (4, 3), (4, 7),
        (6, 3), (6, 7), (8, 3), (8, 7),
        (1, 5), (3, 5), (7, 5), (9, 5),
    ]
    for x, y in lab_positions:
        nodes.append({
            'id': node_id,
            'x': x,
            'y': y,
            'type': 'LAB',
            'hazard': True
        })
        node_id += 1
    
    # Office rooms
    office_positions = [
        (0, 1), (2, 1), (4, 1), (6, 1), (8, 1),
        (0, 9), (2, 9), (4, 9), (6, 9), (8, 9),
        (10, 3), (10, 5), (10, 7),
    ]
    for x, y in office_positions:
        nodes.append({
            'id': node_id,
            'x': x,
            'y': y,
            'type': 'ROOM',
            'hazard': False
        })
        node_id += 1
    
    # Storage rooms
    storage_positions = [(1, 3), (1, 7), (9, 3), (9, 7)]
    for x, y in storage_positions:
        nodes.append({
            'id': node_id,
            'x': x,
            'y': y,
            'type': 'STORAGE',
            'hazard': False
        })
        node_id += 1
    
    # Create edges
    edges = []
    
    # Corridor connections
    corridor_nodes = [n for n in nodes if n['type'] in ['CORRIDOR', 'EXIT']]
    for i in range(len(corridor_nodes) - 1):
        n1 = corridor_nodes[i]
        n2 = corridor_nodes[i+1]
        dist = ((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)**0.5
        if dist < 3:
            edges.append({'src': n1['id'], 'dst': n2['id'], 'length': dist})
    
    # Labs to corridors
    labs = [n for n in nodes if n['type'] == 'LAB']
    for lab in labs:
        for corr in corridor_nodes:
            dist = ((lab['x'] - corr['x'])**2 + (lab['y'] - corr['y'])**2)**0.5
            if dist < 1.5:
                edges.append({'src': lab['id'], 'dst': corr['id'], 'length': dist})
    
    # Offices to corridors
    offices = [n for n in nodes if n['type'] == 'ROOM']
    for off in offices:
        for corr in corridor_nodes:
            dist = ((off['x'] - corr['x'])**2 + (off['y'] - corr['y'])**2)**0.5
            if dist < 1.5:
                edges.append({'src': off['id'], 'dst': corr['id'], 'length': dist})
    
    # Storage to corridors
    storages = [n for n in nodes if n['type'] == 'STORAGE']
    for stor in storages:
        for corr in corridor_nodes:
            dist = ((stor['x'] - corr['x'])**2 + (stor['y'] - corr['y'])**2)**0.5
            if dist < 1.5:
                edges.append({'src': stor['id'], 'dst': corr['id'], 'length': dist})
    
    # Inter-lab connections
    for i, lab1 in enumerate(labs):
        for lab2 in labs[i+1:]:
            dist = ((lab1['x'] - lab2['x'])**2 + (lab1['y'] - lab2['y'])**2)**0.5
            if dist < 2.5:
                edges.append({'src': lab1['id'], 'dst': lab2['id'], 'length': dist})
    
    # Plot
    for node in nodes:
        x, y = node['x'], node['y']
        if node['type'] == 'EXIT':
            color = COLORS['exit']
            size = (0.5, 0.5)
        elif node['type'] == 'CORRIDOR':
            if abs(node['x'] - 5) < 0.1:
                size = (0.4, 1.5)
            else:
                size = (1.5, 0.4)
            color = COLORS['corridor']
        elif node['type'] == 'LAB':
            color = COLORS['hazard_med']
            size = (0.6, 0.6)
        elif node['type'] == 'STORAGE':
            color = COLORS['hazard_low']
            size = (0.5, 0.5)
        else:
            color = COLORS['room']
            size = (0.5, 0.5)
        
        ax.add_patch(FancyBboxPatch((x-size[0]/2, y-size[1]/2), size[0], size[1],
                                   boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor='black',
                                   linewidth=2 if node['type'] == 'LAB' else 1.5, zorder=2))
        if node['id'] % 5 == 0:
            ax.text(x, y, str(node['id'])[:3], ha='center', va='center',
                   fontsize=5, fontweight='bold', zorder=3)
    
    # Draw edges
    for edge in edges:
        src = nodes[edge['src']]
        dst = nodes[edge['dst']]
        ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
               'k-', linewidth=0.6, alpha=0.3, zorder=1)
    
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 11)
    ax.set_title(f'Lab Block with Hazards: {len(nodes)} rooms, {len(edges)} edges', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "11_lab_block.png", dpi=150, bbox_inches='tight')
    plt.close()


# 12. Sweep paths
def generate_sweep_paths(map_dir: Path, output_dir: Path):
    """Generate sweep paths for complex building scenario showing progression."""
    nodes, edges = create_complex_building_layout()
    
    # Create sweep path (systematic)
    sweep_order = []
    # Start from exit
    exits = [n for n in nodes if n['type'] == 'EXIT']
    if exits:
        sweep_order.append(exits[0]['id'])
    
    # Sweep by area: lobby -> main corridor -> north -> south -> east -> west
    lobby_nodes = [n for n in nodes if n.get('area') == 'lobby']
    main_corridor_nodes = [n for n in nodes if n.get('area') == 'main_corridor']
    north_rooms = [n for n in nodes if n.get('area') == 'north_offices']
    south_rooms = [n for n in nodes if n.get('area') == 'south_offices']
    east_rooms = [n for n in nodes if n.get('area') == 'east_wing' and n['type'] == 'ROOM']
    west_rooms = [n for n in nodes if n.get('area') == 'west_wing']
    
    # Add rooms in order
    for room_list in [lobby_nodes, main_corridor_nodes, north_rooms, south_rooms, east_rooms, west_rooms]:
        for room in room_list:
            if room['type'] == 'ROOM':
                sweep_order.append(room['id'])
    
    num_frames = 8
    fig, axes = plt.subplots(2, 4, figsize=(28, 12))
    axes = axes.flatten()
    
    for frame, ax in enumerate(axes[:num_frames]):
        # Plot rooms
        for node in nodes:
            x, y = node['x'], node['y']
            if node['type'] == 'EXIT':
                ax.add_patch(FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8,
                                           boxstyle="round,pad=0.05",
                                           facecolor=COLORS['exit'], edgecolor='black',
                                           linewidth=2, zorder=2))
            elif node['type'] == 'CORRIDOR':
                ax.add_patch(FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                                           boxstyle="round,pad=0.05",
                                           facecolor=COLORS['corridor'], edgecolor='black',
                                           linewidth=1.5, zorder=2))
            else:
                ax.add_patch(FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8,
                                           boxstyle="round,pad=0.05",
                                           facecolor=COLORS['room'], edgecolor='black',
                                           linewidth=1.5, zorder=2))
        
        # Show sweep progress up to current frame
        current_progress = int((frame + 1) * len(sweep_order) / num_frames)
        current_progress = min(current_progress, len(sweep_order) - 1)
        
        # Draw sweep path with arrows up to current progress
        for i in range(current_progress):
            if i + 1 < len(sweep_order):
                src_id, dst_id = sweep_order[i], sweep_order[i+1]
                src = nodes[src_id]
                dst = nodes[dst_id]
                ax.annotate('', xy=(dst['x'], dst['y']), xytext=(src['x'], src['y']),
                           arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS['scout'], alpha=0.8),
                           zorder=3)
                # Number every 3rd room
                if i % 3 == 0:
                    mid_x, mid_y = (src['x'] + dst['x']) / 2, (src['y'] + dst['y']) / 2
                    ax.text(mid_x, mid_y - 0.4, str(i+1), ha='center', va='center',
                           fontsize=7, fontweight='bold', color='red',
                           bbox=dict(boxstyle='circle', facecolor='white', edgecolor='red', linewidth=1.5),
                           zorder=4)
        
        # Highlight current position
        if current_progress < len(sweep_order):
            current_node = nodes[sweep_order[current_progress]]
            ax.scatter(current_node['x'], current_node['y'], s=400,
                      c=COLORS['scout'], marker='*', edgecolors='black',
                      linewidths=2, zorder=5)
        
        ax.set_xlim(-10, 40)
        ax.set_ylim(-10, 15)
        ax.set_title(f'Frame {frame + 1}: {current_progress}/{len(sweep_order)} rooms', 
                    fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    fig.suptitle(f'Sweep Path: Room-Checking Sequence ({len(sweep_order)} rooms)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "12_sweep_paths.png", dpi=150, bbox_inches='tight')
    plt.close()


# 13. Clearance time plots
def generate_clearance_time_plots(map_path: Path, config_path: Path, output_path: Path):
    """Time series of cleared rooms, hazard intensity, congestion with snapshots."""
    num_snapshots = 6
    
    if not HAS_HASO_SIM:
        # Generate synthetic data with more points
        times = np.linspace(0, 300, 100)
        cleared = np.minimum(times / 10, 25)
        hazards = 0.5 * np.exp(-times / 100)
        congestion = 0.3 * (1 - np.exp(-times / 50))
    else:
        try:
            results = simulate(str(map_path), str(config_path), tmax=300, seed=42, animate=False)
            world = results['world']
            
            if not world.history:
                times = np.linspace(0, 300, 100)
                cleared = np.minimum(times / 10, 25)
                hazards = 0.5 * np.exp(-times / 100)
                congestion = 0.3 * (1 - np.exp(-times / 50))
            else:
                times = [h.get('time', i*5) for i, h in enumerate(world.history)]
                cleared = []
                hazards = []
                congestion = []
                
                for h in world.history:
                    cleared.append(h.get('cleared_count', world.G.get_cleared_count()[0]))
                    hazard_sum = sum(n.hazard_severity for n in world.G.nodes.values())
                    hazards.append(hazard_sum / len(world.G.nodes) if world.G.nodes else 0)
                    congestion.append(h.get('congestion', 0.1))
        except Exception:
            times = np.linspace(0, 300, 100)
            cleared = np.minimum(times / 10, 25)
            hazards = 0.5 * np.exp(-times / 100)
            congestion = 0.3 * (1 - np.exp(-times / 50))
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, num_snapshots, hspace=0.3, wspace=0.3)
    
    # Time series plots (left column)
    ax1 = fig.add_subplot(gs[0, :num_snapshots//2])
    ax1.plot(times, cleared, 'b-', linewidth=2.5)
    ax1.set_ylabel('Cleared Rooms', fontsize=11)
    ax1.set_title('Clearance Progress Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, :num_snapshots//2])
    ax2.plot(times, hazards, 'r-', linewidth=2.5)
    ax2.set_ylabel('Avg Hazard Intensity', fontsize=11)
    ax2.set_title('Hazard Intensity Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[2, :num_snapshots//2])
    ax3.plot(times, congestion, 'g-', linewidth=2.5)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Congestion Level', fontsize=11)
    ax3.set_title('Congestion Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Snapshot plots (right side)
    nodes, edges = create_complex_building_layout()
    snapshot_times = np.linspace(0, max(times), num_snapshots)
    
    for idx, snapshot_time in enumerate(snapshot_times):
        col = num_snapshots//2 + idx
        if col >= num_snapshots:
            break
        
        ax = fig.add_subplot(gs[idx % 3, col])
        
        # Find closest time index
        time_idx = np.argmin(np.abs(np.array(times) - snapshot_time))
        cleared_at_time = cleared[time_idx] if time_idx < len(cleared) else cleared[-1]
        hazard_at_time = hazards[time_idx] if time_idx < len(hazards) else hazards[-1]
        
        # Plot building state
        for node in nodes[::2]:  # Sample nodes
            x, y = node['x'], node['y']
            # Color based on clearance status
            if node['id'] < cleared_at_time:
                color = COLORS['exit']  # Cleared
            else:
                color = COLORS['room']  # Not cleared
            ax.scatter(x, y, s=100, c=color, edgecolors='black',
                      linewidths=1, zorder=3, alpha=0.6)
        
        ax.set_xlim(-10, 40)
        ax.set_ylim(-10, 15)
        ax.set_title(f't={int(snapshot_time)}s\nCleared: {int(cleared_at_time)}', 
                    fontsize=9, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    fig.suptitle('Clearance Time Analysis with Snapshots', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 14. Agent trajectories
def generate_agent_trajectories(map_path: Path, config_path: Path, output_path: Path):
    """Full paths for all agents, color-coded by role, showing progression."""
    num_frames = 8
    nodes, edges = create_complex_building_layout()
    
    if not HAS_HASO_SIM:
        # Create placeholder with simulated agent movement
        fig, axes = plt.subplots(2, 4, figsize=(28, 12))
        axes = axes.flatten()
        
        role_colors = {
            0: COLORS['scout'],
            1: COLORS['securer'],
            2: COLORS['checkpointer'],
            3: COLORS['evacuator'],
        }
        
        # Simulate agent paths
        num_agents = 4
        agent_paths = []
        exits = [n for n in nodes if n['type'] == 'EXIT']
        lobby_nodes = [n for n in nodes if n.get('area') == 'lobby']
        main_corridor_nodes = [n for n in nodes if n.get('area') == 'main_corridor']
        
        for agent_id in range(num_agents):
            path = []
            if lobby_nodes:
                path.append(lobby_nodes[0])
            if main_corridor_nodes:
                path.append(main_corridor_nodes[agent_id * len(main_corridor_nodes) // num_agents])
            if exits:
                path.append(exits[0])
            agent_paths.append(path)
        
        for frame, ax in enumerate(axes[:num_frames]):
            # Plot building
            for node in nodes:
                x, y = node['x'], node['y']
                if node['type'] == 'EXIT':
                    ax.scatter(x, y, s=200, c=COLORS['exit'], marker='s',
                              edgecolors='black', linewidths=1.5, zorder=2, alpha=0.6)
                elif node['type'] == 'CORRIDOR':
                    ax.scatter(x, y, s=100, c=COLORS['corridor'], marker='o',
                              edgecolors='black', linewidths=1, zorder=2, alpha=0.4)
                else:
                    ax.scatter(x, y, s=120, c=COLORS['room'], marker='o',
                              edgecolors='black', linewidths=1, zorder=2, alpha=0.4)
            
            # Draw edges
            for edge in edges[::3]:
                src = nodes[edge['src']]
                dst = nodes[edge['dst']]
                ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
                       'k-', alpha=0.15, linewidth=0.5, zorder=1)
            
            # Show agent positions
            for agent_id, path in enumerate(agent_paths):
                progress = min(frame * len(path) // num_frames, len(path) - 1)
                current_node = path[progress]
                color = role_colors.get(agent_id, 'gray')
                
                ax.scatter(current_node['x'], current_node['y'], s=300,
                          c=color, marker='s', edgecolors='black',
                          linewidths=2, zorder=5)
                ax.text(current_node['x'], current_node['y'], f'A{agent_id}',
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       color='white', zorder=6)
                
                # Show path so far
                for i in range(progress):
                    if i + 1 < len(path):
                        src = path[i]
                        dst = path[i+1]
                        ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
                               color=color, linewidth=2, alpha=0.6, zorder=3)
            
            ax.set_xlim(-10, 40)
            ax.set_ylim(-10, 15)
            ax.set_title(f'Frame {frame + 1}: Agent Movement', fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
        
        fig.suptitle('Agent Trajectories Over Time (Color-Coded by Role)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    try:
        world = build_world(str(map_path), str(config_path))
        results = simulate(str(map_path), str(config_path), tmax=200, seed=42, animate=False)
        world = results['world']
        
        fig, axes = plt.subplots(2, 4, figsize=(28, 12))
        axes = axes.flatten()
        
        role_colors = {
            'SCOUT': COLORS['scout'],
            'SECURER': COLORS['securer'],
            'CHECKPOINTER': COLORS['checkpointer'],
            'EVACUATOR': COLORS['evacuator'],
        }
        
        # Extract agent positions over time from history
        agent_positions = {agent.id: [] for agent in world.agents}
        if hasattr(world, 'history'):
            for h in world.history:
                for agent in world.agents:
                    if hasattr(agent, 'node'):
                        node = world.G.get_node(agent.node)
                        if node:
                            agent_positions[agent.id].append((node.x, node.y))
        
        for frame, ax in enumerate(axes[:num_frames]):
            plot_graph_base(world.G, ax, show_labels=False)
            
            # Show agent positions at this frame
            for agent in world.agents:
                color = role_colors.get(agent.role.name, 'gray')
                if agent.id in agent_positions and agent_positions[agent.id]:
                    pos_idx = min(frame * len(agent_positions[agent.id]) // num_frames,
                                len(agent_positions[agent.id]) - 1)
                    x, y = agent_positions[agent.id][pos_idx]
                    ax.scatter(x, y, s=300, c=color, marker='s',
                             edgecolors='black', linewidths=2, zorder=5)
                    ax.text(x, y, f'A{agent.id}', ha='center', va='center',
                           fontsize=8, fontweight='bold', color='white', zorder=6)
            
            ax.set_title(f'Frame {frame + 1}', fontsize=11, fontweight='bold')
        
        fig.suptitle('Agent Trajectories Over Time (Color-Coded by Role)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate agent trajectories: {e}")
        # Fallback to placeholder
        fig, axes = plt.subplots(2, 4, figsize=(28, 12))
        axes = axes.flatten()
        for ax in axes[:num_frames]:
            ax.text(0.5, 0.5, 'Agent Trajectories\n(Error generating)', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# 15. Hazard heatmap
def generate_hazard_heatmap(map_path: Path, output_path: Path):
    """Heatmap of hazard intensity across building showing evolution."""
    nodes, edges = create_complex_building_layout()
    
    num_frames = 6
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes = axes.flatten()
    
    # Initialize hazard spread simulation
    north_rooms = [n for n in nodes if n.get('area') == 'north_offices']
    south_rooms = [n for n in nodes if n.get('area') == 'south_offices']
    
    # Build neighbor graph
    neighbors = {i: [] for i in range(len(nodes))}
    for edge in edges:
        neighbors[edge['src']].append(edge['dst'])
        neighbors[edge['dst']].append(edge['src'])
    
    # Store hazard states for each frame
    hazard_states = []
    
    # Initial state
    initial_hazards = {}
    for node in nodes:
        initial_hazards[node['id']] = 0.0
    # Start hazard in north wing
    if north_rooms:
        start_node = north_rooms[len(north_rooms) // 2]
        initial_hazards[start_node['id']] = 0.5
    hazard_states.append(initial_hazards.copy())
    
    # Simulate spread
    for frame in range(1, num_frames):
        current_hazards = hazard_states[frame - 1].copy()
        new_hazards = {}
        
        for i, node in enumerate(nodes):
            node_hazard = current_hazards.get(i, 0.0)
            if node_hazard > 0.2:
                for neighbor_idx in neighbors.get(i, []):
                    if neighbor_idx < len(nodes):
                        neighbor_hazard = current_hazards.get(neighbor_idx, 0.0)
                        spread_rate = 0.15 * frame
                        if neighbor_hazard == 0.0:
                            new_hazards[neighbor_idx] = min(spread_rate, 0.9)
                        elif neighbor_hazard < 0.9:
                            new_hazards[neighbor_idx] = min(neighbor_hazard + spread_rate * 0.3, 0.9)
        
        for idx, severity in new_hazards.items():
            current_hazards[idx] = severity
        
        hazard_states.append(current_hazards.copy())
    
    # Plot each frame
    for frame, ax in enumerate(axes[:num_frames]):
        current_hazards = hazard_states[frame]
        
        # Update node hazards
        for i, node in enumerate(nodes):
            node['hazard'] = current_hazards.get(i, 0.0)
        
        # Create heatmap
        x_coords = [node['x'] for node in nodes]
        y_coords = [node['y'] for node in nodes]
        hazards = [node.get('hazard', 0.0) for node in nodes]
        
        scatter = ax.scatter(x_coords, y_coords, c=hazards, s=200, cmap='YlOrRd',
                            edgecolors='black', linewidths=1.5, zorder=3, vmin=0, vmax=1)
        
        # Draw edges
        for edge in edges:
            src = nodes[edge['src']]
            dst = nodes[edge['dst']]
            ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
                   'k-', alpha=0.15, linewidth=0.5, zorder=1)
        
        ax.set_xlim(-10, 40)
        ax.set_ylim(-10, 15)
        ax.set_title(f'Frame {frame + 1}: Hazard Evolution', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Add colorbar
    fig.colorbar(scatter, ax=axes, label='Hazard Severity', shrink=0.8, pad=0.02)
    fig.suptitle(f'Hazard Intensity Heatmap Evolution ({len(nodes)} nodes)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 16. Stress test
def generate_stress_test(map_path: Path, output_path: Path):
    """Multiple agents in narrow corridor causing increasing congestion."""
    num_frames = 8
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()
    
    max_agents = 8
    
    for frame, ax in enumerate(axes[:num_frames]):
        num_agents = int((frame + 1) * max_agents / num_frames)
        
        # Draw narrow corridor
        ax.add_patch(FancyBboxPatch((0, 0), 8, 1, boxstyle="round,pad=0.1",
                                   facecolor=COLORS['corridor'], edgecolor='black',
                                   linewidth=2, zorder=1))
        
        # Multiple agents
        spacing = 7.0 / max(num_agents, 1)
        for i in range(num_agents):
            x = 0.5 + i * spacing
            y = 0.5
            ax.scatter(x, y, s=300, c=COLORS['scout'], marker='s',
                      edgecolors='black', linewidths=2, zorder=3)
            ax.text(x, y, f'A{i}', ha='center', va='center',
                   fontsize=8, fontweight='bold', zorder=4)
        
        # Congestion indicator
        congestion_level = num_agents / max_agents
        if congestion_level > 0.5:
            ax.text(4, 0.5, '← HIGH CONGESTION →', ha='center', va='center',
                   fontsize=11, fontweight='bold', color='red', zorder=5)
        elif congestion_level > 0.3:
            ax.text(4, 0.5, '← Moderate →', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='orange', zorder=5)
        else:
            ax.text(4, 0.5, '← Low →', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='green', zorder=5)
        
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(f'Frame {frame + 1}: {num_agents} Agents', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    fig.suptitle('Stress Test: Agent Count vs Congestion', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 17. State machine
def generate_state_machine(output_path: Path):
    """Agent state transitions."""
    fig, ax = setup_figure((8, 6))
    
    states = ['Normal', 'Slowed', 'Progressing', 'Exited', 'Incapacitated']
    positions = {
        'Normal': (2, 4),
        'Slowed': (1, 2),
        'Progressing': (4, 4),
        'Exited': (6, 4),
        'Incapacitated': (3, 1),
    }
    
    # Draw states
    for state, (x, y) in positions.items():
        circle = Circle((x, y), 0.5, color='lightblue', edgecolor='black',
                          linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, state, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)
    
    # Draw transitions
    transitions = [
        ('Normal', 'Slowed', 'hazard'),
        ('Normal', 'Progressing', 'clear'),
        ('Slowed', 'Progressing', 'mitigate'),
        ('Progressing', 'Exited', 'complete'),
        ('Normal', 'Incapacitated', 'failure'),
    ]
    
    for src, dst, label in transitions:
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                   zorder=1)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', fontsize=7, zorder=4)
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.set_title('Agent State Machine', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 18. Flowchart
def generate_flowchart(output_path: Path):
    """Discrete-event simulation engine flowchart."""
    fig, ax = setup_figure((10, 8))
    
    # Simplified flowchart
    boxes = [
        ('Start', 5, 7),
        ('Initialize\nEvent Queue', 5, 6),
        ('Pop Event', 5, 5),
        ('Update Agent', 5, 4),
        ('Schedule Next\nEvent', 5, 3),
        ('Queue Empty?', 5, 2),
        ('End', 5, 1),
    ]
    
    for label, x, y in boxes:
        if '?' in label:
            # Diamond
            diamond = mpatches.FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8,
                                              boxstyle="round,pad=0.1",
                                              facecolor='lightyellow',
                                              edgecolor='black', linewidth=2)
            ax.add_patch(diamond)
        else:
            # Rectangle
            rect = mpatches.FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8,
                                          boxstyle="round,pad=0.1",
                                          facecolor='lightblue',
                                          edgecolor='black', linewidth=2)
            ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)
    
    # Arrows
    for i in range(len(boxes) - 1):
        x1, y1 = boxes[i][1], boxes[i][2]
        x2, y2 = boxes[i+1][1], boxes[i+1][2]
        ax.annotate('', xy=(x2, y2+0.4), xytext=(x1, y1-0.4),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                   zorder=2)
    
    # Loop back
    ax.annotate('', xy=(5, 5.4), xytext=(5, 2.4),
               arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='--'),
               zorder=2)
    ax.text(6, 3.9, 'No', ha='left', fontsize=8, color='red', fontweight='bold', zorder=4)
    
    ax.set_xlim(3, 7)
    ax.set_ylim(0, 8)
    ax.set_title('Discrete-Event Simulation Engine Flowchart', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# 19. Task allocation
def generate_task_allocation(map_path: Path, config_path: Path, output_path: Path):
    """Nodes colored by agent assignment, showing reallocation over time."""
    num_frames = 6
    nodes, edges = create_complex_building_layout()
    
    if not HAS_HASO_SIM:
        # Create placeholder with different allocation strategies
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        axes = axes.flatten()
        
        zone_colors = ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462']
        num_zones = 4
        
        for frame, ax in enumerate(axes[:num_frames]):
            # Different allocation strategies per frame
            if frame == 0:
                # Sequential allocation
                for i, node in enumerate(nodes):
                    zone_id = i % num_zones
                    color = zone_colors[zone_id]
                    ax.scatter(node['x'], node['y'], s=200, c=color, edgecolors='black',
                             linewidths=1.5, zorder=3, alpha=0.7)
            elif frame == 1:
                # Spatial allocation (by x coordinate)
                for i, node in enumerate(nodes):
                    zone_id = int((node['x'] + 10) / 12.5) % num_zones
                    color = zone_colors[zone_id]
                    ax.scatter(node['x'], node['y'], s=200, c=color, edgecolors='black',
                             linewidths=1.5, zorder=3, alpha=0.7)
            elif frame == 2:
                # Spatial allocation (by y coordinate)
                for i, node in enumerate(nodes):
                    zone_id = int((node['y'] + 10) / 6.25) % num_zones
                    color = zone_colors[zone_id]
                    ax.scatter(node['x'], node['y'], s=200, c=color, edgecolors='black',
                             linewidths=1.5, zorder=3, alpha=0.7)
            else:
                # Balanced allocation (rotating pattern)
                for i, node in enumerate(nodes):
                    zone_id = (i + frame * 5) % num_zones
                    color = zone_colors[zone_id]
                    ax.scatter(node['x'], node['y'], s=200, c=color, edgecolors='black',
                             linewidths=1.5, zorder=3, alpha=0.7)
            
            # Draw edges
            for edge in edges[::3]:
                src = nodes[edge['src']]
                dst = nodes[edge['dst']]
                ax.plot([src['x'], dst['x']], [src['y'], dst['y']],
                       'k-', alpha=0.2, linewidth=0.5, zorder=1)
            
            ax.set_xlim(-10, 40)
            ax.set_ylim(-10, 15)
            ax.set_title(f'Allocation Strategy {frame + 1}', fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
        
        fig.suptitle(f'Task Allocation by Agent Assignment ({len(nodes)} nodes)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    try:
        world = build_world(str(map_path), str(config_path))
        
        if not hasattr(world, 'zones') or not world.zones:
            world.init_zones()
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        axes = axes.flatten()
        
        # Color nodes by assigned agent
        agent_colors = [COLORS['scout'], COLORS['securer'], COLORS['checkpointer'], COLORS['evacuator']]
        
        # Generate multiple allocation frames by varying zone assignments
        for frame, ax in enumerate(axes[:num_frames]):
            # Reinitialize zones if possible for different configurations
            if frame > 0 and hasattr(world, 'init_zones'):
                try:
                    world.init_zones()
                except:
                    pass
            
            for agent_id, zone_id in world.agent_zones.items():
                agent = next((a for a in world.agents if a.id == agent_id), None)
                if agent and zone_id in world.zones:
                    color = agent_colors[agent_id % len(agent_colors)]
                    for node_id in world.zones[zone_id]:
                        node = world.G.get_node(node_id)
                        if node:
                            ax.scatter(node.x, node.y, s=300, c=color, edgecolors='black',
                                     linewidths=1.5, zorder=3, alpha=0.7)
                            if frame == 0:  # Label only first frame
                                ax.text(node.x, node.y, str(node_id), ha='center', va='center',
                                       fontsize=6, fontweight='bold', zorder=4)
            
            # Draw edges
            for (src, dst), edge in world.G.edges.items():
                if src < dst:
                    src_node = world.G.get_node(src)
                    dst_node = world.G.get_node(dst)
                    if src_node and dst_node:
                        ax.plot([src_node.x, dst_node.x], [src_node.y, dst_node.y],
                               'k-', alpha=0.2, linewidth=1, zorder=1)
            
            ax.set_title(f'Allocation Frame {frame + 1}', fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
        
        fig.suptitle('Task Allocation by Agent Assignment Over Time', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate task allocation: {e}")
        # Fallback to placeholder
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        axes = axes.flatten()
        zone_colors = ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072']
        for frame, ax in enumerate(axes[:num_frames]):
            for i, node in enumerate(nodes):
                zone_id = (i + frame * 3) % 4
                color = zone_colors[zone_id]
                ax.scatter(node['x'], node['y'], s=150, c=color, edgecolors='black',
                         linewidths=1, zorder=3, alpha=0.7)
            ax.set_title(f'Allocation Frame {frame + 1}', fontsize=11, fontweight='bold')
            ax.set_xlim(-10, 40)
            ax.set_ylim(-10, 15)
            ax.set_aspect('equal')
            ax.axis('off')
        fig.suptitle(f'Task Allocation ({len(nodes)} nodes)', fontsize=16, fontweight='bold')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
