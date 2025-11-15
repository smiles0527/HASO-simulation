"""
HASO Algorithm Visualization System

Complete visualization suite for the Hierarchical Agent-based Sweep Optimization algorithm.
Creates network diagrams showing:
- Room clearance status (color-coded)
- Evacuee locations and counts
- Flow dynamics through the building
- Agent positions and zones
- Hazard propagation
"""

print("\n" + "="*70)
print("HASO ALGORITHM - EVACUATION VISUALIZATION SYSTEM")
print("="*70 + "\n")

import sys
import os

print("Loading HASO framework...")
from notebooks import simulate
from haso_sim import (
    generate_summary_report,
    analyze_simulation_results,
    Graph,
    FlowDynamics,
    suggest_optimal_routes,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

print("Framework loaded successfully.\n")

# Run simulation
print("Running HASO evacuation simulation...")
print()

results = simulate(
    map_path="notebooks/data/office_building_simple.yaml",
    config_path="notebooks/data/config_baseline.yaml",
    tmax=300,
    seed=42,
    animate=False
)

world = results['world']
cleared, total = world.G.get_cleared_count()

print(f"Simulation complete: {cleared}/{total} rooms cleared\n")

# ==================== NETWORK VISUALIZATION ====================

def create_network_diagram(world, save_path='demo_results/haso_network.png'):
    """
    Create a network diagram showing:
    - Color-coded nodes by status
    - Evacuee counts
    - Flow indicators
    - Exits marked
    - Electrical flow dynamics (current, voltage, resistance)
    """
    fig, ax = plt.subplots(figsize=(18, 12))
    
    G = world.G
    
    # Initialize flow dynamics model
    flow_model = FlowDynamics(G)
    
    # Calculate flows for all edges
    for (src, dst) in G.edges.keys():
        flow = flow_model.calculate_flow_rate(src, dst)
        flow_model.edge_flow[(src, dst)] = flow
    
    # Color scheme
    COLOR_EXIT = '#4A90E2'          # Blue - exits
    COLOR_EVACUATING = '#FFA726'    # Orange - evacuating
    COLOR_CLEARED = '#66BB6A'       # Green - cleared
    COLOR_NOT_CLEARED = '#EF5350'   # Red - not cleared
    COLOR_UNKNOWN = '#8D6E63'       # Brown - unknown
    
    # Draw edges first (background)
    for (src, dst), edge in G.edges.items():
        if src < dst:  # Only draw once
            node_src = G.get_node(src)
            node_dst = G.get_node(dst)
            
            if node_src and node_dst:
                # Calculate flow (number of evacuees moving)
                flow = 0.0
                if node_src.evacuees:
                    flow = len([e for e in node_src.evacuees if e.evacuating]) * 0.1
                
                # Calculate electrical properties
                edge_length = edge.length if hasattr(edge, 'length') else 10.0
                edge_width = edge.width if hasattr(edge, 'width') else 1.5
                resistance = flow_model.calculate_edge_resistance(edge, node_src, node_dst)
                edge_flow_rate = flow_model.edge_flow.get((src, dst), 0.0)
                
                # Edge thickness based on conductance (1/R)
                # Higher conductance = thicker line = better flow
                conductance = 1.0 / resistance if resistance > 0 and resistance != float('inf') else 0.0
                thickness = max(1, min(6, conductance * 3))
                
                # Edge color based on flow rate
                # Blue = high flow, Gray = low/no flow
                if edge_flow_rate > 0.5:
                    edge_color = '#2196F3'  # Blue - high flow
                    alpha = 0.8
                elif edge_flow_rate > 0.1:
                    edge_color = '#4CAF50'  # Green - moderate flow
                    alpha = 0.6
                else:
                    edge_color = 'black'    # Black - minimal flow
                    alpha = 0.4
                
                # Draw edge
                ax.plot([node_src.x, node_dst.x], 
                       [node_src.y, node_dst.y],
                       edge_color, linewidth=thickness, alpha=alpha, zorder=1)
                
                # Add edge info label at midpoint
                mid_x = (node_src.x + node_dst.x) / 2
                mid_y = (node_src.y + node_dst.y) / 2
                
                # Show resistance and flow
                info_text = f'{edge_length:.0f}m'
                if resistance < 100:
                    info_text += f'\nR={resistance:.1f}Ω'
                if edge_flow_rate > 0.05:
                    info_text += f'\nI={edge_flow_rate:.2f}A'
                
                if edge_length >= 10 or edge_flow_rate > 0.05:
                    bg_color = 'lightcyan' if edge_flow_rate > 0.05 else 'lightyellow'
                    ax.text(mid_x, mid_y, info_text,
                           ha='center', va='center',
                           fontsize=7, style='italic',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=bg_color, alpha=0.8, edgecolor='black', linewidth=0.5),
                           zorder=2)
                
                # Draw arrow to show flow direction
                if flow > 0:
                    dx = node_dst.x - node_src.x
                    dy = node_dst.y - node_src.y
                    arrow = FancyArrowPatch(
                        (mid_x - dx*0.15, mid_y - dy*0.15),
                        (mid_x + dx*0.15, mid_y + dy*0.15),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=3, color='black', alpha=0.7, zorder=2
                    )
                    ax.add_patch(arrow)
    
    # Draw nodes
    for node_id, node in G.nodes.items():
        # Determine color based on status
        if node.node_type.name == 'EXIT':
            color = COLOR_EXIT
            marker_style = 'o'  # Circle for exits
            size = 1000
        elif node.cleared:
            color = COLOR_CLEARED
            marker_style = 'o'
            size = 800
        elif node.fog_state == 0:
            color = COLOR_UNKNOWN
            marker_style = 'o'
            size = 800
        else:
            # Check if evacuees are evacuating
            if node.evacuees and any(e.evacuating for e in node.evacuees):
                color = COLOR_EVACUATING
            else:
                color = COLOR_NOT_CLEARED
            marker_style = 'o'
            size = 800
        
        # Draw node
        ax.scatter(node.x, node.y, 
                  c=color, s=size, marker=marker_style,
                  edgecolors='black', linewidths=3, zorder=3)
        
        # Add progress bar for nodes being secured
        # Check if any agent is currently at this node
        agents_here = [a for a in world.agents if a.node == node_id]
        if agents_here and not node.cleared and node.node_type.name != 'EXIT':
            # Calculate progress based on search time
            # Assume nodes are being searched if agents are present
            progress = 0.5  # 50% progress as an example
            
            # Draw progress bar above the node
            bar_width = 4
            bar_height = 0.5
            bar_x = node.x - bar_width / 2
            bar_y = node.y + 3.5
            
            # Background bar (gray)
            bar_bg = mpatches.Rectangle((bar_x, bar_y), bar_width, bar_height,
                                       facecolor='lightgray', edgecolor='black', 
                                       linewidth=1, zorder=5)
            ax.add_patch(bar_bg)
            
            # Progress bar (yellow to green gradient)
            if progress > 0:
                bar_color = '#FFC107' if progress < 0.7 else '#4CAF50'
                bar_progress = mpatches.Rectangle((bar_x, bar_y), bar_width * progress, bar_height,
                                                 facecolor=bar_color, edgecolor='none', zorder=6)
                ax.add_patch(bar_progress)
            
            # Progress text
            ax.text(node.x, bar_y + bar_height + 0.3, 
                   f'{progress*100:.0f}%',
                   ha='center', va='bottom',
                   fontsize=8, fontweight='bold',
                   color='black', zorder=6)
        
        # Add node label (room number or '?' for unknown)
        if node.fog_state == 0:
            label_text = '?'
        else:
            label_text = str(node_id)
        
        ax.text(node.x, node.y, label_text,
               ha='center', va='center',
               fontsize=16, fontweight='bold',
               color='white',
               zorder=4)
        
        # Add exit arrow below exit nodes
        if node.node_type.name == 'EXIT':
            arrow_exit = FancyArrowPatch(
                (node.x, node.y - 4),
                (node.x, node.y - 6),
                arrowstyle='->', mutation_scale=25,
                linewidth=3, color='black', zorder=5
            )
            ax.add_patch(arrow_exit)
        
        # Add evacuee count and voltage below node
        if node.evacuees or node.node_type.name != 'EXIT':
            info_lines = []
            
            if node.evacuees:
                evac_count = len(node.evacuees)
                flow_ratio = len([e for e in node.evacuees if e.evacuating]) / max(1, evac_count)
                info_lines.append(f"# Evacuees = {evac_count}")
                info_lines.append(f"Flow = {flow_ratio:.1f}")
            
            # Add voltage (pressure) for non-exit nodes
            if node.node_type.name != 'EXIT':
                voltage = flow_model.node_pressure.get(node_id, 0)
                if voltage > 1.0:
                    info_lines.append(f"V = {voltage:.1f}V")
            
            if info_lines:
                offset = 8 if node.node_type.name == 'EXIT' else 4
                ax.text(node.x, node.y - offset, 
                       '\n'.join(info_lines),
                       ha='center', va='top',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.4', 
                               facecolor='white', alpha=0.9, edgecolor='black'),
                       zorder=5)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_EXIT, edgecolor='black', label='Blue - exits'),
        mpatches.Patch(facecolor=COLOR_EVACUATING, edgecolor='black', label='Orange - evacuating'),
        mpatches.Patch(facecolor=COLOR_CLEARED, edgecolor='black', label='Green - cleared'),
        mpatches.Patch(facecolor=COLOR_NOT_CLEARED, edgecolor='black', label='Red - not cleared'),
        mpatches.Patch(facecolor=COLOR_UNKNOWN, edgecolor='black', label='Brown - unknown'),
        mpatches.Patch(facecolor='white', edgecolor='white', label=''),  # Spacer
        mpatches.Patch(facecolor='#2196F3', edgecolor='black', label='High flow path'),
        mpatches.Patch(facecolor='#4CAF50', edgecolor='black', label='Moderate flow'),
        mpatches.Patch(facecolor='gray', edgecolor='black', label='Low/no flow'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95, 
             edgecolor='black', fancybox=False, title='Node Status & Flow Dynamics')
    
    # Styling
    ax.set_aspect('equal')
    ax.grid(False)  # No grid for cleaner look
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_title('', fontsize=16, fontweight='bold', pad=20)
    
    # Remove axis ticks and labels for cleaner appearance
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1)
    
    # Add timestamp and stats at bottom
    flow_metrics = flow_model.get_flow_metrics()
    stats_text = f"Time: {world.time:.0f}s | Cleared: {cleared}/{total} | Evacuees: {sum(len(n.evacuees) for n in G.nodes.values())}"
    stats_text += f" | Total Flow: {flow_metrics['total_flow']:.2f}A | Avg Pressure: {flow_metrics['avg_pressure']:.1f}V"
    ax.text(0.5, -0.02, stats_text,
           transform=ax.transAxes, ha='center', va='top',
           fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Network diagram saved: {save_path}")
    
    return fig, ax


def create_zone_visualization(world, save_path='demo_results/haso_zones.png'):
    """
    Visualize zone assignments with color coding.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    G = world.G
    
    # Zone colors
    zone_colors = ['#E3F2FD', '#FFE0B2', '#C8E6C9', '#F8BBD0', '#D1C4E9', '#FFCCBC']
    
    # Draw zones as shaded regions
    if world.zones:
        for zone_id, node_list in world.zones.items():
            if not node_list:
                continue
            
            # Get node positions
            xs = [G.get_node(n).x for n in node_list if G.get_node(n)]
            ys = [G.get_node(n).y for n in node_list if G.get_node(n)]
            
            if xs and ys:
                # Draw convex hull around zone
                from matplotlib.patches import Polygon
                if len(xs) >= 3:
                    # Simple bounding box for now
                    min_x, max_x = min(xs) - 3, max(xs) + 3
                    min_y, max_y = min(ys) - 3, max(ys) + 3
                    rect = mpatches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                             facecolor=zone_colors[zone_id % len(zone_colors)],
                                             alpha=0.3, edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
                    
                    # Label zone
                    ax.text((min_x + max_x) / 2, max_y + 1,
                           f'Zone {zone_id}',
                           ha='center', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Draw edges
    for (src, dst), edge in G.edges.items():
        if src < dst:
            node_src = G.get_node(src)
            node_dst = G.get_node(dst)
            if node_src and node_dst:
                ax.plot([node_src.x, node_dst.x], 
                       [node_src.y, node_dst.y],
                       'k-', linewidth=1.5, alpha=0.4, zorder=1)
    
    # Draw nodes
    for node_id, node in G.nodes.items():
        color = '#4A90E2' if node.node_type.name == 'EXIT' else '#BDBDBD'
        ax.scatter(node.x, node.y, c=color, s=400,
                  edgecolors='black', linewidths=2, zorder=3)
        ax.text(node.x, node.y, str(node_id),
               ha='center', va='center',
               fontsize=12, fontweight='bold', color='white', zorder=4)
    
    # Draw agents
    agent_colors = {'SCOUT': '#4CAF50', 'SECURER': '#2196F3', 
                   'CHECKPOINTER': '#FF9800', 'EVACUATOR': '#9C27B0'}
    for agent in world.agents:
        node = G.get_node(agent.node)
        if node:
            color = agent_colors.get(agent.role.name, '#000000')
            # Draw agent as star
            ax.scatter(node.x, node.y, c=color, s=600, marker='*',
                      edgecolors='black', linewidths=2, zorder=5)
            ax.text(node.x + 1.5, node.y, f"A{agent.id}\n{agent.role.name[:3]}",
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('X Position (meters)', fontsize=12)
    ax.set_ylabel('Y Position (meters)', fontsize=12)
    ax.set_title('HASO Zone Assignments & Agent Positions', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Zone visualization saved: {save_path}")
    
    return fig, ax


# ==================== CREATE VISUALIZATIONS ====================

print("="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)
print()

# Create output directory
os.makedirs('demo_results', exist_ok=True)

# 1. Network diagram
print("[1/3] Creating network diagram...")
create_network_diagram(world)

# 2. Zone visualization
print("[2/3] Creating zone visualization...")
create_zone_visualization(world)

# 3. Comprehensive dashboard
print("[3/3] Creating full dashboard...")
from haso_sim import create_summary_dashboard
create_summary_dashboard(world, save_path='demo_results/haso_complete.png')

print()
print("="*70)
print("HASO METRICS")
print("="*70)
print()

# Analysis
analysis = analyze_simulation_results(world)

print(f"Time Elapsed:          {world.time:.1f} seconds ({world.time/60:.1f} minutes)")
print(f"Rooms Cleared:         {cleared}/{total} ({cleared/total*100:.1f}%)")
print(f"Efficiency (eta):      {analysis['haso_efficiency_ratio']:.4f} rooms/second")
print(f"Redundancy Index (R):  {analysis['haso_redundancy_index']:.3f}")
print(f"Risk Exposure (E):     {analysis['haso_risk_exposure']:.3f}")
print()

# Zone assignments
if world.zones:
    print("ZONE ASSIGNMENTS:")
    print("-" * 70)
    for agent in world.agents:
        if agent.assigned_zone != -1:
            zone_size = len(world.zones.get(agent.assigned_zone, []))
            print(f"  Agent {agent.id} ({agent.role.name:15s}) -> Zone {agent.assigned_zone} ({zone_size} rooms)")
    print()

# Hazard status
hazard_nodes = [n for n in world.G.nodes.values() if n.hazard_severity > 0]
if hazard_nodes:
    print("HAZARD STATUS:")
    print("-" * 70)
    for node in hazard_nodes:
        print(f"  Node {node.id} ({node.name}): {node.hazard.name}")
        print(f"    Severity: {node.hazard_severity:.2f} | Visibility: {node.visibility:.2f}")
    print()

# Full report
print("="*70)
print("COMPLETE SIMULATION REPORT")
print("="*70)
print()
report = generate_summary_report(world)
print(report)

print()
print("="*70)
print("VISUALIZATIONS GENERATED")
print("="*70)
print()
print("Files created in demo_results/:")
print("  1. haso_network.png     - Network diagram")
print("  2. haso_zones.png       - Zone assignments with agent positions")  
print("  3. haso_complete.png    - Complete dashboard with all metrics")
print()
print("="*70)
print("HASO Visualization Complete")
print("="*70)
print()

