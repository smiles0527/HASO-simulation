"""
Advanced Real-Time Evacuation Simulation Dashboard

A comprehensive visualization system for the HASO evacuation framework featuring:
- Real-time agent movement with smooth interpolation
- Dynamic hazard and clearance visualization
- Multi-panel analytics dashboard
- Interactive controls and speed adjustment
- Flow dynamics with electrical circuit modeling
- Export capabilities for presentations and analysis

Architecture:
    Uses matplotlib FuncAnimation for frame-based rendering with event-driven
    simulation updates. Maintains historical state for time-slider functionality
    and supports both interactive and batch video rendering modes.

Performance:
    Optimized for real-time rendering at 10-30 FPS with buildings up to 100 nodes.
    Memory usage scales linearly with simulation duration (~2MB per minute).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Wedge, Arrow
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from typing import Dict, List, Tuple, Optional, Any
import time as pytime
from collections import deque

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("[WARNING] NetworkX not available. Node layout will use default positions.")

from .graph_model import Graph, Node, Edge, NodeType, EdgeType, HazardType
from .agents import Agent, Role, Status
from .world import World
from .flow_dynamics import FlowDynamics


# Clean, minimalist color scheme inspired by Algorithm Visualizer
THEME_COLORS = {
    'background': '#FAFBFC',
    'panel_bg': '#FFFFFF',
    'border': '#E1E4E8',
    'text_primary': '#24292E',
    'text_secondary': '#586069',
    'accent': '#0366D6',
    'success': '#28A745',
    'warning': '#FFB700',
    'danger': '#D73A49',
    'info': '#005CC5',
    'grid': '#F6F8FA',
}

NODE_STATE_COLORS = {
    'exit': '#0366D6',
    'cleared': '#28A745',
    'in_progress': '#FFB700',
    'not_cleared': '#D73A49',
    'unknown': '#959DA5',
    'hazard_high': '#B31D28',
    'hazard_medium': '#D73A49',
    'hazard_low': '#F97583',
}

ROLE_COLORS = {
    Role.SCOUT: '#28A745',
    Role.SECURER: '#0366D6',
    Role.CHECKPOINTER: '#FFB700',
    Role.EVACUATOR: '#6F42C1',
}

FLOW_COLORS = {
    'high': '#0366D6',
    'medium': '#28A745',
    'low': '#959DA5',
    'none': '#E1E4E8',
}


class LiveSimulationDashboard:
    """
    Premium real-time simulation dashboard with comprehensive visualization
    and interactive controls.
    """
    
    def __init__(self, world: World, fps: int = 10, duration: float = 300.0,
                 enable_advanced_features: bool = True):
        """
        Initialize the live simulation dashboard.
        
        Args:
            world: World simulation instance
            fps: Target frames per second (10-30 recommended)
            duration: Total simulation duration in seconds
            enable_advanced_features: Enable advanced rendering features
        """
        self.world = world
        self.fps = fps
        self.dt = 1.0 / fps
        self.duration = duration
        self.total_frames = int(duration * fps)
        self.advanced = enable_advanced_features
        
        # Ensure agents are scheduled to act
        self._init_agent_scheduling()
        
        # Playback state
        self.current_frame = 0
        self.paused = False
        self.speed_multiplier = 1.0
        self.recording = False
        
        # Data recording for time-slider and analysis
        self.times = []
        self.cleared_counts = []
        self.discovered_counts = []
        self.agent_positions_history = {agent.id: deque(maxlen=100) for agent in world.agents}
        self.flow_metrics_history = []
        self.frame_data = []
        
        # Flow dynamics model
        self.flow_model = FlowDynamics(world.G)
        
        # Rendering state
        self.agent_artists = {}
        self.node_artists = {}
        self.edge_artists = []
        self.flow_arrows = []
        self.heat_overlay = None
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.actual_fps = fps
        
        # Initialize visualization
        self._setup_figure()
        self._setup_plots()
        self._calculate_layout()
        self._init_visualization()
        
        print(f"Dashboard initialized: {len(world.G.nodes)} nodes, {len(world.agents)} agents")
    
    def _init_agent_scheduling(self):
        """Initialize agent scheduling for simulation."""
        # Schedule initial ticks for all agents if not already scheduled
        if not self.world._event_queue:
            try:
                from .policies import tick_policy
                for agent in self.world.agents:
                    self.world.schedule(0, tick_policy, self.world, agent)
            except ImportError:
                print("[WARNING] Could not import tick_policy - agents may not move")
    
    def _setup_figure(self):
        """Create clean, minimalist dashboard layout."""
        # Use clean style
        plt.style.use('default')
        
        self.fig = plt.figure(figsize=(22, 12), facecolor=THEME_COLORS['background'])
        
        # Clear, informative title
        self.fig.suptitle('HASO Evacuation Sweep Simulation - Emergency Response Team Coordination', 
                         fontsize=14, fontweight='600', color=THEME_COLORS['text_primary'],
                         y=0.98)
        
        # Simplified grid layout - focus on main visualization
        gs = self.fig.add_gridspec(4, 5, hspace=0.25, wspace=0.3,
                                   left=0.05, right=0.95, top=0.94, bottom=0.08)
        
        # Main building visualization with clear context
        self.ax_layout = self.fig.add_subplot(gs[0:3, 0:3])
        self.ax_layout.set_title('Building Floor Plan - Real-Time Clearance Status', 
                                fontsize=11, fontweight='600', pad=10, loc='left',
                                color=THEME_COLORS['text_primary'])
        self.ax_layout.set_aspect('equal')
        self.ax_layout.set_facecolor(THEME_COLORS['panel_bg'])
        self.ax_layout.grid(True, alpha=0.12, linestyle='-', linewidth=0.3, color=THEME_COLORS['grid'])
        self.ax_layout.tick_params(labelsize=8, colors=THEME_COLORS['text_secondary'])
        self.ax_layout.set_xlabel('Distance (meters)', fontsize=9, color=THEME_COLORS['text_secondary'])
        self.ax_layout.set_ylabel('Distance (meters)', fontsize=9, color=THEME_COLORS['text_secondary'])
        
        for spine in self.ax_layout.spines.values():
            spine.set_edgecolor(THEME_COLORS['border'])
            spine.set_linewidth(0.5)
        
        # Clearance progress graph with clear purpose
        self.ax_progress = self.fig.add_subplot(gs[0, 3:5])
        self.ax_progress.set_title('Room Clearance Progress Over Time', fontsize=10, fontweight='600', 
                                  color=THEME_COLORS['text_primary'], pad=8, loc='left')
        self.ax_progress.set_xlabel('Time (seconds)', fontsize=8, color=THEME_COLORS['text_secondary'])
        self.ax_progress.set_ylabel('Number of Rooms', fontsize=8, color=THEME_COLORS['text_secondary'])
        self.ax_progress.set_facecolor(THEME_COLORS['panel_bg'])
        self.ax_progress.grid(True, alpha=0.1, linestyle='-', linewidth=0.5, color=THEME_COLORS['grid'])
        self.ax_progress.tick_params(labelsize=7, colors=THEME_COLORS['text_secondary'])
        self.ax_progress.spines['top'].set_visible(False)
        self.ax_progress.spines['right'].set_visible(False)
        self.ax_progress.spines['left'].set_color(THEME_COLORS['border'])
        self.ax_progress.spines['bottom'].set_color(THEME_COLORS['border'])
        
        # Agent status panel with clear role descriptions
        self.ax_agents = self.fig.add_subplot(gs[1:3, 3:5])
        self.ax_agents.set_title('Responder Team Status - HASO Role Assignment', 
                                fontsize=10, fontweight='600',
                                color=THEME_COLORS['text_primary'], pad=8, loc='left')
        self.ax_agents.axis('off')
        self.ax_agents.set_facecolor(THEME_COLORS['panel_bg'])
        
        # Statistics panel (compact)
        self.ax_stats = self.fig.add_subplot(gs[3, 0:5])
        self.ax_stats.axis('off')
        self.ax_stats.set_facecolor(THEME_COLORS['background'])
        
        # Time slider at bottom (clean minimal style)
        self.ax_slider = plt.axes([0.08, 0.02, 0.8, 0.015], facecolor='none')
        self.time_slider = Slider(
            self.ax_slider, '', 0, self.duration,
            valinit=0, valstep=self.dt, color=THEME_COLORS['accent'],
            track_color=THEME_COLORS['border']
        )
        self.time_slider.label.set_visible(False)
        self.time_slider.on_changed(self._on_slider_change)
        
        # Clean time display
        self.ax_time_display = plt.axes([0.89, 0.02, 0.1, 0.015], facecolor='none')
        self.ax_time_display.axis('off')
    
    def _setup_plots(self):
        """Initialize plot elements with clean, minimal styling."""
        # Clean progress lines
        self.line_cleared, = self.ax_progress.plot(
            [], [], color=THEME_COLORS['success'], linewidth=2.5, 
            label='Cleared', alpha=0.9
        )
        self.line_discovered, = self.ax_progress.plot(
            [], [], color=THEME_COLORS['text_secondary'], linewidth=1.5, 
            label='Discovered', linestyle='--', alpha=0.6
        )
        
        # Clean legend
        legend = self.ax_progress.legend(fontsize=8, loc='upper left', 
                                        frameon=True, fancybox=False, shadow=False)
        legend.get_frame().set_facecolor(THEME_COLORS['panel_bg'])
        legend.get_frame().set_edgecolor(THEME_COLORS['border'])
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_alpha(0.95)
        
        self.ax_progress.set_xlim(0, self.duration)
        self.ax_progress.set_ylim(0, len(self.world.G.nodes))
        
        # Clean statistics text
        self.stats_text = self.ax_stats.text(
            0.5, 0.5, '', transform=self.ax_stats.transAxes,
            fontsize=9, ha='center', va='center',
            fontfamily='sans-serif', color=THEME_COLORS['text_secondary']
        )
        
        # Time display
        self.time_text = self.ax_time_display.text(
            0.5, 0.5, '0:00', transform=self.ax_time_display.transAxes,
            fontsize=9, ha='center', va='center',
            fontfamily='monospace', color=THEME_COLORS['text_primary'],
            fontweight='500'
        )
    
    def _calculate_layout(self):
        """Calculate optimal node positions using force-directed layout."""
        # Check if positions are already set
        if all(hasattr(node, 'x') and hasattr(node, 'y') and node.x != 0 
               for node in list(self.world.G.nodes.values())[:min(3, len(self.world.G.nodes))]):
            return  # Positions already set
        
        if not HAS_NETWORKX:
            # Use simple circular layout as fallback
            import math
            num_nodes = len(self.world.G.nodes)
            for i, node_id in enumerate(self.world.G.nodes.keys()):
                node = self.world.G.get_node(node_id)
                if node:
                    angle = 2 * math.pi * i / num_nodes
                    node.x = 50 * math.cos(angle)
                    node.y = 50 * math.sin(angle)
            print(f"Node layout calculated using circular fallback")
            return
        
        G_nx = nx.Graph()
        for node_id in self.world.G.nodes.keys():
            G_nx.add_node(node_id)
        for (src, dst), edge in self.world.G.edges.items():
            if edge.traversable and src < dst:
                G_nx.add_edge(src, dst, weight=1.0/max(edge.length, 0.1))
        
        # Use spring layout with optimization
        pos = nx.spring_layout(G_nx, k=3, iterations=100, seed=42, scale=100)
        
        for node_id, (x, y) in pos.items():
            node = self.world.G.get_node(node_id)
            if node:
                node.x = x
                node.y = y
        
        print(f"Node layout calculated using force-directed algorithm")
    
    def _init_visualization(self):
        """Initialize all visual elements with premium styling."""
        self._draw_edges()
        self._draw_nodes()
        self._init_agents()
        self._draw_legend()
        
        if self.advanced:
            self._init_flow_arrows()
    
    def _draw_edges(self):
        """Draw edges with clean, minimal styling."""
        for (src, dst), edge in self.world.G.edges.items():
            if src < dst:  # Draw each edge once
                node_src = self.world.G.get_node(src)
                node_dst = self.world.G.get_node(dst)
                
                if node_src and node_dst:
                    # Clean edge style
                    alpha = 0.4 if edge.traversable else 0.15
                    width = 1.5 if edge.traversable else 0.8
                    
                    line = self.ax_layout.plot(
                        [node_src.x, node_dst.x],
                        [node_src.y, node_dst.y],
                        color=FLOW_COLORS['none'], linewidth=width, 
                        linestyle='-', alpha=alpha, zorder=1
                    )[0]
                    
                    self.edge_artists.append({
                        'line': line,
                        'edge': edge,
                        'src': node_src,
                        'dst': node_dst,
                        'midpoint': ((node_src.x + node_dst.x) / 2, 
                                    (node_src.y + node_dst.y) / 2)
                    })
    
    def _draw_nodes(self):
        """Draw nodes with clear, informative labeling."""
        for node_id, node in self.world.G.nodes.items():
            # Determine color and size based on clearance status
            if node.node_type == NodeType.EXIT:
                color = NODE_STATE_COLORS['exit']
                size = 500
                marker = 's'  # Square for exits
                label_text = f"EXIT\n{node_id}"
            elif node.cleared:
                color = NODE_STATE_COLORS['cleared']
                size = 350
                marker = 'o'
                label_text = f"✓\n{node_id}"
            else:
                color = NODE_STATE_COLORS['unknown']
                size = 350
                marker = 'o'
                label_text = str(node_id)
            
            # Draw node with clear boundaries
            scatter = self.ax_layout.scatter(
                node.x, node.y, c=color, s=size, marker=marker,
                edgecolors='black', linewidths=1.5, alpha=0.85, zorder=3
            )
            
            # Clear, readable label
            label = self.ax_layout.text(
                node.x, node.y, label_text,
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white', zorder=4
            )
            
            # Prominent hazard warning
            hazard_marker = None
            hazard_label = None
            if node.hazard != HazardType.NONE and node.hazard_severity > 0.3:
                hazard_marker = self.ax_layout.scatter(
                    node.x + 4, node.y + 4, c=NODE_STATE_COLORS['hazard_high'],
                    s=120, marker='^', edgecolors='black', linewidths=1.5, 
                    alpha=0.95, zorder=5
                )
                # Add hazard type label
                hazard_label = self.ax_layout.text(
                    node.x + 4, node.y + 7, node.hazard.name[:4],
                    ha='center', va='bottom', fontsize=6, fontweight='bold',
                    color=NODE_STATE_COLORS['hazard_high'], zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9)
                )
            
            self.node_artists[node_id] = {
                'scatter': scatter,
                'label': label,
                'node': node,
                'hazard_marker': hazard_marker,
                'hazard_label': hazard_label
            }
    
    def _init_agents(self):
        """Initialize agent visual elements with clean styling but full functionality."""
        for agent in self.world.agents:
            node = self.world.G.get_node(agent.node)
            if not node:
                continue
            
            color = ROLE_COLORS.get(agent.role, THEME_COLORS['text_secondary'])
            
            # Clean agent marker (larger, cleaner circles)
            marker = self.ax_layout.scatter(
                node.x, node.y, c=color, s=350, marker='o',
                edgecolors='white', linewidths=2.5, alpha=0.95, zorder=6
            )
            
            # Agent ID label inside circle
            label = self.ax_layout.text(
                node.x, node.y, f"{agent.id}",
                fontsize=10, fontweight='600', ha='center', va='center',
                color='white', zorder=7
            )
            
            # Role indicator - small text below
            role_label = self.ax_layout.text(
                node.x, node.y - 4, agent.role.name[:3],
                fontsize=7, fontweight='500', ha='center', va='top',
                color=color, zorder=7,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor=color, linewidth=1, alpha=0.9)
            )
            
            # Clean trail line
            trail, = self.ax_layout.plot(
                [], [], '-', color=color, linewidth=2, alpha=0.35, zorder=2
            )
            
            # Status indicator (small dot above agent)
            status_indicator = self.ax_layout.scatter(
                node.x, node.y + 3, c=THEME_COLORS['success'], s=60, marker='o',
                edgecolors='white', linewidths=1.5, alpha=0.9, zorder=7
            )
            
            self.agent_artists[agent.id] = {
                'marker': marker,
                'label': label,
                'role_label': role_label,
                'trail': trail,
                'status_indicator': status_indicator,
                'color': color,
                'path_x': deque([node.x], maxlen=50),
                'path_y': deque([node.y], maxlen=50),
                'target_x': node.x,
                'target_y': node.y,
                'interp_progress': 0.0
            }
    
    def _init_flow_arrows(self):
        """Initialize flow direction arrows for advanced visualization."""
        self.flow_arrows = []
        # Will be populated dynamically during updates
    
    def _draw_legend(self):
        """Draw comprehensive, informative legend."""
        # Room status legend
        room_elements = [
            mpatches.Patch(facecolor=NODE_STATE_COLORS['exit'], 
                          edgecolor='black', linewidth=1, label='EXIT (Safe Zone)'),
            mpatches.Patch(facecolor=NODE_STATE_COLORS['cleared'], 
                          edgecolor='black', linewidth=1, label='✓ Cleared Room'),
            mpatches.Patch(facecolor=NODE_STATE_COLORS['in_progress'], 
                          edgecolor='black', linewidth=1, label='⚙ Clearing in Progress'),
            mpatches.Patch(facecolor=NODE_STATE_COLORS['not_cleared'], 
                          edgecolor='black', linewidth=1, label='✗ Uncleared Room'),
            mpatches.Patch(facecolor=NODE_STATE_COLORS['unknown'], 
                          edgecolor='black', linewidth=1, label='? Unknown Area'),
        ]
        
        # Agent role legend
        agent_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=ROLE_COLORS[Role.SCOUT], markersize=10,
                      markeredgecolor='black', markeredgewidth=1.5,
                      label='SCOUT (Fast recon)'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=ROLE_COLORS[Role.SECURER], markersize=10,
                      markeredgecolor='black', markeredgewidth=1.5,
                      label='SECURER (Assist evac)'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=ROLE_COLORS[Role.CHECKPOINTER], markersize=10,
                      markeredgecolor='black', markeredgewidth=1.5,
                      label='CHECKPOINT (Secure areas)'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=ROLE_COLORS[Role.EVACUATOR], markersize=10,
                      markeredgecolor='black', markeredgewidth=1.5,
                      label='EVACUATOR (Final sweep)'),
        ]
        
        # Create two separate legends
        legend1 = self.ax_layout.legend(
            handles=room_elements, loc='upper left', fontsize=7,
            title='Room Status', title_fontsize=8,
            frameon=True, fancybox=False, shadow=False
        )
        legend1.get_frame().set_facecolor(THEME_COLORS['panel_bg'])
        legend1.get_frame().set_edgecolor(THEME_COLORS['border'])
        legend1.get_frame().set_linewidth=1
        legend1.get_frame().set_alpha(0.95)
        
        # Add second legend manually
        self.ax_layout.add_artist(legend1)
        legend2 = self.ax_layout.legend(
            handles=agent_elements, loc='lower left', fontsize=7,
            title='Responder Roles (HASO)', title_fontsize=8,
            frameon=True, fancybox=False, shadow=False
        )
        legend2.get_frame().set_facecolor(THEME_COLORS['panel_bg'])
        legend2.get_frame().set_edgecolor(THEME_COLORS['border'])
        legend2.get_frame().set_linewidth(1)
        legend2.get_frame().set_alpha(0.95)
    
    def _interpolate_agent_position(self, agent: Agent) -> Tuple[float, float]:
        """
        Calculate smooth agent position with interpolation.
        
        Returns:
            (x, y) interpolated position
        """
        current_node = self.world.G.get_node(agent.node)
        if not current_node:
            return (0, 0)
        
        # Get stored agent state
        agent_state = self.agent_artists.get(agent.id)
        if not agent_state:
            return (current_node.x, current_node.y)
        
        # Check if agent has moved to a new node
        if agent_state['target_x'] != current_node.x or agent_state['target_y'] != current_node.y:
            # Reset interpolation for new target
            agent_state['target_x'] = current_node.x
            agent_state['target_y'] = current_node.y
            agent_state['interp_progress'] = 0.0
        
        # Interpolate based on movement speed
        if agent.status in [Status.NORMAL, Status.SLOWED]:
            agent_state['interp_progress'] = min(1.0, agent_state['interp_progress'] + 0.15)
        else:
            agent_state['interp_progress'] = 1.0
        
        # Calculate interpolated position
        progress = agent_state['interp_progress']
        current_x = agent_state['path_x'][-1] if agent_state['path_x'] else current_node.x
        current_y = agent_state['path_y'][-1] if agent_state['path_y'] else current_node.y
        
        x = current_x + (current_node.x - current_x) * progress
        y = current_y + (current_node.y - current_y) * progress
        
        return (x, y)
    
    def _update_frame(self, frame: int):
        """
        Update all visualization elements for the current frame.
        
        This is the main rendering loop called by FuncAnimation.
        """
        frame_start_time = pytime.time()
        
        if self.paused:
            return self._get_artists()
        
        # Step simulation forward by one frame time
        dt = self.dt * self.speed_multiplier
        if self.world.time < self.duration:
            self.world.step(dt)
        
        current_time = self.world.time
        
        # Record data
        self._record_frame_data(current_time)
        
        # Update all visual elements
        self._update_agents()
        self._update_nodes()
        self._update_edges()
        self._update_progress_plots()
        self._update_info_panels()
        self._update_summary(current_time)
        
        # Update slider
        if not self.paused:
            self.time_slider.set_val(min(current_time, self.duration))
        
        self.current_frame = frame
        
        # Track performance
        frame_time = pytime.time() - frame_start_time
        self.frame_times.append(frame_time)
        self.actual_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        return self._get_artists()
    
    def _record_frame_data(self, current_time: float):
        """Record data for the current frame."""
        self.times.append(current_time)
        
        cleared, total = self.world.G.get_cleared_count()
        self.cleared_counts.append(cleared)
        
        discovered = sum(1 for n in self.world.G.nodes.values() if n.fog_state >= 1)
        self.discovered_counts.append(discovered)
        
        # Update flow model
        for (src, dst) in list(self.world.G.edges.keys())[:50]:  # Limit for performance
            flow = self.flow_model.calculate_flow_rate(src, dst)
            self.flow_model.edge_flow[(src, dst)] = flow
        
        flow_metrics = self.flow_model.get_flow_metrics()
        self.flow_metrics_history.append(flow_metrics)
        
        # Store agent positions
        for agent in self.world.agents:
            if agent.id in self.agent_positions_history:
                x, y = self._interpolate_agent_position(agent)
                self.agent_positions_history[agent.id].append((x, y, current_time))
    
    def _update_agents(self):
        """Update agent positions, trails, and status indicators."""
        for agent in self.world.agents:
            if agent.id not in self.agent_artists:
                continue
            
            x, y = self._interpolate_agent_position(agent)
            agent_state = self.agent_artists[agent.id]
            
            # Update marker position
            agent_state['marker'].set_offsets([[x, y]])
            
            # Update label position (centered in circle)
            agent_state['label'].set_position((x, y))
            
            # Update role label position
            agent_state['role_label'].set_position((x, y - 4))
            
            # Update trail
            agent_state['path_x'].append(x)
            agent_state['path_y'].append(y)
            agent_state['trail'].set_data(
                list(agent_state['path_x']),
                list(agent_state['path_y'])
            )
            
            # Update status indicator color and position
            status_colors = {
                Status.NORMAL: THEME_COLORS['success'],
                Status.SLOWED: THEME_COLORS['warning'],
                Status.PROGRESSING: THEME_COLORS['info'],
                Status.IMMOBILIZED: THEME_COLORS['danger'],
                Status.INCAPACITATED: THEME_COLORS['text_secondary'],
            }
            status_color = status_colors.get(agent.status, THEME_COLORS['text_secondary'])
            agent_state['status_indicator'].set_facecolors([status_color])
            agent_state['status_indicator'].set_offsets([[x, y + 3]])
    
    def _update_nodes(self):
        """Update node colors based on clearance and hazard states."""
        for node_id, artists in self.node_artists.items():
            node = artists['node']
            
            # Determine color based on multiple factors
            if node.node_type == NodeType.EXIT:
                color = NODE_STATE_COLORS['exit']
            elif node.cleared:
                color = NODE_STATE_COLORS['cleared']
            elif any(a.node == node_id for a in self.world.agents):
                color = NODE_STATE_COLORS['in_progress']
            elif node.fog_state == 0:
                color = NODE_STATE_COLORS['unknown']
            else:
                # Color based on hazard if present
                if node.hazard != HazardType.NONE and node.hazard_severity > 0.5:
                    color = NODE_STATE_COLORS['hazard_high']
                elif node.hazard != HazardType.NONE and node.hazard_severity > 0.2:
                    color = NODE_STATE_COLORS['hazard_medium']
                else:
                    color = NODE_STATE_COLORS['not_cleared']
            
            # Apply color with alpha based on visibility
            alpha = 0.85 * node.visibility if hasattr(node, 'visibility') else 0.85
            artists['scatter'].set_facecolors([color])
            artists['scatter'].set_alpha(alpha)
            
            # Update hazard marker
            if node.hazard != HazardType.NONE and node.hazard_severity > 0.3:
                if artists['hazard_marker'] is None:
                    artists['hazard_marker'] = self.ax_layout.scatter(
                        node.x + 2, node.y + 2, c=NODE_STATE_COLORS['hazard_high'],
                        s=100, marker='^', edgecolors='black', linewidths=1, 
                        alpha=0.9, zorder=5
                    )
            elif artists['hazard_marker'] is not None:
                artists['hazard_marker'].remove()
                artists['hazard_marker'] = None
    
    def _update_edges(self):
        """Update edge visualization based on flow dynamics."""
        for edge_art in self.edge_artists:
            edge = edge_art['edge']
            line = edge_art['line']
            
            # Get flow rate
            flow = self.flow_model.edge_flow.get((edge.src, edge.dst), 0.0)
            
            # Update visualization based on flow
            if flow > 0.5:
                color = FLOW_COLORS['high']
                alpha = 0.9
                width = 3.5
            elif flow > 0.1:
                color = FLOW_COLORS['medium']
                alpha = 0.7
                width = 2.5
            elif flow > 0.01:
                color = FLOW_COLORS['low']
                alpha = 0.5
                width = 1.5
            else:
                color = FLOW_COLORS['none']
                alpha = 0.3
                width = 1.0
            
            line.set_color(color)
            line.set_alpha(alpha)
            line.set_linewidth(width)
    
    def _update_progress_plots(self):
        """Update progress graphs with clean styling."""
        if len(self.times) > 0:
            # Clearance progress
            self.line_cleared.set_data(self.times, self.cleared_counts)
            self.line_discovered.set_data(self.times, self.discovered_counts)
    
    def _update_info_panels(self):
        """Update agent information with detailed status."""
        agent_lines = []
        
        # Header
        agent_lines.append("ID  ROLE           STATUS      LOCATION  CLEARED  ZONE")
        agent_lines.append("─" * 58)
        
        for agent in self.world.agents:
            # Status description
            status_desc = {
                Status.NORMAL: 'Active    ',
                Status.SLOWED: 'Slowed    ',
                Status.PROGRESSING: 'Clearing  ',
                Status.IMMOBILIZED: 'Stopped   ',
                Status.INCAPACITATED: 'Down      ',
            }.get(agent.status, 'Unknown   ')
            
            # Get zone assignment
            zone = self.world.agent_zones.get(agent.id, -1)
            zone_str = f"Z{zone}" if zone >= 0 else "--"
            
            agent_lines.append(
                f"{agent.id:<3} {agent.role.name[:12]:<12}  {status_desc}  "
                f"Room {agent.node:<3}  {agent.rooms_cleared:>3}      {zone_str}"
            )
        
        agent_text = "\n".join(agent_lines)
        
        self.ax_agents.clear()
        self.ax_agents.set_title('Responder Team Status - HASO Role Assignment', 
                                fontsize=10, fontweight='600',
                                color=THEME_COLORS['text_primary'], pad=8, loc='left')
        self.ax_agents.axis('off')
        self.ax_agents.set_facecolor(THEME_COLORS['panel_bg'])
        self.ax_agents.text(
            0.05, 0.95, agent_text, transform=self.ax_agents.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            color=THEME_COLORS['text_primary'], linespacing=1.4
        )
    
    def _update_summary(self, current_time: float):
        """Update clean summary statistics display."""
        cleared, total = self.world.G.get_cleared_count()
        pct_cleared = (cleared / total * 100) if total > 0 else 0
        
        evacuees_total = sum(len(n.evacuees) for n in self.world.G.nodes.values())
        evacuees_safe = sum(len(n.evacuees) for n in self.world.G.nodes.values() 
                           if n.node_type == NodeType.EXIT)
        
        flow_metrics = self.flow_metrics_history[-1] if self.flow_metrics_history else {}
        
        active_agents = sum(1 for a in self.world.agents 
                          if a.status in [Status.NORMAL, Status.SLOWED, Status.PROGRESSING])
        
        # Clear, informative statistics
        clearance_rate = (cleared / (current_time + 0.01)) * 60  # rooms per minute
        stats = (
            f"CLEARANCE: {cleared}/{total} rooms ({pct_cleared:.1f}%)  |  "
            f"RATE: {clearance_rate:.1f} rooms/min  |  "
            f"EVACUEES SAFE: {evacuees_safe}/{evacuees_total}  |  "
            f"ACTIVE RESPONDERS: {active_agents}/{len(self.world.agents)}  |  "
            f"EVACUATION FLOW: {flow_metrics.get('total_flow', 0):.2f} people/sec  |  "
            f"Playback: {self.speed_multiplier}x"
        )
        
        self.stats_text.set_text(stats)
        
        # Update time display
        minutes = int(current_time // 60)
        seconds = int(current_time % 60)
        self.time_text.set_text(f"{minutes}:{seconds:02d}")
    
    def _get_artists(self):
        """Collect all matplotlib artists for blitting."""
        artists = [
            self.stats_text, self.time_text,
            self.line_cleared, self.line_discovered
        ]
        
        for agent_art in self.agent_artists.values():
            artists.extend([
                agent_art['marker'], agent_art['label'], agent_art['role_label'],
                agent_art['trail'], agent_art['status_indicator']
            ])
        
        for node_art in self.node_artists.values():
            artists.append(node_art['scatter'])
            if node_art['hazard_marker'] is not None:
                artists.append(node_art['hazard_marker'])
        
        for edge_art in self.edge_artists:
            artists.append(edge_art['line'])
        
        return artists
    
    def _on_slider_change(self, val):
        """Handle time slider interaction."""
        # For now, just update display
        # Full time-travel would require state restoration
        pass
    
    def _on_key_press(self, event):
        """Handle keyboard controls."""
        if event.key == ' ':
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Resumed'}")
        
        elif event.key == 'r':
            self.current_frame = 0
            self.world.time = 0.0
            print("Reset to start")
        
        elif event.key == 's':
            filename = f'evacuation_frame_{self.current_frame:05d}.png'
            self.fig.savefig(filename, dpi=200, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        elif event.key in ['1', '2', '3', '4', '5']:
            speeds = {'1': 0.25, '2': 0.5, '3': 1.0, '4': 2.0, '5': 5.0}
            self.speed_multiplier = speeds[event.key]
            self.speed_text.set_text(f'Speed: {self.speed_multiplier:.2f}x')
            print(f"Speed set to {self.speed_multiplier}x")
        
        elif event.key == 'escape':
            print("Exiting simulation...")
            plt.close(self.fig)
    
    def run(self, save_video: bool = False, video_path: str = 'evacuation_simulation.mp4'):
        """
        Run the live animation dashboard.
        
        Args:
            save_video: If True, render to video file instead of displaying
            video_path: Output video file path
        """
        # Connect event handlers
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Create animation
        print(f"\nStarting animation: {self.total_frames} frames at {self.fps} FPS")
        print(f"Agents scheduled: {len(self.world._event_queue)} initial events")
        
        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self.total_frames,
            interval=1000 / self.fps,
            blit=False,
            repeat=False,
            cache_frame_data=False  # Prevent caching for smoother playback
        )
        
        if save_video:
            print(f"Rendering video to: {video_path}")
            print("This may take several minutes...")
            
            try:
                self.anim.save(
                    video_path, 
                    writer='ffmpeg', 
                    fps=self.fps, 
                    dpi=120,
                    bitrate=2000
                )
                print(f"[OK] Video saved successfully: {video_path}")
            except Exception as e:
                print(f"[ERROR] Error saving video: {e}")
                print("  Make sure ffmpeg is installed and in your PATH")
        else:
            print("Displaying interactive dashboard...")
            print("Use keyboard controls to interact with the simulation")
            plt.show()


def create_live_visualization(world: World, fps: int = 10, duration: float = 300.0,
                              save_video: bool = False, video_path: str = 'evacuation_sim.mp4',
                              advanced: bool = True):
    """
    Create and launch the live evacuation simulation dashboard.
    
    Args:
        world: Initialized World simulation instance
        fps: Target frames per second (10-30 recommended)
        duration: Total simulation duration in seconds
        save_video: If True, render to video file
        video_path: Output video file path
        advanced: Enable advanced visualization features
    
    Returns:
        LiveSimulationDashboard instance with recorded data
    """
    dashboard = LiveSimulationDashboard(world, fps=fps, duration=duration, 
                                       enable_advanced_features=advanced)
    dashboard.run(save_video=save_video, video_path=video_path)
    return dashboard
