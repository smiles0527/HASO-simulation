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
from pathlib import Path
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


# Professional emergency response color scheme
THEME_COLORS = {
    'background': '#F5F7FA',
    'panel_bg': '#FFFFFF',
    'border': '#D1D9E6',
    'text_primary': '#1A202C',
    'text_secondary': '#4A5568',
    'accent': '#3182CE',
    'success': '#38A169',
    'warning': '#DD6B20',
    'danger': '#E53E3E',
    'info': '#3182CE',
    'grid': '#EDF2F7',
    'shadow': 'rgba(0,0,0,0.08)',
}

# Modern, high-contrast room colors
NODE_STATE_COLORS = {
    'exit': '#38A169',  # Vibrant green for exits
    'cleared': '#4299E1',  # Bright blue for cleared
    'in_progress': '#ED8936',  # Orange for in progress
    'not_cleared': '#FC8181',  # Light red for danger
    'corridor': '#E2E8F0',  # Light gray for corridors
    'hazard_high': '#C53030',
    'hazard_medium': '#E53E3E',
    'hazard_low': '#FC8181',
}

# Vibrant, distinct agent colors
ROLE_COLORS = {
    Role.SCOUT: '#48BB78',  # Bright green
    Role.SECURER: '#4299E1',  # Bright blue
    Role.CHECKPOINTER: '#ED8936',  # Bright orange
    Role.EVACUATOR: '#9F7AEA',  # Bright purple
}

FLOW_COLORS = {
    'high': '#3182CE',
    'medium': '#38A169',
    'low': '#A0AEC0',
    'none': '#E2E8F0',
}

FOG_COLORS = {
    0: (0.05, '#1F2933'),  # Unknown-unknown -> dark overlay
    1: (0.15, '#4B5563'),  # Known-unknown
    2: (0.12, '#FBBF24'),  # Unknown-known (information via signal)
    3: (0.00, '#FFFFFF'),  # Known-known
}


class LiveSimulationDashboard:
    """
    Premium real-time simulation dashboard with comprehensive visualization
    and interactive controls.
    """
    
    def __init__(self, world: World, fps: int = 120, duration: float = 300.0,
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
        # Clamp FPS to safe minimum
        self.fps = max(1, int(fps))
        self.dt = 1.0 / float(self.fps)
        self.duration = duration
        self.total_frames = int(duration * self.fps)
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
        self.agent_scatter = None
        self._agent_id_order = []
        self._agent_colors = None
        self.node_artists = {}
        self.edge_artists = []
        self.flow_arrows = []
        self.heat_overlay = None
        # Evacuee rendering
        self.evacuee_scatter = None
        self._evacuees = []  # evacuee state list
        self.agent_labels: Dict[int, Any] = {}
        self.reference_axes = []
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.actual_fps = self.fps
        
        # Initialize visualization
        self._setup_figure()
        self._setup_plots()
        self._calculate_layout()
        self._init_visualization()
        
        print(f"Dashboard initialized: {len(world.G.nodes)} nodes, {len(world.agents)} agents")
    
    def _init_agent_scheduling(self):
        """Initialize agent scheduling for simulation."""
        try:
            from .policies import tick_policy
        except ImportError:
            print("[WARNING] Could not import tick_policy - agents may not move")
            return
        # Avoid duplicate scheduling by checking existing tick events
        scheduled_ids = {
            getattr(evt.args[1], 'id', None)
            for evt in self.world._event_queue
            if getattr(evt.fn, '__name__', '') == 'tick_policy' and len(evt.args) >= 2
        }
        for agent in self.world.agents:
            if agent.id not in scheduled_ids:
                self.world.schedule(0, tick_policy, self.world, agent)
    
    def _setup_figure(self):
        """Create premium professional dashboard layout."""
        # Use clean style
        plt.style.use('default')
        
        self.fig = plt.figure(figsize=(20, 11), facecolor=THEME_COLORS['background'])
        # Hide toolbar where supported to remove controls
        try:
            self.fig.canvas.toolbar_visible = False  # Matplotlib >= 3.7
        except Exception:
            try:
                # Fallback for some backends
                self.fig.canvas.manager.toolbar.setVisible(False)  # type: ignore[attr-defined]
            except Exception:
                pass
        
        self.fig.suptitle('EMERGENCY EVACUATION COMMAND CENTER', 
                         fontsize=16, fontweight='700', color=THEME_COLORS['text_primary'],
                         y=0.975, family='sans-serif')
        self.fig.text(0.5, 0.955, 'HASO Multi-Agent Coordination System', 
                     ha='center', fontsize=10, color=THEME_COLORS['text_secondary'],
                     style='italic')
        
        # Simplified grid layout - focus on main visualization
        gs = self.fig.add_gridspec(4, 5, hspace=0.25, wspace=0.3,
                                   left=0.05, right=0.95, top=0.94, bottom=0.08)
        
        # Main building visualization - PREMIUM DESIGN
        self.ax_layout = self.fig.add_subplot(gs[0:3, 0:3])
        self.ax_layout.set_title('🏢 BUILDING FLOOR PLAN', 
                                fontsize=12, fontweight='700', pad=12, loc='left',
                                color=THEME_COLORS['text_primary'])
        self.ax_layout.set_aspect('equal')
        self.ax_layout.set_facecolor('#FAFBFC')  # Clean white-gray background
        # Enhanced grid for better map appearance
        self.ax_layout.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, 
                           color='#CBD5E0', zorder=0)
        self.ax_layout.tick_params(labelsize=9, colors=THEME_COLORS['text_secondary'], 
                                   width=1.5, length=4)
        self.ax_layout.set_xlabel('DISTANCE (meters)', fontsize=10, 
                                  color=THEME_COLORS['text_secondary'], fontweight='600')
        self.ax_layout.set_ylabel('DISTANCE (meters)', fontsize=10, 
                                  color=THEME_COLORS['text_secondary'], fontweight='600')
        
        # Professional borders with shadow effect
        for spine in self.ax_layout.spines.values():
            spine.set_edgecolor(THEME_COLORS['border'])
            spine.set_linewidth(2)
        
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
        
        # REMOVED: Time slider causes major lag - skip it for better performance
        
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
        self.agent_panel_text = self.ax_agents.text(
            0.05, 0.95, '', transform=self.ax_agents.transAxes,
            fontsize=9, verticalalignment='top',
            color=THEME_COLORS['text_primary'], linespacing=1.8
        )
        self.agent_panel_text.set_animated(True)
        
        # Time display
        self.time_text = self.ax_time_display.text(
            0.5, 0.5, '0:00', transform=self.ax_time_display.transAxes,
            fontsize=9, ha='center', va='center',
            fontfamily='monospace', color=THEME_COLORS['text_primary'],
            fontweight='500'
        )
        self.speed_text = self.ax_time_display.text(
            0.5, 0.2, 'Speed:1.00x', transform=self.ax_time_display.transAxes,
            fontsize=7, ha='center', va='center',
            fontfamily='monospace', color=THEME_COLORS['text_secondary']
        )
        self._load_reference_images()
    
    def _calculate_layout(self):
        """Use actual building coordinates from YAML for realistic layout."""
        # Nodes already have x,y from building YAML - just verify they're loaded
        has_coords = False
        for node_id, node in list(self.world.G.nodes.items())[:3]:
            if hasattr(node, 'x') and hasattr(node, 'y'):
                has_coords = True
                break
        
        if has_coords:
            print(f"Using building coordinates from YAML ({len(self.world.G.nodes)} rooms)")
            return
        
        # Fallback only if coordinates missing
        import math
        num_nodes = len(self.world.G.nodes)
        for i, node_id in enumerate(self.world.G.nodes.keys()):
            node = self.world.G.get_node(node_id)
            if node:
                angle = 2 * math.pi * i / num_nodes
                node.x = 50 * math.cos(angle)
                node.y = 50 * math.sin(angle)
        print(f"Using fallback circular layout")
    
    def _init_visualization(self):
        """Initialize all visual elements with premium styling."""
        self._draw_edges()
        self._draw_nodes()
        self._set_axes_limits()
        self._draw_legend()
        if self.advanced:
            self._init_flow_arrows()
        self._init_agents()
        self._init_evacuees()

    def _load_reference_images(self):
        """Display reference PNGs (1_..4_) at the bottom of the dashboard."""
        candidates = [
            Path('demo_results'),
            Path.cwd() / 'demo_results',
            Path(__file__).resolve().parents[2] / 'demo_results'
        ]
        titles = [
            ('1_building_layout.png', 'Layout'),
            ('2_clearance_progress.png', 'Progress'),
            ('3_agent_paths.png', 'Paths'),
            ('4_complete_dashboard.png', 'Dashboard')
        ]
        base = None
        for cand in candidates:
            if cand.exists():
                base = cand
                break
        if base is None:
            return
        left = 0.05
        width = 0.2
        spacing = 0.02
        bottom = 0.01
        height = 0.16
        for idx, (fname, title) in enumerate(titles):
            path = base / fname
            if not path.exists():
                continue
            try:
                img = plt.imread(path)
            except Exception:
                continue
            ax = self.fig.add_axes([
                left + idx * (width + spacing),
                bottom,
                width,
                height
            ])
            ax.imshow(img)
            ax.set_title(title, fontsize=8, color=THEME_COLORS['text_primary'])
            ax.axis('off')
            self.reference_axes.append(ax)

    def _set_axes_limits(self):
        """Set layout axes limits based on node extents for immediate visibility."""
        if not self.world.G.nodes:
            return
        xs = [n.x for n in self.world.G.nodes.values()]
        ys = [n.y for n in self.world.G.nodes.values()]
        if not xs or not ys:
            return
        pad = 5.0
        xmin, xmax = min(xs) - pad, max(xs) + pad
        ymin, ymax = min(ys) - pad, max(ys) + pad
        # Ensure non-degenerate limits
        if abs(xmax - xmin) < 1e-3:
            xmin -= 1
            xmax += 1
        if abs(ymax - ymin) < 1e-3:
            ymin -= 1
            ymax += 1
        self.ax_layout.set_xlim(xmin, xmax)
        self.ax_layout.set_ylim(ymin, ymax)
    
    def _draw_edges(self):
        """Draw edges as subtle connections (map-like)."""
        for (src, dst), edge in self.world.G.edges.items():
            if src < dst:  # Draw each edge once
                node_src = self.world.G.get_node(src)
                node_dst = self.world.G.get_node(dst)
                
                if node_src and node_dst:
                    # Subtle connection lines (like on a map)
                    alpha = 0.25 if edge.traversable else 0.1
                    width = 1.0 if edge.traversable else 0.5
                    
                    line = self.ax_layout.plot(
                        [node_src.x, node_dst.x],
                        [node_src.y, node_dst.y],
                        color='#C5CAD5', linewidth=width, 
                        linestyle='--', alpha=alpha, zorder=2
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
        """Draw nodes as clean rectangles - OPTIMIZED for performance."""
        for node_id, node in self.world.G.nodes.items():
            # Determine room size and styling based on type
            room_name = getattr(node, 'name', '') or str(node_id)
            if node.node_type == NodeType.EXIT:
                # Exits - Distinctive green squares
                width, height = 5, 5
                color = NODE_STATE_COLORS['exit']
                label_text = room_name if room_name else "EXIT"
                edgecolor = '#2F855A'
                edgewidth = 3
            elif node.node_type == NodeType.CORRIDOR:
                # Corridors - Elongated rectangles
                width, height = 7, 3
                color = NODE_STATE_COLORS['corridor']
                label_text = room_name
                edgecolor = '#A0AEC0'
                edgewidth = 2
            else:
                # Regular rooms - sized by area
                area = getattr(node, 'area', 20)
                width = height = np.sqrt(area) * 0.9
                
                if node.cleared:
                    color = NODE_STATE_COLORS['cleared']
                    label_text = f"✓ {room_name}"
                    edgecolor = '#2C5282'
                    edgewidth = 2.5
                else:
                    color = NODE_STATE_COLORS['not_cleared']
                    label_text = room_name
                    edgecolor = '#C53030'
                    edgewidth = 2.5
            
            # Draw room as simple rectangle - FAST!
            rect = Rectangle(
                (node.x - width/2, node.y - height/2), 
                width, height,
                facecolor=color, edgecolor=edgecolor, linewidth=edgewidth,
                alpha=0.9, zorder=3
            )
            self.ax_layout.add_patch(rect)
            
            # Room label
            label = self.ax_layout.text(
                node.x, node.y, label_text,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white' if node.node_type != NodeType.CORRIDOR else '#2D3748',
                zorder=4
            )
            
            # Hazard indicator (simple + severity)
            hazard_marker = None
            hazard_text = None
            if node.hazard != HazardType.NONE and node.hazard_severity > 0.3:
                hazard_circle = Circle(
                    (node.x + width/2 - 1, node.y + height/2 - 1), 
                    0.7,
                    facecolor='#F56565', edgecolor='#C53030',
                    linewidth=1.5, alpha=0.95, zorder=5
                )
                self.ax_layout.add_patch(hazard_circle)
                hazard_marker = hazard_circle
                hazard_text = self.ax_layout.text(
                    node.x + width/2 - 1, node.y + height/2 - 1,
                    f"{int(node.hazard_severity*100)}%",
                    ha='center', va='center', fontsize=6, fontweight='bold',
                    color='white', zorder=6
                )
            # Fog overlay (semi-transparent rectangle)
            fog_alpha, fog_color = FOG_COLORS.get(node.fog_state, (0.18, '#4B5563'))
            fog_overlay = Rectangle(
                (node.x - width/2, node.y - height/2),
                width, height,
                facecolor=fog_color, edgecolor='none',
                alpha=fog_alpha, zorder=4.2
            )
            self.ax_layout.add_patch(fog_overlay)
            # Occupancy label (number of evacuees inside)
            occ_count = len(node.evacuees)
            occ_label = self.ax_layout.text(
                node.x, node.y - height/2 - 0.8,
                f"Evac:{occ_count}", fontsize=6, fontweight='bold',
                ha='center', va='top', color=THEME_COLORS['text_secondary'],
                zorder=5
            )
            
            self.node_artists[node_id] = {
                'rect': rect,
                'label': label,
                'node': node,
                'hazard_marker': hazard_marker,
                'hazard_label': hazard_text,
                'fog_overlay': fog_overlay,
                'occupancy_label': occ_label,
                'width': width,
                'height': height
            }
    
    def _init_agents(self):
        """Initialize agent visual elements - supports ultra-fast scatter mode."""
        positions = []
        colors = []
        self._agent_id_order = []
        for agent in self.world.agents:
            node = self.world.G.get_node(agent.node)
            if not node:
                continue
            positions.append([node.x, node.y])
            colors.append(ROLE_COLORS.get(agent.role, THEME_COLORS['text_secondary']))
            self._agent_id_order.append(agent.id)
            self.agent_artists[agent.id] = {
                'color': colors[-1],
                'path_x': deque([node.x], maxlen=60),
                'path_y': deque([node.y], maxlen=60),
                'current_x': node.x,
                'current_y': node.y,
                'target_x': node.x,
                'target_y': node.y,
                'interp_progress': 1.0,
                'last_node': agent.node
            }
        if positions:
            pos_arr = np.array(positions)
            color_arr = np.array(colors)
        else:
            pos_arr = np.empty((0, 2))
            color_arr = np.empty((0,))
        self._agent_colors = color_arr
        self.agent_scatter = self.ax_layout.scatter(
            pos_arr[:, 0] if pos_arr.size else np.array([]),
            pos_arr[:, 1] if pos_arr.size else np.array([]),
            s=36, c=color_arr if color_arr.size else '#2563EB', marker='o', linewidths=0,
            zorder=10, animated=True
        )
        # Agent labels (ID + role)
        for agent_id, agent in zip(self._agent_id_order, self.world.agents):
            node = self.world.G.get_node(agent.node)
            if not node:
                continue
            label = self.ax_layout.text(
                node.x, node.y + 1.2,
                f"A{agent.id}:{agent.role.name[0]}", fontsize=7, fontweight='bold',
                ha='center', va='bottom', color='#111827', zorder=11
            )
            self.agent_labels[agent.id] = label
    
    def _init_flow_arrows(self):
        """Initialize flow direction arrows for advanced visualization."""
        self.flow_arrows = []
        # Will be populated dynamically during updates
    
    def _draw_legend(self):
        """Draw clean, simple legends - OPTIMIZED."""
        # Room status legend - simple rectangles
        room_elements = [
            mpatches.Rectangle((0, 0), 1, 1, 
                facecolor=NODE_STATE_COLORS['exit'], 
                edgecolor='#2F855A', linewidth=2, 
                label='EXIT'),
            mpatches.Rectangle((0, 0), 1, 1, 
                facecolor=NODE_STATE_COLORS['corridor'], 
                edgecolor='#A0AEC0', linewidth=1.5, 
                label='Corridor'),
            mpatches.Rectangle((0, 0), 1, 1,
                facecolor=NODE_STATE_COLORS['cleared'], 
                edgecolor='#2C5282', linewidth=1.5, 
                label='Cleared'),
            mpatches.Rectangle((0, 0), 1, 1,
                facecolor=NODE_STATE_COLORS['in_progress'], 
                edgecolor='#C05621', linewidth=1.5, 
                label='In Progress'),
            mpatches.Rectangle((0, 0), 1, 1,
                facecolor=NODE_STATE_COLORS['not_cleared'], 
                edgecolor='#C53030', linewidth=1.5, 
                label='Uncleared'),
        ]
        
        # Agent role legend - simple circles
        agent_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=ROLE_COLORS[Role.SCOUT], markersize=10,
                      markeredgecolor='white', markeredgewidth=2,
                      label='SCOUT'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=ROLE_COLORS[Role.SECURER], markersize=10,
                      markeredgecolor='white', markeredgewidth=2,
                      label='SECURER'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=ROLE_COLORS[Role.CHECKPOINTER], markersize=10,
                      markeredgecolor='white', markeredgewidth=2,
                      label='CHECKPOINT'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=ROLE_COLORS[Role.EVACUATOR], markersize=10,
                      markeredgecolor='white', markeredgewidth=2,
                      label='EVACUATOR'),
        ]
        
        # Create clean legends
        legend1 = self.ax_layout.legend(
            handles=room_elements, loc='upper left', fontsize=8,
            title='ROOM STATUS', title_fontsize=8,
            frameon=True, fancybox=False, shadow=False,
            framealpha=0.95, edgecolor=THEME_COLORS['border']
        )
        legend1.get_frame().set_facecolor('white')
        legend1.get_frame().set_linewidth(1.5)
        legend1.get_title().set_fontweight('bold')

        # Add second legend
        self.ax_layout.add_artist(legend1)
        legend2 = self.ax_layout.legend(
            handles=agent_elements, loc='lower left', fontsize=8,
            title='AGENT ROLES', title_fontsize=8,
            frameon=True, fancybox=False, shadow=False,
            framealpha=0.95, edgecolor=THEME_COLORS['border']
        )
        legend2.get_frame().set_facecolor('white')
        legend2.get_frame().set_linewidth(1.5)
        legend2.get_title().set_fontweight('bold')
    
    def _interpolate_agent_position(self, agent: Agent) -> Tuple[float, float]:
        """
        Calculate smooth agent position using world movement tracker.
        
        Returns:
            (x, y) interpolated position
        """
        current_node = self.world.G.get_node(agent.node)
        if not current_node:
            return (0.0, 0.0)
        
        agent_state = self.agent_artists.get(agent.id)
        if not agent_state:
            return (current_node.x, current_node.y)
        
        movement = self.world.movement_tracker.get(agent.id)
        if movement:
            start_node = self.world.G.get_node(movement.get('start_node'))
            end_node = self.world.G.get_node(movement.get('end_node'))
            if start_node and end_node:
                start_time = movement.get('start_time', self.world.time)
                end_time = movement.get('end_time', self.world.time)
                duration = max(1e-6, end_time - start_time)
                progress = (self.world.time - start_time) / duration
                progress = max(0.0, min(1.0, progress))
                x = start_node.x + (end_node.x - start_node.x) * progress
                y = start_node.y + (end_node.y - start_node.y) * progress
                agent_state['current_x'] = x
                agent_state['current_y'] = y
                agent_state['target_x'] = end_node.x
                agent_state['target_y'] = end_node.y
                agent_state['interp_progress'] = progress
                agent_state['last_node'] = agent.node
                return (x, y)
        
        # Fallback: agent stationary at current node
        agent_state['current_x'] = current_node.x
        agent_state['current_y'] = current_node.y
        agent_state['target_x'] = current_node.x
        agent_state['target_y'] = current_node.y
        agent_state['interp_progress'] = 1.0
        agent_state['last_node'] = agent.node
        return (current_node.x, current_node.y)
    
    def _nearest_exit(self, start: int) -> Optional[int]:
        exits = self.world.G.exits
        if not exits:
            return None
        # Pick closest by Euclidean distance
        return min(exits, key=lambda e: self.world.G.distance(start, e))

    def _path_any(self, start: int, goal: int) -> Optional[List[int]]:
        """Fast A* path ignoring fog; respects traversable/open edges."""
        if start == goal:
            return [start]
        from heapq import heappush, heappop
        G = self.world.G
        open_set = []
        heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: G.distance(start, goal)}
        while open_set:
            _, current = heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            for neighbor in G.neighbors(current):
                edge = G.get_edge(current, neighbor)
                if not edge or not edge.traversable or not edge.is_open:
                    continue
                # Use evac base speed 1.2 m/s for timing heuristic
                cost = edge.get_traversal_time(base_speed=1.2, hazard_modifier=self.world.get_hazard_modifier(neighbor))
                tentative = g_score[current] + cost
                if neighbor not in g_score or tentative < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + G.distance(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def _init_evacuees(self):
        """Create evacuee scatter and movement state."""
        positions = []
        colors = []
        for node in self.world.G.nodes.values():
            if not node.evacuees:
                continue
            for evac in node.evacuees:
                # Start moving by default so we always see motion
                needs_help = getattr(evac, 'needs_assistance', False)
                start_moving = True
                # Minimum visible speed; boosted when assisted
                speed = max(0.3, 1.0 * getattr(evac, 'speed_multiplier', 1.0))
                # Compute path to nearest exit
                goal = self._nearest_exit(evac.node)
                path = self._path_any(evac.node, goal) if goal is not None else [evac.node]
                state = {
                    'id': evac.id,
                    'node': evac.node,
                    'outside': getattr(evac, 'outside', False),
                    'moving': bool(start_moving),
                    'needs_help': bool(needs_help),
                    'speed': max(0.1, speed),
                    'path': path or [evac.node],  # list of node ids
                    'seg_idx': 0,                # index of current segment start in path
                    'seg_progress': 1.0,         # force initialize first segment
                    'x': node.x,
                    'y': node.y,
                }
                positions.append([node.x, node.y])
                # Color: moving evacuees = dark blue, waiting = gray
                colors.append('#2563EB' if state['moving'] else '#9CA3AF')
                self._evacuees.append(state)
        if positions:
            self.evacuee_scatter = self.ax_layout.scatter(
                np.array(positions)[:, 0], np.array(positions)[:, 1],
                s=20, c=np.array(colors), marker='o', linewidths=0, alpha=0.9, zorder=9, animated=True
            )

    def _update_evacuees(self, current_time: float) -> None:
        """Advance evacuees along paths; start evac when assisted."""
        if not self._evacuees:
            return
        positions = []
        agents_by_node = {}
        for a in self.world.agents:
            agents_by_node.setdefault(a.node, []).append(a)
        for e in self._evacuees:
            if e['outside']:
                positions.append([e['x'], e['y']])
                continue
            if not e['moving'] and e['needs_help']:
                for a in agents_by_node.get(e['node'], []):
                    if a.role in (Role.SECURER, Role.EVACUATOR):
                        e['moving'] = True
                        e['speed'] = max(e['speed'], 1.0)
                        break
            path = e['path']
            if not path or len(path) <= 1:
                positions.append([e['x'], e['y']])
                continue
            if e['seg_progress'] >= 1.0 or 'start_node' not in e or 'target_node' not in e:
                if e['seg_idx'] >= len(path) - 1:
                    last_node = self.world.G.get_node(path[-1])
                    e['x'] = last_node.x if last_node else e['x']
                    e['y'] = last_node.y if last_node else e['y']
                    e['outside'] = True
                    positions.append([e['x'], e['y']])
                    continue
                start_id = path[e['seg_idx']]
                target_id = path[e['seg_idx'] + 1]
                e['start_node'] = start_id
                e['target_node'] = target_id
                e['seg_progress'] = 0.0
                s_node = self.world.G.get_node(start_id)
                if s_node:
                    e['x'], e['y'] = s_node.x, s_node.y
            if e['moving']:
                src = self.world.G.get_node(e['start_node'])
                dst = self.world.G.get_node(e['target_node'])
                if src and dst:
                    edge = self.world.G.get_edge(src.id, dst.id)
                    hazard_mod = self.world.get_hazard_modifier(dst.id)
                    seg_time = edge.get_traversal_time(base_speed=e['speed'], hazard_modifier=hazard_mod) if edge else 1.0
                    seg_time = max(0.05, float(seg_time))
                    prog_inc = (self.dt * self.speed_multiplier) / seg_time
                    e['seg_progress'] = min(1.0, e['seg_progress'] + prog_inc)
                    e['x'] = src.x + (dst.x - src.x) * e['seg_progress']
                    e['y'] = src.y + (dst.y - src.y) * e['seg_progress']
                    if e['seg_progress'] >= 1.0:
                        e['seg_idx'] += 1
                        if e['seg_idx'] >= len(path) - 1:
                            if dst.node_type == NodeType.EXIT:
                                e['outside'] = True
                positions.append([e['x'], e['y']])
            else:
                positions.append([e['x'], e['y']])
        if self.evacuee_scatter is not None:
            offsets = np.asarray(positions) if positions else np.empty((0, 2))
            if offsets.ndim == 1:
                offsets = offsets.reshape(-1, 2)
            self.evacuee_scatter.set_offsets(offsets)
    
    def _update_frame(self, frame: int):
        """
        Update all visualization elements for the current frame - OPTIMIZED for smooth playback.
        
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
        
        self._record_frame_data(current_time)
        self._update_agents()
        self._update_evacuees(current_time)
        self._update_nodes()
        if frame % 5 == 0:
            self._update_edges()
        if frame % 10 == 0:
            self._update_progress_plots()
            self._update_info_panels()
            self._update_summary(current_time)
        
        self.current_frame = frame
        
        # Track performance (lightweight)
        frame_time = pytime.time() - frame_start_time
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        self.frame_times.append(frame_time)
        avg = (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 1e-6
        self.actual_fps = 1.0 / max(1e-6, avg)
        
        return self._get_artists()
    
    def _record_frame_data(self, current_time: float):
        """Record data for the current frame."""
        self.times.append(current_time)
        cleared, total = self.world.G.get_cleared_count()
        self.cleared_counts.append(cleared)
        discovered = sum(1 for n in self.world.G.nodes.values() if n.fog_state >= 1)
        self.discovered_counts.append(discovered)
        for (src, dst) in list(self.world.G.edges.keys())[:50]:
            flow = self.flow_model.calculate_flow_rate(src, dst)
            self.flow_model.edge_flow[(src, dst)] = flow
        flow_metrics = self.flow_model.get_flow_metrics()
        self.flow_metrics_history.append(flow_metrics)
        for agent in self.world.agents:
            if agent.id in self.agent_positions_history:
                x, y = self._interpolate_agent_position(agent)
                self.agent_positions_history[agent.id].append((x, y, current_time))
    
    def _update_agents(self):
        """Update agent positions - OPTIMIZED for maximum performance."""
        positions = []
        agent_by_id = {a.id: a for a in self.world.agents}
        for agent_id in self._agent_id_order:
            agent = agent_by_id.get(agent_id)
            if agent is None:
                state = self.agent_artists.get(agent_id)
                if state:
                    positions.append([state.get('current_x', 0.0), state.get('current_y', 0.0)])
                continue
            state = self.agent_artists[agent_id]
            x, y = self._interpolate_agent_position(agent)
            last_x = state['path_x'][-1]
            last_y = state['path_y'][-1]
            if abs(x - last_x) > 0.05 or abs(y - last_y) > 0.05:
                state['path_x'].append(x)
                state['path_y'].append(y)
            positions.append([x, y])
            label = self.agent_labels.get(agent_id)
            if label:
                role_char = agent.role.name[0]
                label.set_position((x, y + 1.2))
                label.set_text(f"A{agent.id}:{role_char}")
        if self.agent_scatter is not None:
            offsets = np.asarray(positions) if positions else np.empty((0, 2))
            if offsets.ndim == 1:
                offsets = offsets.reshape(-1, 2)
            self.agent_scatter.set_offsets(offsets)
    
    def _update_nodes(self):
        """Update node colors with smooth transitions."""
        for node_id, artists in self.node_artists.items():
            node = artists['node']
            
            # Determine color (simplified logic for speed)
            if node.node_type == NodeType.EXIT:
                continue  # Exits never change color
            elif node.node_type == NodeType.CORRIDOR:
                continue  # Corridors don't change much
            elif node.cleared and not artists.get('was_cleared'):
                # Room just cleared! Update with celebration effect
                artists['rect'].set_facecolor(NODE_STATE_COLORS['cleared'])
                artists['rect'].set_edgecolor('#2C5282')
                artists['rect'].set_linewidth(2.5)
                room_display = getattr(node, 'name', '') or str(node_id)
                artists['label'].set_text(f"✓ {room_display}")
                artists['was_cleared'] = True
            elif any(a.node == node_id for a in self.world.agents):
                # Agent currently in this room (clearing in progress)
                if not node.cleared:
                    artists['rect'].set_facecolor(NODE_STATE_COLORS['in_progress'])
                    artists['rect'].set_edgecolor('#C05621')
            elif not node.cleared and not any(a.node == node_id for a in self.world.agents):
                # Not cleared and no agents present
                if not artists.get('was_cleared'):
                    artists['rect'].set_facecolor(NODE_STATE_COLORS['not_cleared'])
                    artists['rect'].set_edgecolor('#C53030')
            # Update hazard label if present
            if node.hazard != HazardType.NONE and node.hazard_severity > 0.3:
                if artists.get('hazard_marker') is not None:
                    artists['hazard_marker'].set_visible(True)
                if artists.get('hazard_label') is not None:
                    artists['hazard_label'].set_text(f"{int(node.hazard_severity * 100)}%")
                    artists['hazard_label'].set_visible(True)
            else:
                if artists.get('hazard_marker') is not None:
                    artists['hazard_marker'].set_visible(False)
                if artists.get('hazard_label') is not None:
                    artists['hazard_label'].set_visible(False)
            # Update fog overlay
            fog_alpha, fog_color = FOG_COLORS.get(node.fog_state, (0.18, '#4B5563'))
            fog_overlay = artists.get('fog_overlay')
            if fog_overlay:
                fog_overlay.set_alpha(fog_alpha)
                fog_overlay.set_facecolor(fog_color)
            occ_label = artists.get('occupancy_label')
            if occ_label:
                occ_label.set_text(f"Evac:{len(node.evacuees)}")
    
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
        """Update agent information with detailed status - SIMPLIFIED."""
        agent_lines = []
        
        for agent in self.world.agents:
            status_icon = {
                Status.NORMAL: '●',
                Status.SLOWED: '◐',
                Status.PROGRESSING: '⚙',
                Status.IMMOBILIZED: '○',
                Status.INCAPACITATED: '✕',
            }.get(agent.status, '○')
            
            zone = self.world.agent_zones.get(agent.id, -1)
            agent_lines.append(
                f"{status_icon} A{agent.id} {agent.role.name[:4]} | Room{agent.node} | Cleared:{agent.rooms_cleared} | Z{zone}"
            )
        
        agent_text = "\n\n".join(agent_lines)
        self.agent_panel_text.set_text(agent_text)
    
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
        artists = []
        if self.agent_scatter is not None:
            artists.append(self.agent_scatter)
        if self.evacuee_scatter is not None:
            artists.append(self.evacuee_scatter)
        for edge_art in self.edge_artists:
            artists.append(edge_art['line'])
        for node_art in self.node_artists.values():
            artists.append(node_art['rect'])
            artists.append(node_art['label'])
            if node_art.get('hazard_marker') is not None:
                artists.append(node_art['hazard_marker'])
            if node_art.get('hazard_label') is not None:
                artists.append(node_art['hazard_label'])
            if node_art.get('fog_overlay') is not None:
                artists.append(node_art['fog_overlay'])
            if node_art.get('occupancy_label') is not None:
                artists.append(node_art['occupancy_label'])
        artists.extend([
            self.stats_text,
            self.time_text,
            self.speed_text,
            self.line_cleared,
            self.line_discovered,
            self.agent_panel_text,
        ])
        artists.extend(self.agent_labels.values())
        for ax in self.reference_axes:
            artists.extend(ax.get_children())
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
        # Create animation
        print(f"\nStarting animation: {self.total_frames} frames at {self.fps} FPS")
        print(f"Agents scheduled: {len(self.world._event_queue)} initial events")
        
        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self.total_frames,
            interval=1000.0 / max(1, self.fps),
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
            print("Displaying dashboard...")
            plt.show()


def create_live_visualization(world: World, fps: int = 120, duration: float = 300.0,
                              save_video: bool = False, video_path: str = 'evacuation_sim.mp4',
                              advanced: bool = True):
    """
    Create and launch the live evacuation simulation dashboard.
    
    Args:
        world: Initialized World simulation instance
        fps: Target frames per second (120 default for ultra smooth animation)
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
