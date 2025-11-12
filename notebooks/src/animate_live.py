"""
Real-Time Evacuation Simulation Dashboard

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
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time as pytime
from collections import deque

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

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
    Real-time simulation dashboard with comprehensive visualization
    and interactive controls.
    """
    
    def __init__(self, world: World, fps: int = 20, duration: float = 300.0,
                 enable_advanced_features: bool = True, quiet: bool = False):
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
        self.quiet = quiet
        
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
        self._agent_index: Dict[int, int] = {}
        self._agent_positions = None
        self.node_artists = {}
        self.node_collection: Optional[PatchCollection] = None
        self.node_order: List[int] = []
        self._node_facecolors: Optional[np.ndarray] = None
        self._node_index: Dict[int, int] = {}
        self.edge_artists = []
        self.flow_arrows = []
        self.heat_overlay = None
        # Evacuee rendering
        self.evacuee_scatter = None
        self._evac_ids: List[int] = []
        self._evac_positions: Optional[np.ndarray] = None
        self._evac_colors: Optional[np.ndarray] = None
        self.reference_axes = []
        # Caches to skip redundant drawing work
        self.node_cache: Dict[int, Dict[str, Any]] = {}
        self.edge_cache: Dict[Tuple[int, int], str] = {}
        self._exit_nodes = [node for node in self.world.G.nodes.values() if node.node_type == NodeType.EXIT]
        self._room_nodes = [
            node for node in self.world.G.nodes.values()
            if node.node_type not in [NodeType.CORRIDOR, NodeType.EXIT, NodeType.CHECKPOINT]
        ]
        self._total_evacuees = sum(len(node.evacuees) for node in self.world.G.nodes.values())
        self._last_agent_panel_text: str = ""
        self._last_stats_text: str = ""
        self._last_time_text: str = ""
        self._speed_levels = [0.25, 0.5, 1.0, 2.0, 5.0]
        self._speed_index = self._speed_levels.index(1.0) if 1.0 in self._speed_levels else 0
        self._control_buttons: Dict[str, Button] = {}
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.actual_fps = self.fps
        # Update throttling
        self.metrics_interval = 10
        self.flow_update_interval = 10

        # Axis expansion
        self._expandable_axes: List[Axes] = []
        self._axis_positions: Dict[Axes, Any] = {}
        self._expanded_axis: Optional[Axes] = None
        
        # Initialize visualization
        self._setup_figure()
        self._setup_plots()
        self._calculate_layout()
        self._register_expandable_axes()
        self._init_visualization()
        # Connect keyboard controls for responsiveness
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', self._on_axes_click)
        # Draw static background once before animation for blitting
        self.fig.canvas.draw()
        # Ensure all agents have an initial tick scheduled so movement begins immediately
        try:
            from .policies import tick_policy
            for agent in self.world.agents:
                self.world.schedule(0, tick_policy, self.world, agent)
        except ImportError:
            pass
        
        if not self.quiet:
            print(f"Dashboard initialized: {len(world.G.nodes)} nodes, {len(world.agents)} agents")
    
    def _init_agent_scheduling(self):
        """Initialize agent scheduling for simulation."""
        try:
            from .policies import tick_policy
        except ImportError:
            if not self.quiet:
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
        
        self.fig = plt.figure(figsize=(22, 12), facecolor=THEME_COLORS['background'])
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
        gs = self.fig.add_gridspec(
            4, 5,
            height_ratios=[1.25, 1.25, 1.15, 0.85],
            width_ratios=[3.6, 3.6, 3.6, 2.5, 2.5],
            hspace=0.32,
            wspace=0.34,
            left=0.035,
            right=0.97,
            top=0.94,
            bottom=0.06,
        )
        
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
        self.line_cleared.set_animated(False)
        self.line_discovered, = self.ax_progress.plot(
            [], [], color=THEME_COLORS['text_secondary'], linewidth=1.5, 
            label='Discovered', linestyle='--', alpha=0.6
        )
        self.line_discovered.set_animated(False)
        
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
        self.stats_text.set_animated(False)
        self.agent_panel_text = self.ax_agents.text(
            0.05, 0.95, '', transform=self.ax_agents.transAxes,
            fontsize=9, verticalalignment='top',
            color=THEME_COLORS['text_primary'], linespacing=1.8
        )
        self.agent_panel_text.set_animated(False)
        
        # Time display
        self.time_text = self.ax_time_display.text(
            0.5, 0.5, '0:00', transform=self.ax_time_display.transAxes,
            fontsize=9, ha='center', va='center',
            fontfamily='monospace', color=THEME_COLORS['text_primary'],
            fontweight='500'
        )
        self.time_text.set_animated(False)
        self.speed_text = self.ax_time_display.text(
            0.5, 0.2, 'Speed:1.00x', transform=self.ax_time_display.transAxes,
            fontsize=7, ha='center', va='center',
            fontfamily='monospace', color=THEME_COLORS['text_secondary']
        )
        self.speed_text.set_animated(False)
        self._load_reference_images()
        self._setup_controls()
    
    def _register_expandable_axes(self) -> None:
        """Record default axis positions for expansion toggling."""
        self._expandable_axes = [self.ax_layout, self.ax_progress, self.ax_agents]
        self._axis_positions = {ax: ax.get_position().frozen() for ax in self._expandable_axes}
        self._expanded_axis = None
    
    def _calculate_layout(self):
        """Use actual building coordinates from YAML for realistic layout."""
        # Nodes already have x,y from building YAML - just verify they're loaded
        has_coords = False
        for node_id, node in list(self.world.G.nodes.items())[:3]:
            if hasattr(node, 'x') and hasattr(node, 'y'):
                has_coords = True
                break
        
        if has_coords:
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
        return
    
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
        left = 0.045
        width = 0.225
        spacing = 0.02
        bottom = 0.02
        height = 0.26
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
            ax.set_facecolor(THEME_COLORS['panel_bg'])
            for spine in ax.spines.values():
                spine.set_edgecolor(THEME_COLORS['border'])
                spine.set_linewidth(1.5)
            ax.axis('off')
            self.reference_axes.append(ax)

    def _setup_controls(self) -> None:
        """Create button controls for improved usability."""
        btn_w = 0.09
        btn_h = 0.045
        spacing = 0.012
        left = 0.05
        bottom = 0.005

        def add_button(key: str, label: str, callback):
            nonlocal left
            ax = self.fig.add_axes([left, bottom, btn_w, btn_h])
            button = Button(ax, label, color='#E2E8F0', hovercolor='#CBD5E0')
            button.on_clicked(callback)
            self._control_buttons[key] = button
            left += btn_w + spacing

        add_button('play', 'Pause', self._on_btn_toggle_pause)
        add_button('reset', 'Reset', self._on_btn_reset)
        add_button('slower', 'Slower', self._on_btn_slower)
        add_button('faster', 'Faster', self._on_btn_faster)
        add_button('snapshot', 'Snapshot', self._on_btn_snapshot)
        add_button('quit', 'Exit', self._on_btn_exit)

        self._refresh_control_states()

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
    
    def _determine_node_color_key(self, node: Node, agent_present: bool) -> str:
        if node.node_type == NodeType.EXIT:
            return 'exit'
        if node.node_type == NodeType.CORRIDOR:
            return 'corridor'
        if node.cleared:
            return 'cleared'
        if agent_present:
            return 'in_progress'
        return 'not_cleared'

    def _draw_nodes(self):
        """Draw nodes as clean rectangles - OPTIMIZED for performance."""
        patches = []
        facecolors = []
        self.node_order = []
        self._node_index = {}
        for idx, (node_id, node) in enumerate(self.world.G.nodes.items()):
            room_name = getattr(node, 'name', '') or str(node_id)
            if node.node_type == NodeType.EXIT:
                width, height = 5, 5
                color_key = 'exit'
                label_text = room_name if room_name else "EXIT"
            elif node.node_type == NodeType.CORRIDOR:
                width, height = 7, 3
                color_key = 'corridor'
                label_text = room_name
            else:
                area = getattr(node, 'area', 20)
                width = height = np.sqrt(area) * 0.9
                color_key = 'cleared' if node.cleared else 'not_cleared'
                label_text = room_name

            rect = Rectangle(
                (node.x - width/2, node.y - height/2),
                width, height
            )
            patches.append(rect)
            facecolors.append(NODE_STATE_COLORS[color_key])
            label = self.ax_layout.text(
                node.x, node.y, label_text,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white' if node.node_type != NodeType.CORRIDOR else '#2D3748',
                zorder=4
            )
            self.node_artists[node_id] = {
                'label': label,
                'node': node,
                'width': width,
                'height': height,
                'color_key': color_key
            }
            self.node_cache[node_id] = {
                'color_key': color_key
            }
            self.node_order.append(node_id)
            self._node_index[node_id] = idx

        if patches:
            self.node_collection = PatchCollection(
                patches,
                facecolors=[mcolors.to_rgba(c) for c in facecolors],
                edgecolors=THEME_COLORS['border'],
                linewidths=1.5,
                alpha=0.9,
                zorder=3
            )
            self.ax_layout.add_collection(self.node_collection)
            self._node_facecolors = self.node_collection.get_facecolors()
    
    def _init_agents(self):
        """Initialize agent visual elements - supports ultra-fast scatter mode."""
        positions = []
        colors = []
        self._agent_id_order = []
        for idx, agent in enumerate(self.world.agents):
            node = self.world.G.get_node(agent.node)
            if not node:
                continue
            positions.append([node.x, node.y])
            colors.append(ROLE_COLORS.get(agent.role, THEME_COLORS['text_secondary']))
            self._agent_id_order.append(agent.id)
            self._agent_index[agent.id] = len(self._agent_id_order) - 1
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
        self._agent_positions = np.array(pos_arr) if pos_arr.size else np.empty((0, 2))
        self.agent_scatter = self.ax_layout.scatter(
            pos_arr[:, 0] if pos_arr.size else np.array([]),
            pos_arr[:, 1] if pos_arr.size else np.array([]),
            s=36, c=color_arr if color_arr.size else '#2563EB', marker='o', linewidths=0,
            zorder=10, animated=True
        )
        
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
    
    def _expand_axis(self, axis: Axes) -> None:
        """Expand a single axis to fill the canvas."""
        if self._expanded_axis is axis:
            return
        for other in self._expandable_axes:
            if other is axis:
                continue
            other.set_visible(False)
        axis.set_visible(True)
        axis.set_position([0.08, 0.12, 0.86, 0.78])
        self._expanded_axis = axis
        self.fig.canvas.draw_idle()

    def _restore_axes(self) -> None:
        """Restore all axes to their original layout."""
        if self._expanded_axis is None:
            return
        for axis in self._expandable_axes:
            pos = self._axis_positions.get(axis)
            if pos is not None:
                axis.set_position(pos)
            axis.set_visible(True)
        self._expanded_axis = None
        self.fig.canvas.draw_idle()

    def _on_axes_click(self, event) -> None:
        """Toggle axis expansion when clicking on a panel."""
        if event.button != 1:
            return
        axis = event.inaxes
        if axis is None or axis not in self._expandable_axes:
            if self._expanded_axis is not None:
                self._restore_axes()
            return
        if self._expanded_axis is axis:
            self._restore_axes()
        else:
            self._expand_axis(axis)
    
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
    
    def _init_evacuees(self):
        """Initialize evacuee scatter based on world state."""
        states = self.world.get_evacuee_states()
        self._evac_ids = [s["id"] for s in states]

        if not states:
            self._evac_positions = None
            self._evac_colors = None
            if self.evacuee_scatter is not None:
                self.evacuee_scatter.remove()
            self.evacuee_scatter = None
            return

        positions = np.array([[s["x"], s["y"]] for s in states], dtype=float)
        colors = np.array([self._color_for_evac_state(s) for s in states])

        self._evac_positions = positions
        self._evac_colors = colors

        self.evacuee_scatter = self.ax_layout.scatter(
            positions[:, 0],
            positions[:, 1],
            s=20,
            c=colors,
            marker='o',
            linewidths=0,
            alpha=0.9,
            zorder=9,
            animated=True,
        )

    def _color_for_evac_state(self, state: Dict[str, Any]) -> str:
        """Determine color for evacuee based on movement state."""
        if state.get("outside"):
            return THEME_COLORS['success']
        return '#2563EB' if state.get("moving") else '#9CA3AF'

    def _update_evacuees(self, current_time: float) -> None:
        """Update evacuee scatter positions from world state."""
        states = self.world.get_evacuee_states()

        if not states:
            if self.evacuee_scatter is not None:
                self.evacuee_scatter.set_offsets(np.empty((0, 2)))
            self._evac_positions = None
            self._evac_colors = None
            self._evac_ids = []
            return

        if self.evacuee_scatter is None or len(states) != len(self._evac_ids):
            self._init_evacuees()
            return

        if self._evac_positions is None or self._evac_positions.shape[0] != len(states):
            self._evac_positions = np.zeros((len(states), 2))
        if self._evac_colors is None or len(self._evac_colors) != len(states):
            self._evac_colors = np.empty(len(states), dtype=object)

        for idx, state in enumerate(states):
            self._evac_positions[idx, 0] = state["x"]
            self._evac_positions[idx, 1] = state["y"]
            self._evac_colors[idx] = self._color_for_evac_state(state)

        if self.evacuee_scatter is not None:
            self.evacuee_scatter.set_offsets(self._evac_positions)
            self.evacuee_scatter.set_color(self._evac_colors.tolist())
    
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
        if self.current_frame % self.metrics_interval != 0:
            return
        self.times.append(current_time)
        cleared, total = self.world.G.get_cleared_count()
        self.cleared_counts.append(cleared)
        discovered = sum(1 for n in self.world.G.nodes.values() if n.fog_state >= 1)
        self.discovered_counts.append(discovered)
        if self.current_frame % self.flow_update_interval == 0:
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
        if self._agent_positions is None:
            return
        agent_map = {a.id: a for a in self.world.agents}
        for agent_id in self._agent_id_order:
            idx = self._agent_index.get(agent_id)
            if idx is None:
                continue
            agent = agent_map.get(agent_id)
            if agent is None:
                continue
            state = self.agent_artists[agent_id]
            x, y = self._interpolate_agent_position(agent)
            last_x = state['path_x'][-1]
            last_y = state['path_y'][-1]
            if abs(x - last_x) > 0.05 or abs(y - last_y) > 0.05:
                state['path_x'].append(x)
                state['path_y'].append(y)
            self._agent_positions[idx, 0] = x
            self._agent_positions[idx, 1] = y
        if self.agent_scatter is not None and self._agent_positions.size:
            self.agent_scatter.set_offsets(self._agent_positions)
    
    def _update_nodes(self):
        """Update node colors with smooth transitions."""
        if self.node_collection is None or self._node_facecolors is None:
            return
        agent_nodes = {a.node for a in self.world.agents}
        facecolors = self._node_facecolors
        dirty = False
        for node_id, artists in self.node_artists.items():
            node = artists['node']
            if node is None:
                continue
            agent_present = node_id in agent_nodes
            new_key = self._determine_node_color_key(node, agent_present)
            cached = self.node_cache.get(node_id, {}).get('color_key')
            if cached == new_key:
                continue
            self.node_cache[node_id]['color_key'] = new_key
            idx = self._node_index.get(node_id)
            if idx is None or idx >= len(facecolors):
                continue
            facecolors[idx] = mcolors.to_rgba(NODE_STATE_COLORS[new_key])
            dirty = True
            room_display = getattr(node, 'name', '') or str(node_id)
            if node.node_type != NodeType.CORRIDOR:
                artists['label'].set_text(f"✓ {room_display}" if new_key == 'cleared' else room_display)
        if dirty:
            self.node_collection.set_facecolors(facecolors)
    
    def _update_edges(self):
        """Update edge visualization based on flow dynamics."""
        for edge_art in self.edge_artists:
            edge = edge_art['edge']
            line = edge_art['line']
            
            # Get flow rate
            flow = self.flow_model.edge_flow.get((edge.src, edge.dst), 0.0)
            
            # Update visualization based on flow
            if flow > 0.5:
                bucket = 'high'
            elif flow > 0.1:
                bucket = 'medium'
            elif flow > 0.01:
                bucket = 'low'
            else:
                bucket = 'none'
            if self.edge_cache.get((edge.src, edge.dst)) == bucket:
                continue
            self.edge_cache[(edge.src, edge.dst)] = bucket
            color = FLOW_COLORS[bucket]
            alpha = {'high':0.9,'medium':0.7,'low':0.5,'none':0.3}[bucket]
            width = {'high':3.5,'medium':2.5,'low':1.5,'none':1.0}[bucket]
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
        
        agent_text = "\n".join(agent_lines)
        if agent_text != self._last_agent_panel_text:
            self.agent_panel_text.set_text(agent_text)
            self._last_agent_panel_text = agent_text
    
    def _update_summary(self, current_time: float):
        """Update clean summary statistics display."""
        total_rooms = len(self._room_nodes)
        cleared_rooms = sum(1 for node in self._room_nodes if node.cleared)
        pct_cleared = (cleared_rooms / total_rooms * 100) if total_rooms > 0 else 0.0
        
        evac_summary = self.world.get_evacuee_summary()
        evacuees_safe = evac_summary.get("safe", 0)
        total_evacuees = evac_summary.get("total", 0)
        evac_moving = evac_summary.get("moving", 0)

        flow_metrics = self.flow_metrics_history[-1] if self.flow_metrics_history else {}
        flow_value = flow_metrics.get('total_flow', 0.0) if isinstance(flow_metrics, dict) else 0.0
        
        active_agents = sum(
            1 for agent in self.world.agents
            if agent.status in [Status.NORMAL, Status.SLOWED, Status.PROGRESSING]
        )
        
        clearance_rate = 0.0
        if current_time > 1e-3 and cleared_rooms > 0:
            clearance_rate = (cleared_rooms / current_time) * 60.0
        
        stats_lines = [
            (
                f"CLEARANCE {cleared_rooms}/{total_rooms} ({pct_cleared:.1f}%)   "
                f"RESPONDERS {active_agents}/{len(self.world.agents)}   "
                f"EVAC SAFE {evacuees_safe}/{total_evacuees}   MOVING {evac_moving}"
            ),
            (
                f"RATE {clearance_rate:.1f} rooms/min   "
                f"FLOW {flow_value:.2f} people/s   "
                f"Playback {self.speed_multiplier:.2f}x"
            ),
        ]
        stats_text = "\n".join(stats_lines)
        if stats_text != self._last_stats_text:
            self.stats_text.set_text(stats_text)
            self._last_stats_text = stats_text
        
        # Update time display
        minutes = int(current_time // 60)
        seconds = int(current_time % 60)
        time_label = f"{minutes}:{seconds:02d}"
        if time_label != self._last_time_text:
            self.time_text.set_text(time_label)
            self._last_time_text = time_label
    
    def _get_artists(self):
        """Collect all matplotlib artists for blitting."""
        artists = []
        if self.agent_scatter is not None:
            artists.append(self.agent_scatter)
        if self.evacuee_scatter is not None:
            artists.append(self.evacuee_scatter)
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
            if not self.quiet:
                print(f"{'Paused' if self.paused else 'Resumed'}")
            self._refresh_control_states()
        
        elif event.key == 'r':
            self._reset_simulation()
        
        elif event.key == 's':
            self._save_snapshot()
        
        elif event.key in ['1', '2', '3', '4', '5']:
            speeds = {'1': 0.25, '2': 0.5, '3': 1.0, '4': 2.0, '5': 5.0}
            self._set_speed_multiplier(speeds[event.key])
        
        elif event.key == 'escape':
            if not self.quiet:
                print("Exiting simulation...")
            plt.close(self.fig)
    
    def run(self, save_video: bool = False, video_path: str = 'evacuation_simulation.mp4'):
        """
        Run the live animation dashboard.
        
        Args:
            save_video: If True, render to video file instead of displaying
            video_path: Output video file path
        """
        if not self.quiet:
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
            if not self.quiet:
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
                if not self.quiet:
                    print(f"[OK] Video saved successfully: {video_path}")
            except Exception as e:
                if not self.quiet:
                    print(f"[ERROR] Error saving video: {e}")
                    print("  Make sure ffmpeg is installed and in your PATH")
        else:
            if not self.quiet:
                print("Displaying dashboard...")
            plt.show()

    def _init_blit_artists(self):
        """Deprecated: retained for compatibility."""
        return []

    # ------------------------------------------------------------------
    # Button callbacks and helpers (usability controls)
    # ------------------------------------------------------------------

    def _on_btn_toggle_pause(self, _event=None) -> None:
        self.paused = not self.paused
        if not self.quiet:
            print('Paused' if self.paused else 'Resumed')
        self._refresh_control_states()

    def _on_btn_reset(self, _event=None) -> None:
        self._reset_simulation()

    def _on_btn_slower(self, _event=None) -> None:
        if self._speed_index > 0:
            self._speed_index -= 1
            self._set_speed_multiplier(self._speed_levels[self._speed_index])

    def _on_btn_faster(self, _event=None) -> None:
        if self._speed_index < len(self._speed_levels) - 1:
            self._speed_index += 1
            self._set_speed_multiplier(self._speed_levels[self._speed_index])

    def _on_btn_snapshot(self, _event=None) -> None:
        self._save_snapshot()

    def _on_btn_exit(self, _event=None) -> None:
        if not self.quiet:
            print("Exiting simulation...")
        plt.close(self.fig)

    def _reset_simulation(self) -> None:
        self.current_frame = 0
        self.world.time = 0.0
        for agent in self.world.agents:
            agent.clear_busy(0.0)
        if not self.quiet:
            print("Reset to start")
        self._refresh_control_states()

    def _save_snapshot(self) -> None:
        filename = f'evacuation_frame_{self.current_frame:05d}.png'
        self.fig.savefig(filename, dpi=200, bbox_inches='tight')
        if not self.quiet:
            print(f"Saved: {filename}")

    def _set_speed_multiplier(self, multiplier: float) -> None:
        multiplier = max(0.05, float(multiplier))
        if multiplier in self._speed_levels:
            self._speed_index = self._speed_levels.index(multiplier)
        else:
            diffs = [abs(multiplier - lvl) for lvl in self._speed_levels]
            self._speed_index = diffs.index(min(diffs))
            multiplier = self._speed_levels[self._speed_index]
        self.speed_multiplier = multiplier
        self.speed_text.set_text(f'Speed:{multiplier:.2f}x')
        if not self.quiet:
            print(f"Speed set to {multiplier}x")
        self._refresh_control_states()

    def _refresh_control_states(self) -> None:
        play_label = 'Play' if self.paused else 'Pause'
        btn_play = self._control_buttons.get('play')
        if btn_play:
            btn_play.label.set_text(play_label)
        self._update_button_enabled('slower', self._speed_index > 0)
        self._update_button_enabled('faster', self._speed_index < len(self._speed_levels) - 1)
        self.fig.canvas.draw_idle()

    def _update_button_enabled(self, key: str, enabled: bool) -> None:
        button = self._control_buttons.get(key)
        if not button:
            return
        facecolor = '#E2E8F0' if enabled else '#F7FAFC'
        button.ax.set_facecolor(facecolor)
        button.label.set_alpha(1.0 if enabled else 0.4)
        button.hovercolor = '#CBD5E0' if enabled else facecolor
        button.eventson = enabled


def create_live_visualization(world: World, fps: int = 20, duration: float = 300.0,
                              save_video: bool = False, video_path: str = 'evacuation_sim.mp4',
                              advanced: bool = True, quiet: bool = False):
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
                                       enable_advanced_features=advanced,
                                       quiet=quiet)
    dashboard.run(save_video=save_video, video_path=video_path)
    return dashboard
