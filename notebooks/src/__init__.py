"""
HiMCM Emergency Evacuation Sweep Simulator

Agent-based simulation framework for optimizing emergency evacuation sweeps
in multi-floor buildings using graph theory and discrete-event simulation.
"""

__version__ = "0.1.0"

# Re-export main API from parent notebooks package
# This allows: from notebooks.src import Graph, Agent, etc.

from .graph_model import (
    Graph,
    Node,
    Edge,
    NodeType,
    EdgeType,
    HazardType,
    Evacuee,
    load_map_yaml,
)

from .agents import (
    Agent,
    Role,
    Status,
    Action,
    SweepMode,
)

from .world import (
    World,
    FogOfWar,
    Event,
)

from .policies import (
    make_default_policies,
    tick_policy,
    scout_policy,
    securer_policy,
    checkpointer_policy,
    evacuator_policy,
)

from .utils import (
    load_config,
    save_results,
    analyze_agent_performance,
    analyze_simulation_results,
    generate_summary_report,
    validate_map,
    create_building_summary,
)

from .visualize import (
    plot_building_layout,
    plot_fog_of_war,
    plot_clearance_progress,
    plot_agent_paths,
    animate_run,
    create_summary_dashboard,
)

__all__ = [
    # Graph model
    "Graph",
    "Node",
    "Edge",
    "NodeType",
    "EdgeType",
    "HazardType",
    "Evacuee",
    "load_map_yaml",
    # Agents
    "Agent",
    "Role",
    "Status",
    "Action",
    "SweepMode",
    # World
    "World",
    "FogOfWar",
    "Event",
    # Policies
    "make_default_policies",
    "tick_policy",
    "scout_policy",
    "securer_policy",
    "checkpointer_policy",
    "evacuator_policy",
    # Utils
    "load_config",
    "save_results",
    "analyze_agent_performance",
    "analyze_simulation_results",
    "generate_summary_report",
    "validate_map",
    "create_building_summary",
    # Visualize
    "plot_building_layout",
    "plot_fog_of_war",
    "plot_clearance_progress",
    "plot_agent_paths",
    "animate_run",
    "create_summary_dashboard",
]

