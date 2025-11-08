"""
HiMCM_EvacSim: lightweight agent-based evacuation simulator with clear visuals.

Public API:
- load_map(path) -> Graph
- build_world(map_path, config_path=None, *, seed=0) -> World
- simulate(map_path, config_path=None, *, tmax=1200, seed=0, animate=False, out_path=None) -> dict
- Key classes/enums re-exported: World, Agent, Role, Status, Graph, Node, Edge
"""

from __future__ import annotations

# ---- Versioning & metadata ----------------------------------------------------
__title__ = "HiMCM_EvacSim"
__version__ = "0.1.0"
__author__ = "Team HiMCM"
__all__ = [
    "load_map",
    "build_world",
    "simulate",
    # re-exports
    "World", "Agent", "Role", "Status",
    "Graph", "Node", "Edge",
]

# ---- Re-exports (thin, intentional surface) ----------------------------------
# Graph & map I/O
from .src.graph_model import Graph, Node, Edge, load_map_yaml as load_map  # load_map(path) -> Graph

# Core simulation
from .src.world import World
from .src.agents import Agent, Role, Status
from .src.policies import make_default_policies  # expected to return a dict role->callable
from .src.utils import load_config  # optional; returns dict or {} if None

# Optional visualization (soft import)
try:
    from .src.visualize import animate_run  # animate_run(world, history, out_path=None)
except Exception:  # pragma: no cover
    animate_run = None

# ---- Convenience builders -----------------------------------------------------
def build_world(map_path: str, config_path: str | None = None, *, seed: int = 0) -> World:
    """
    Load a map and config, then construct a World with default agents and policies.

    Assumptions:
    - load_config returns a dict with keys like:
        agents: list of dicts -> {role: "SCOUT"/... , node: int, sweep_mode: "right"/"left"/"corridor", personal_priority: int}
        weights: dict used by scoring
        known_nodes: list[int] initial KU nodes (corridors etc.)
    - make_default_policies(world) binds role policies.
    """
    G = load_map(map_path)
    cfg = load_config(config_path) if config_path else {}

    # Build agents from config (fallback to a simple 4-agent team at node 0)
    agents_cfg = cfg.get("agents", [])
    agents: list[Agent] = []
    if not agents_cfg:
        agents = [
            Agent(id=0, role=Role.SCOUT,       node=0, sweep_mode="right", personal_priority=4),
            Agent(id=1, role=Role.SECURER,     node=0, sweep_mode="right", personal_priority=3),
            Agent(id=2, role=Role.CHECKPOINTER,node=0),
            Agent(id=3, role=Role.EVACUATOR,   node=0),
        ]
    else:
        role_map = {
            "SCOUT": Role.SCOUT, "SECURER": Role.SECURER,
            "CHECKPOINTER": Role.CHECKPOINTER, "EVACUATOR": Role.EVACUATOR
        }
        for i, ac in enumerate(agents_cfg):
            agents.append(
                Agent(
                    id=ac.get("id", i),
                    role=role_map.get(ac.get("role", "SCOUT").upper(), Role.SCOUT),
                    node=int(ac.get("node", 0)),
                    sweep_mode=ac.get("sweep_mode", "right"),
                    personal_priority=int(ac.get("personal_priority", 3)),
                )
            )

    world = World(G=G, agents=agents)
    # Optional weights for scoring/priorities
    if "weights" in cfg and isinstance(cfg["weights"], dict):
        world.weights.update(cfg["weights"])

    # Initialize fog-of-war
    known_nodes = cfg.get("known_nodes", [0])
    world.init_fog(known_nodes=known_nodes)

    # Attach policies (expected to schedule first ticks externally)
    world.policies = make_default_policies(world)  # type: ignore[attr-defined]

    return world


# ---- One-liner simulate() for demos/notebooks --------------------------------
def simulate(
    map_path: str,
    config_path: str | None = None,
    *,
    tmax: float = 1200.0,
    seed: int = 0,
    animate: bool = False,
    out_path: str | None = None,
) -> dict:
    """
    Build a world, schedule agents' first ticks, run the simulation, and (optionally) animate.

    Returns a small dict with useful outputs:
    {
        "world": World,
        "history": history,           # optional time series your runner/World may record
        "all_cleared": bool,
        "agent_logs": {id: [str, ...]},
    }

    Notes:
    - This function assumes World exposes:
        - schedule(dt, fn, *args)
        - run(tmax)
        - (optional) history or a method to fetch frame-by-frame snapshots for animation
    - Policies are expected to be triggered by scheduling initial ticks here.
    """
    import random
    random.seed(seed)

    world = build_world(map_path, config_path=config_path, seed=seed)

    # Kick off policies: each agent gets a zero-delay tick
    if hasattr(world, "agents") and hasattr(world, "schedule"):
        from .src.policies import tick_policy  # local import to avoid polluting namespace
        for a in world.agents:
            world.schedule(0, tick_policy, world, a)

    # Run the simulation
    world.run(tmax=tmax)

    # Collect logs & simple outputs
    agent_logs = {}
    for a in getattr(world, "agents", []):
        agent_logs[a.id] = list(getattr(a, "log", []))[:]

    # all_cleared flag (if world.cleared exists)
    all_cleared = False
    if hasattr(world, "cleared") and isinstance(world.cleared, dict):
        all_cleared = all(world.cleared.values()) if world.cleared else False

    # Try to animate if requested and available
    history = getattr(world, "history", None)
    if animate:
        if animate_run is None:
            print("[HiMCM_EvacSim] Visualization not available: animate_run() not imported.")
        else:
            try:
                animate_run(world, history, out_path=out_path)
            except Exception as e:  # pragma: no cover
                print(f"[HiMCM_EvacSim] Animation failed: {e}")

    return {
        "world": world,
        "history": history,
        "all_cleared": all_cleared,
        "agent_logs": agent_logs,
    }
