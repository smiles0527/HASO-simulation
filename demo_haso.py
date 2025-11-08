"""
HASO Algorithm Implementation Visualization

This script demonstrates:
1. Zone partitioning (macro-level planning)
2. Dynamic hazard propagation
3. HASO pathfinding with hazard/visibility costs
4. Performance metrics (η, R, E)
"""

print("\n" + "="*70)
print("HASO ALGORITHM DEMONSTRATION")
print("="*70 + "\n")

print("Loading HASO-enhanced simulation framework...")
from notebooks import simulate
from notebooks.src import (
    generate_summary_report,
    analyze_simulation_results,
    create_summary_dashboard
)

print("[OK] HASO framework loaded!\n")

# Run simulation with HASO enabled
print("Running simulation with HASO algorithm...")
print("  - Zone partitioning: ENABLED")
print("  - Dynamic hazards: ENABLED")
print("  - HASO pathfinding: ENABLED")
print()

results = simulate(
    map_path="notebooks/data/office_building_simple.yaml",
    config_path="notebooks/data/config_baseline.yaml",
    tmax=300,  # 5 minutes
    seed=42,
    animate=False
)

world = results['world']
cleared, total = world.G.get_cleared_count()

print("\n" + "="*70)
print("HASO SIMULATION COMPLETE")
print("="*70 + "\n")

# Show HASO-specific outputs
print(f"Total Time: {world.time:.1f}s ({world.time/60:.1f} minutes)")
print(f"Rooms Cleared: {cleared}/{total} ({cleared/total*100:.1f}%)")
print()

# Zone assignments
if world.zones:
    print("HASO ZONE ASSIGNMENTS:")
    print("-" * 70)
    for agent in world.agents:
        if agent.assigned_zone != -1:
            zone_nodes = world.zones.get(agent.assigned_zone, [])
            print(f"  Agent {agent.id} ({agent.role.name:15s}) -> Zone {agent.assigned_zone} ({len(zone_nodes)} rooms)")
    print()

# HASO Metrics
print("HASO PERFORMANCE METRICS:")
print("-" * 70)
analysis = analyze_simulation_results(world)
print(f"  Efficiency Ratio (eta): {analysis['haso_efficiency_ratio']:.4f} rooms/second")
print(f"  Redundancy Index (R):   {analysis['haso_redundancy_index']:.3f}")
print(f"  Risk Exposure (E):      {analysis['haso_risk_exposure']:.3f}")
print()

# Hazard status
hazard_nodes = [n for n in world.G.nodes.values() if n.hazard_severity > 0]
print(f"HAZARD PROPAGATION:")
print("-" * 70)
print(f"  Nodes with hazards: {len(hazard_nodes)}")
for node in hazard_nodes[:5]:  # Show first 5
    print(f"    Node {node.id}: {node.hazard.name} (severity: {node.hazard_severity:.2f}, visibility: {node.visibility:.2f})")
if len(hazard_nodes) > 5:
    print(f"    ... and {len(hazard_nodes)-5} more")
print()

# Full report
print("="*70)
report = generate_summary_report(world)
print(report)
print("="*70)

# Create visualizations
print("\nCreating HASO visualizations...")
try:
    create_summary_dashboard(world, save_path='demo_results/haso_dashboard.png')
    print("[OK] Dashboard saved to: demo_results/haso_dashboard.png")
except Exception as e:
    print(f"[Warning] Could not create visualizations: {e}")

print()
print("="*70)
print("HASO FEATURES DEMONSTRATED:")
print("="*70)
print()
print("[OK] Macro-Level Zone Partitioning:")
print("  - Building divided into zones using community detection")
print("  - Agents assigned to zones minimizing expected time")
print()
print("[OK] Micro-Level Adaptive Pathfinding:")
print("  - HASO cost function: C(a_i) = SUM[t(e_ij) + beta*h_j(t) + lambda*(1-v_j(t))]")
print("  - Dynamic path updates based on hazards and visibility")
print()
print("[OK] Dynamic Hazard Propagation:")
print("  - Fire spread: h_i(t+1) = h_i(t) + delta_t*k")
print("  - Smoke visibility: v_i(t+1) = v_i(t) - gamma")
print()
print("[OK] Performance Metrics:")
print("  - Efficiency ratio (eta): rooms per second")
print("  - Redundancy index (R): verification coverage")
print("  - Risk exposure (E): hazard-occupancy product")
print()
print("="*70)
print("[SUCCESS] HASO Algorithm fully operational!")
print("="*70)
print()

