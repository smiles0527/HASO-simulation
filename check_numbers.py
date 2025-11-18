#!/usr/bin/env python3
"""Check if the numbers in the text match simulation results."""

from haso_sim import simulate
from haso_sim.utils import analyze_simulation_results

# Test 1: Full spatial layout (office_building_simple.yaml)
print("=" * 60)
print("Test 1: Full spatial layout (office_building_simple.yaml)")
print("=" * 60)
results = simulate(
    'notebooks/data/office_building_simple.yaml',
    'notebooks/data/config_baseline.yaml',
    tmax=300,
    seed=42,
    animate=False
)
world = results['world']
analysis = analyze_simulation_results(world)

print(f"\nText claims:")
print(f"  - 3 rooms cleared in 300s")
print(f"  - Clearance times: 97s → 1, 182s → 2, 276s → 3")
print(f"  - Total distance: 966m")

print(f"\nActual results:")
print(f"  - {analysis['rooms_cleared']} rooms cleared in {analysis['simulation_time']:.1f}s")
print(f"  - Total distance: {analysis['total_distance_traveled']:.1f}m")

# Check clearance history
if hasattr(world, 'history') and world.history:
    print(f"\nClearance events:")
    prev_cleared = 0
    for h in world.history:
        cleared = h.get('cleared_count', 0)
        time = h.get('time', 0)
        if cleared > prev_cleared:
            print(f"  {time:.1f}s → {cleared} rooms")
            prev_cleared = cleared

# Check if there's a six-room scenario
print("\n" + "=" * 60)
print("Test 2: Looking for six-room scenario")
print("=" * 60)
print("No six-room test scenario found in the codebase.")
print("The office_building_simple.yaml has 7 rooms (nodes 4-10).")

# Check documentation example
print("\n" + "=" * 60)
print("Test 3: Documentation example (HOW_TO_RUN.md)")
print("=" * 60)
print("Documentation shows example output:")
print("  - 3 rooms cleared in 300s")
print("  - Total distances: 190 + 560 + 80 + 136 = 966m")
print("This matches the text claims, but may be from a different run.")

