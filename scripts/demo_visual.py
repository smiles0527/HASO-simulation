"""
Basic Visualization System - Standard evacuation simulation visualizations.

Generates building layout, clearance progress, agent paths, and dashboard.
For HASO-specific network diagrams, use haso_visualizer.py instead.
"""

print("\n" + "="*70)
print("EVACUATION SIMULATION - VISUALIZATION SYSTEM")
print("="*70 + "\n")

print("Loading simulation framework...")
from notebooks import simulate
from haso_sim import (
    plot_building_layout,
    plot_clearance_progress,
    plot_agent_paths,
    create_summary_dashboard,
    generate_summary_report
)
import matplotlib.pyplot as plt

print("Framework loaded successfully.\n")

# Run simulation
print("Running simulation (this may take 10-30 seconds)...")
print("Simulating 5 minutes of evacuation sweep...\n")

results = simulate(
    map_path="notebooks/data/office_building_simple.yaml",
    config_path="notebooks/data/config_baseline.yaml",
    tmax=300,  # 5 minutes
    seed=42,
    animate=False
)

world = results['world']
cleared, total = world.G.get_cleared_count()

print("Simulation complete.\n")
print(f"Time: {world.time:.1f}s ({world.time/60:.1f} minutes)")
print(f"Rooms cleared: {cleared}/{total} ({cleared/total*100:.1f}%)")
print()

# Generate text report
print("="*70)
report = generate_summary_report(world)
print(report)
print("="*70)
print()

# Create visualizations
print("Creating visualizations...\n")

try:
    # Create 4 separate plots
    
    # 1. Building Layout
    print("[1/4] Building layout with clearance status...")
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    plot_building_layout(world.G, ax=ax1, show_labels=True)
    ax1.set_title('Office Building Layout', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('demo_results/1_building_layout.png', dpi=150, bbox_inches='tight')
    print("      Saved: demo_results/1_building_layout.png")
    
    # 2. Clearance Progress
    print("[2/4] Clearance progress over time...")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    plot_clearance_progress(world, ax=ax2)
    ax2.set_title('Room Clearance Progress', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('demo_results/2_clearance_progress.png', dpi=150, bbox_inches='tight')
    print("      Saved: demo_results/2_clearance_progress.png")
    
    # 3. Agent Paths
    print("[3/4] Agent movement paths...")
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    plot_agent_paths(world, ax=ax3)
    ax3.set_title('Agent Movement Paths', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('demo_results/3_agent_paths.png', dpi=150, bbox_inches='tight')
    print("      Saved: demo_results/3_agent_paths.png")
    
    # 4. Complete Dashboard
    print("[4/4] Creating comprehensive dashboard...")
    create_summary_dashboard(world, save_path='demo_results/4_complete_dashboard.png')
    print("      Saved: demo_results/4_complete_dashboard.png")
    
    print("\nAll visualizations created successfully.\n")
    
    # Show plots
    print("="*70)
    print("VISUALIZATIONS GENERATED")
    print("="*70)
    print()
    print("Files saved in 'demo_results/' folder:")
    print("  1. 1_building_layout.png      - Building map with rooms")
    print("  2. 2_clearance_progress.png   - Progress over time")
    print("  3. 3_agent_paths.png           - Where agents moved")
    print("  4. 4_complete_dashboard.png    - Full summary dashboard")
    print()
    print("Opening dashboard...")
    print()
    
    # Show the dashboard
    img = plt.imread('demo_results/4_complete_dashboard.png')
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Evacuation Simulation Dashboard', 
                 fontsize=18, weight='bold', pad=20)
    plt.tight_layout()
    plt.show(block=True)
    
except Exception as e:
    print(f"Error during visualization: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
print("Visualization Complete")
print("="*70)
print()

