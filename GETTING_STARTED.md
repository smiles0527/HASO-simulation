# Getting Started with Emergency Evacuation Sweep Simulator

## Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Your First Simulation

```python
# In Python or Jupyter
from notebooks import simulate

results = simulate(
    map_path="notebooks/data/office_building_simple.yaml",
    config_path="notebooks/data/config_baseline.yaml",
    tmax=600,
    seed=42
)

print(f"Simulation completed in {results['world'].time:.1f} seconds")
print(f"All rooms cleared: {results['all_cleared']}")
```

### 3. Visualize Results

```python
from notebooks.src import create_summary_dashboard

create_summary_dashboard(results['world'])
```

## Key Concepts

### Building Maps (YAML)

Buildings are defined as graphs:
- **Nodes**: Rooms, corridors, exits
- **Edges**: Doors, hallways with distance/time weights
- **Properties**: Hazards, evacuees, priorities

Example node:
```yaml
- id: 1
  type: "SMALL_ROOM"
  x: 10
  y: 5
  name: "Office 101"
  room_priority: 3  # 1 = highest, 5 = lowest
  hazard: "SMOKE"
  hazard_severity: 0.5  # 0.0 to 1.0
```

### Responder Types

1. **Scout** - Fast explorer, tags evacuees, signals priorities
2. **Securer** - Secures hazards, assists evacuees, clears rooms
3. **Checkpointer** - Guards checkpoints, provides assistance
4. **Evacuator** - Double-checks clearance, final verification

### Configuration

```yaml
agents:
  - id: 0
    role: "SCOUT"
    node: 0              # Starting position
    sweep_mode: "right"  # "right", "left", or "corridor"
    personal_priority: 4

weights:
  room_priority: 1.5     # Weight for room priority scores
  distance: 0.5          # Weight for distance penalties
  hazard_penalty: 2.0    # Penalty for hazardous areas
```

## Common Tasks

### Load and Validate a Map

```python
from notebooks import load_map
from notebooks.src import validate_map, create_building_summary

G = load_map("notebooks/data/office_building_simple.yaml")

# Validate
issues = validate_map(G)
if issues:
    for issue in issues:
        print(issue)

# Summary
summary = create_building_summary(G)
print(f"Nodes: {summary['total_nodes']}")
print(f"Evacuees: {summary['total_evacuees']}")
```

### Visualize Building Layout

```python
from notebooks.src import plot_building_layout
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 10))
plot_building_layout(G, ax=ax, show_labels=True)
plt.show()
```

### Run Simulation and Get Report

```python
from notebooks import simulate
from notebooks.src import generate_summary_report

results = simulate(
    map_path="notebooks/data/office_building_simple.yaml",
    config_path="notebooks/data/config_baseline.yaml",
    tmax=600
)

report = generate_summary_report(results['world'])
print(report)
```

### Analyze Agent Performance

```python
from notebooks.src import analyze_agent_performance

world = results['world']
for agent in world.agents:
    perf = analyze_agent_performance(agent)
    print(f"Agent {perf['agent_id']} ({perf['role']}):")
    print(f"  Rooms cleared: {perf['rooms_cleared']}")
    print(f"  Distance traveled: {perf['distance_traveled']:.1f}m")
```

### Compare Different Strategies

```python
strategies = ['right', 'left', 'corridor']
results_dict = {}

for strategy in strategies:
    # Run simulation (modify config for each strategy)
    res = simulate(
        map_path="notebooks/data/office_building_simple.yaml",
        config_path="notebooks/data/config_baseline.yaml",
        tmax=600,
        seed=42
    )
    results_dict[strategy] = {
        'time': res['world'].time,
        'cleared': res['world'].G.get_cleared_count()[0]
    }

# Print comparison
for strategy, metrics in results_dict.items():
    print(f"{strategy}: {metrics['time']:.1f}s, {metrics['cleared']} rooms")
```

## Example Buildings Provided

1. **office_building_simple.yaml**
   - Single floor office
   - 11 nodes, 7 rooms
   - 7 evacuees
   - 1 hazard area
   - Good for quick testing

2. **hospital_wing.yaml**
   - Hospital wing with patient rooms
   - 15 nodes, 9 patient rooms
   - 13 evacuees (many need assistance)
   - 1 hazard area
   - More complex scenario

## Interactive Notebook

Open `notebooks/simulation_demo.ipynb` in Jupyter Lab for:
- Step-by-step walkthrough
- Building visualization
- Simulation execution
- Results analysis
- Comparative studies

```bash
jupyter lab
# Then open notebooks/simulation_demo.ipynb
```

## Customization

### Create Your Own Building

1. Copy an example YAML file
2. Modify nodes and edges
3. Add evacuees and hazards
4. Validate with `validate_map()`
5. Run simulation

### Modify Agent Behavior

Edit `notebooks/src/policies.py` to change:
- Decision-making logic
- Priority calculations
- Movement strategies
- Task allocation

### Adjust Simulation Parameters

In configuration YAML:
```yaml
simulation:
  tmax: 900              # Max simulation time (seconds)
  record_interval: 3.0   # Recording frequency (seconds)
  seed: 42               # Random seed for reproducibility
```

## Troubleshooting

### Import Errors

```python
import sys
sys.path.append('..')  # Add parent directory to path
```

### Missing Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Visualization Issues

```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

## Next Steps

1. ✅ Run the demo notebook
2. ✅ Try different building layouts
3. ✅ Experiment with agent configurations
4. ✅ Compare sweep strategies
5. ✅ Analyze optimization opportunities
6. ✅ Extend with custom features

## Need Help?

- Check `README.md` for full documentation
- Review `notebooks/simulation_demo.ipynb` for examples
- Examine example YAML files in `notebooks/data/`
- Read inline documentation in source files

## API Quick Reference

### Main Functions

```python
from notebooks import load_map, build_world, simulate

# Load map
G = load_map(path)

# Build world with agents
world = build_world(map_path, config_path, seed=42)

# Run complete simulation
results = simulate(map_path, config_path, tmax=600, seed=42, animate=False)
```

### Visualization Functions

```python
from notebooks.src import (
    plot_building_layout,
    plot_fog_of_war,
    plot_clearance_progress,
    plot_agent_paths,
    create_summary_dashboard,
    animate_run
)
```

### Analysis Functions

```python
from notebooks.src import (
    analyze_simulation_results,
    analyze_agent_performance,
    generate_summary_report,
    validate_map,
    create_building_summary
)
```

Happy simulating! 🚨🏢👥

