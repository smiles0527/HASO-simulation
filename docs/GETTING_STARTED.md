# Getting Started with HASO Emergency Evacuation Simulator

This guide provides a concise introduction to running your first simulations, understanding key concepts, and customizing the simulator for your research needs.

## Quick Setup

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Run your first simulation programmatically:

```python
from haso_sim import simulate

results = simulate(
    map_path="notebooks/data/office_building_simple.yaml",
    config_path="notebooks/data/config_baseline.yaml",
    tmax=600,
    seed=42
)

print(f"Simulation completed in {results['world'].time:.1f} seconds")
print(f"All rooms cleared: {results['all_cleared']}")
```

Visualize results using the summary dashboard:

```python
from haso_sim import create_summary_dashboard

create_summary_dashboard(results['world'])
```

## Key Concepts

### Building Maps

Buildings are defined as weighted graphs encoded in YAML format. Nodes represent rooms, corridors, and exits, while edges capture doors, hallways, and traversal costs. Each node can include properties such as hazards, evacuees, and priority levels.

Example node definition:

```yaml
- id: 1
  type: "SMALL_ROOM"
  x: 10
  y: 5
  name: "Office 101"
  room_priority: 3
  hazard: "SMOKE"
  hazard_severity: 0.5
```

The room_priority field ranges from 1 (highest) to 5 (lowest), while hazard_severity ranges from 0.0 to 1.0.

### Responder Roles

The simulator supports four cooperative responder roles. Scouts serve as fast explorers that tag evacuees and signal priorities. Securers handle hazard mitigation, assist evacuees, and clear rooms. Checkpointers guard strategic locations and provide assistance. Evacuators perform final verification and double-check clearance status.

### Configuration

Agent configurations specify starting positions, roles, sweep modes, and personal priorities:

```yaml
agents:
  - id: 0
    role: "SCOUT"
    node: 0
    sweep_mode: "right"
    personal_priority: 4

weights:
  room_priority: 1.5
  distance: 0.5
  hazard_penalty: 2.0
```

Sweep modes include "right", "left", and "corridor" strategies. Weight parameters control decision-making priorities, with higher values emphasizing critical rooms, distance penalties, or hazard avoidance.

## Common Tasks

### Load and Validate a Map

Use load_map to parse YAML building definitions and validate_map to check for connectivity issues:

```python
from haso_sim import load_map, validate_map, create_building_summary

G = load_map("notebooks/data/office_building_simple.yaml")

issues = validate_map(G)
if issues:
    for issue in issues:
        print(issue)

summary = create_building_summary(G)
print(f"Nodes: {summary['total_nodes']}")
print(f"Evacuees: {summary['total_evacuees']}")
```

### Visualize Building Layout

Generate static layout visualizations using plot_building_layout:

```python
from haso_sim import plot_building_layout
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 10))
plot_building_layout(G, ax=ax, show_labels=True)
plt.show()
```

### Run Simulation and Generate Report

Execute simulations and extract summary reports:

```python
from haso_sim import simulate, generate_summary_report

results = simulate(
    map_path="notebooks/data/office_building_simple.yaml",
    config_path="notebooks/data/config_baseline.yaml",
    tmax=600
)

report = generate_summary_report(results['world'])
print(report)
```

### Analyze Agent Performance

Inspect individual agent metrics including rooms cleared and distance traveled:

```python
from haso_sim import analyze_agent_performance

world = results['world']
for agent in world.agents:
    perf = analyze_agent_performance(agent)
    print(f"Agent {perf['agent_id']} ({perf['role']}):")
    print(f"  Rooms cleared: {perf['rooms_cleared']}")
    print(f"  Distance traveled: {perf['distance_traveled']:.1f}m")
```

### Compare Different Strategies

Run comparative studies across sweep strategies:

```python
strategies = ['right', 'left', 'corridor']
results_dict = {}

for strategy in strategies:
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

for strategy, metrics in results_dict.items():
    print(f"{strategy}: {metrics['time']:.1f}s, {metrics['cleared']} rooms")
```

## Example Buildings

The simulator includes several pre-configured building layouts. The office_building_simple.yaml file provides an entry-level scenario with 11 nodes, 7 rooms, 7 evacuees, and a single hazard area, ideal for quick testing and validation.

The hospital_wing.yaml file presents a more complex scenario with 15 nodes, 9 patient rooms, 13 evacuees (many requiring assistance), and a hazard area, suitable for stress-testing evacuation protocols.

## Interactive Notebook

The notebooks/simulation_demo.ipynb notebook provides a step-by-step walkthrough covering map inspection, fog-of-war visualization, sweep comparisons, and automated report generation. Launch Jupyter Lab and open the notebook:

```bash
jupyter lab
```

The notebook demonstrates building visualization, simulation execution, results analysis, and comparative studies using the create_summary_dashboard, generate_summary_report, and analyze_agent_performance functions.

## Customization

### Create Your Own Building

To define a custom building, copy an example YAML file and modify nodes and edges. Add evacuees and hazards as needed, then validate using validate_map before running simulations. Ensure all rooms connect to at least one exit to prevent simulation stalls.

### Modify Agent Behavior

Edit haso_sim/policies.py to customize decision-making logic, priority calculations, movement strategies, and task allocation. The policy system integrates with the discrete-event scheduler to coordinate multi-agent operations.

### Adjust Simulation Parameters

Configure simulation duration, recording intervals, and random seeds in the configuration YAML:

```yaml
simulation:
  tmax: 900
  record_interval: 3.0
  seed: 42
```

The tmax parameter sets maximum simulation time in seconds, while record_interval controls how frequently state snapshots are captured for visualization and analysis.

## Troubleshooting

Import errors typically indicate that the repository root is not on PYTHONPATH. Run scripts from the project root directory or add the root to your Python path:

```python
import sys
sys.path.append('..')
```

Missing dependency errors can be resolved by upgrading packages:

```bash
pip install --upgrade -r requirements.txt
```

For headless visualization sessions, configure matplotlib to use a non-interactive backend:

```python
import matplotlib
matplotlib.use('Agg')
```

## API Quick Reference

### Main Functions

```python
from haso_sim import load_map, build_world, simulate

G = load_map(path)
world = build_world(map_path, config_path, seed=42)
results = simulate(map_path, config_path, tmax=600, seed=42, animate=False)
```

### Visualization Functions

```python
from haso_sim import (
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
from haso_sim import (
    analyze_simulation_results,
    analyze_agent_performance,
    generate_summary_report,
    validate_map,
    create_building_summary
)
```

## Next Steps

After completing the quickstart, explore different building layouts, experiment with agent configurations, compare sweep strategies, analyze optimization opportunities, and extend the simulator with custom features. Refer to the main README.md for comprehensive documentation, review notebooks/simulation_demo.ipynb for examples, examine YAML files in notebooks/data/ for configuration patterns, and consult inline documentation in source files for implementation details.
