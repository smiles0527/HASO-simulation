# How to Run the Simulation

This document provides step-by-step instructions for executing simulations using various methods, from quick tests to interactive dashboards and custom scripts.

## Scenario Launcher

The interactive scenario launcher provides the fastest way to explore curated scenarios:

```bash
python scripts/launch_simulation.py
```

This opens a start-menu interface where you can select scenarios, adjust responder counts, scale hazards or evacuee density, and choose between a live dashboard or headless run with automatic reporting. The launcher supports the Innovation Hub, Medical Pavilion, and Transit Atrium layouts.

## Quick Test

For a rapid validation of the installation, run the test script:

```bash
python test_simulation.py
```

This script tests all imports, loads the office building map, runs a 30-second simulation, and displays results. Expected output includes agent movement logs and simulation statistics.

## Full Simulation

Execute a complete evacuation simulation:

```bash
python test_full_simulation.py
```

This runs a complete 5-minute evacuation, shows detailed agent performance metrics, and displays which rooms were cleared. Use this method to validate end-to-end functionality before running longer experiments.

## Visual Demo

Generate static visualization outputs:

```bash
python scripts/demo_visual.py
```

This script runs the simulation and creates four visualization files in the demo_results directory: 1_building_layout.png (building map), 2_clearance_progress.png (progress over time), 3_agent_paths.png (agent movements), and 4_complete_dashboard.png (full summary). The script also displays an interactive dashboard and progress graphs.

## Interactive Jupyter Notebook

The recommended method for exploration and experimentation is the Jupyter notebook interface. Install Jupyter Lab if needed:

```bash
pip install jupyterlab
```

Launch Jupyter Lab:

```bash
jupyter lab
```

Navigate to notebooks/simulation_demo.ipynb and run cells using Cell → Run All or Shift+Enter. The notebook provides building visualizations, agent movements, performance graphs, and statistical analysis in an interactive environment.

## Custom Python Script

Create custom simulation scripts for programmatic control:

```python
from haso_sim import simulate
from haso_sim import generate_summary_report

results = simulate(
    map_path="notebooks/data/office_building_simple.yaml",
    config_path="notebooks/data/config_baseline.yaml",
    tmax=600,
    seed=42
)

world = results['world']
print(f"Simulation time: {world.time:.1f}s")
print(f"All rooms cleared: {results['all_cleared']}")

report = generate_summary_report(world)
print(report)
```

Save this as my_simulation.py and execute:

```bash
python my_simulation.py
```

## Python Interactive Shell

For quick experimentation, use the Python interactive shell:

```bash
python
```

Then execute:

```python
>>> from haso_sim import simulate
>>> results = simulate("notebooks/data/office_building_simple.yaml", tmax=300)
>>> world = results['world']
>>> print(f"Time: {world.time:.1f}s")
>>> cleared, total = world.G.get_cleared_count()
>>> print(f"Cleared {cleared}/{total} rooms")
```

## Recommended Path for Beginners

First-time users should start with the quick test script to validate the installation. For visual exploration, run the demo visual script. For deeper investigation, launch Jupyter Lab and open the simulation demo notebook. Once comfortable with the basics, experiment with custom maps and configurations.

## Files You Can Modify

Building maps are defined in YAML files. The office_building_simple.yaml file provides a small office layout with 11 nodes and 7 rooms, while hospital_wing.yaml presents a hospital scenario with 15 nodes and 9 rooms. Configuration files such as config_baseline.yaml control agent setup and parameters.

You can experiment by editing the number of agents, agent roles (Scout, Securer, Checkpointer, Evacuator), sweep strategies (right, left, corridor), room priorities, and hazard locations.

## Troubleshooting

Module not found errors typically indicate missing dependencies. Install or upgrade packages:

```bash
pip install -r requirements.txt
```

For visualization-specific issues, ensure matplotlib is installed:

```bash
pip install matplotlib
```

Import errors often occur when scripts are not run from the project root directory. Change to the project root before executing scripts. If simulations run but no rooms are cleared, this may be normal for very short simulations under 60 seconds. Increase tmax to 300 or 600 seconds and check agent logs to understand behavior.

## Expected Output

When running a simulation, you'll see output similar to:

```
[World] Simulation ended at t=300.00s
[World] Cleared 3/7 rooms

AGENT SUMMARY:
Agent 0 - SCOUT     | Cleared:  0 | Distance: 190.0m
Agent 1 - SECURER   | Cleared:  3 | Distance: 560.0m
Agent 2 - CHECKPOINTER | Cleared:  0 | Distance: 80.0m
Agent 3 - EVACUATOR | Cleared:  0 | Distance: 136.0m

Efficiency: 0.60 rooms/minute
```

## Next Steps

After running your first simulation, analyze results by examining agent logs, clearance rates, and distances traveled. Try different scenarios including hospital and office layouts. Experiment with sweep strategies and team compositions. Optimize configurations to find the fastest way to clear all rooms. Create charts and animations for visualization. Extend the simulator with new features, hazard models, and evacuee behaviors.

## Quick Tips

Start with small scenarios using office_building_simple.yaml. Use Jupyter notebooks for the best visualization and experimentation experience. Check agent logs to understand decision-making. Adjust simulation duration using tmax=600 for complete clearance. Visualizations save automatically to the demo_results folder.

For the fastest start, run the test simulation script. For the best experience, launch Jupyter Lab and open the simulation demo notebook. For additional help, refer to README.md or GETTING_STARTED.md.
