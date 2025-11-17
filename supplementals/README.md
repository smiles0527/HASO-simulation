# Supplemental Figures Generator

This directory contains scripts to generate all supplemental figures for the HiMCM documentation.

## Usage

Generate all figures:

```bash
python generate_all_figures.py
```

This will create a `figures/` subdirectory with all generated PNG files.

## Individual Figure Generation

You can also import and use individual functions from `generate_figures.py`:

```python
from generate_figures import generate_spatial_embedding
from pathlib import Path

generate_spatial_embedding(
    Path("notebooks/data/office_building_simple.yaml"),
    Path("figures/01_spatial_embedding.png")
)
```

## Figure List

1. **01_spatial_embedding.png** - Spatial embedding of nodes with geometric distances
2. **02_weighted_costs.png** - Visualization of weighted traversal costs
3. **03_hazard_routing.png** - Agent routing over hazard-modified weighted graph
4. **04_hazard_diffusion.png** - Hazard intensity diffusion over time (t0, t1, t2)
5. **05_securer_mitigation.png** - Securer mitigation reducing hazard intensity
6. **06_evacuee_escort.png** - Evacuee escort sequence from room to exit
7. **07_hazard_interaction.png** - Hazard-routing-evacuee interaction diagram
8. **08_haso_zones.png** - HASO-generated zone partitions
9. **09_basic_scenario.png** - Basic 6-room scenario (COMAP Figure 1 style)
10. **10_school_wing.png** - Additional layout: School wing
11. **11_lab_block.png** - Additional layout: Lab block with hazards
12. **12_sweep_paths.png** - Sweep path/room-checking sequence
13. **13_clearance_times.png** - Clearance time plots (C(t), hazard, congestion)
14. **14_agent_trajectories.png** - Agent trajectory overlays (color-coded by role)
15. **15_hazard_heatmap.png** - Heatmap of hazard for chosen scenario
16. **16_stress_test.png** - Stress test visual (agent count vs congestion)
17. **17_state_machine.png** - State machine diagram for agent states
18. **18_flowchart.png** - Flowchart of discrete-event simulation engine
19. **19_task_allocation.png** - Task allocation diagram with reallocation

## Requirements

- Python 3.11+
- matplotlib
- numpy
- networkx (optional, for some advanced features)
- haso_sim package (from parent directory)

## Notes

Some figures require running simulations, which may take a few seconds. The scripts use the example building maps from `notebooks/data/` by default.

