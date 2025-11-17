#!/usr/bin/env python3
"""
Generate all supplemental figures for HiMCM documentation.

Run: python generate_all_figures.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_figures import (
    generate_spatial_embedding,
    generate_weighted_traversal_costs,
    generate_hazard_routing,
    generate_hazard_diffusion,
    generate_securer_mitigation,
    generate_evacuee_escort,
    generate_hazard_routing_interaction,
    generate_haso_zones,
    generate_basic_scenario,
    generate_additional_layouts,
    generate_sweep_paths,
    generate_clearance_time_plots,
    generate_agent_trajectories,
    generate_hazard_heatmap,
    generate_stress_test,
    generate_state_machine,
    generate_flowchart,
    generate_task_allocation,
)

def main():
    """Generate all figures."""
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    map_dir = Path(__file__).parent.parent / "notebooks" / "data"
    
    print("Generating supplemental figures...")
    
    # Core model diagrams
    print("1. Spatial embedding...")
    generate_spatial_embedding(map_dir / "office_building_simple.yaml", 
                              output_dir / "01_spatial_embedding.png")
    
    print("2. Weighted traversal costs...")
    generate_weighted_traversal_costs(map_dir / "office_building_simple.yaml",
                                     output_dir / "02_weighted_costs.png")
    
    print("3. Hazard-modified routing...")
    generate_hazard_routing(map_dir / "office_building_simple.yaml",
                           output_dir / "03_hazard_routing.png")
    
    print("4. Hazard diffusion...")
    generate_hazard_diffusion(map_dir / "office_building_simple.yaml",
                            output_dir / "04_hazard_diffusion.png")
    
    print("5. Securer mitigation...")
    generate_securer_mitigation(map_dir / "office_building_simple.yaml",
                               output_dir / "05_securer_mitigation.png")
    
    print("6. Evacuee escort...")
    generate_evacuee_escort(map_dir / "office_building_simple.yaml",
                           output_dir / "06_evacuee_escort.png")
    
    print("7. Hazard-routing interaction...")
    generate_hazard_routing_interaction(map_dir / "office_building_simple.yaml",
                                       output_dir / "07_hazard_interaction.png")
    
    print("8. HASO zones...")
    generate_haso_zones(map_dir / "office_building_simple.yaml",
                       map_dir / "config_baseline.yaml",
                       output_dir / "08_haso_zones.png")
    
    # HiMCM requirement figures
    print("9. Basic scenario...")
    generate_basic_scenario(output_dir / "09_basic_scenario.png")
    
    print("10-11. Additional layouts...")
    generate_additional_layouts(output_dir)
    
    print("12. Sweep paths...")
    generate_sweep_paths(map_dir, output_dir)
    
    print("13. Clearance time plots...")
    generate_clearance_time_plots(map_dir / "office_building_simple.yaml",
                                  map_dir / "config_baseline.yaml",
                                  output_dir / "13_clearance_times.png")
    
    print("14. Agent trajectories...")
    generate_agent_trajectories(map_dir / "office_building_simple.yaml",
                               map_dir / "config_baseline.yaml",
                               output_dir / "14_agent_trajectories.png")
    
    print("15. Hazard heatmap...")
    generate_hazard_heatmap(map_dir / "office_building_simple.yaml",
                           output_dir / "15_hazard_heatmap.png")
    
    print("16. Stress test...")
    generate_stress_test(map_dir / "office_building_simple.yaml",
                        output_dir / "16_stress_test.png")
    
    # Optional diagrams
    print("17. State machine...")
    generate_state_machine(output_dir / "17_state_machine.png")
    
    print("18. Flowchart...")
    generate_flowchart(output_dir / "18_flowchart.png")
    
    print("19. Task allocation...")
    generate_task_allocation(map_dir / "office_building_simple.yaml",
                            map_dir / "config_baseline.yaml",
                            output_dir / "19_task_allocation.png")
    
    print(f"\nAll figures generated in {output_dir}")

if __name__ == "__main__":
    main()

