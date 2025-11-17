#!/usr/bin/env python3
"""
Quick test to verify figure generation works.
Generates a subset of figures to test the system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_figures import (
    generate_spatial_embedding,
    generate_state_machine,
    generate_flowchart,
    generate_stress_test,
)

def main():
    """Test a few simple figures."""
    output_dir = Path(__file__).parent / "test_figures"
    output_dir.mkdir(exist_ok=True)
    
    map_dir = Path(__file__).parent.parent / "notebooks" / "data"
    map_path = map_dir / "office_building_simple.yaml"
    
    if not map_path.exists():
        print(f"Warning: {map_path} not found. Some tests will be skipped.")
        map_path = None
    
    print("Testing figure generation...")
    
    # Test 1: State machine (no dependencies)
    print("1. State machine...")
    try:
        generate_state_machine(output_dir / "test_state_machine.png")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 2: Flowchart (no dependencies)
    print("2. Flowchart...")
    try:
        generate_flowchart(output_dir / "test_flowchart.png")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 3: Stress test (no dependencies)
    print("3. Stress test...")
    try:
        generate_stress_test(map_path if map_path else Path("dummy.yaml"),
                            output_dir / "test_stress.png")
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 4: Spatial embedding (requires map)
    if map_path and map_path.exists():
        print("4. Spatial embedding...")
        try:
            generate_spatial_embedding(map_path, output_dir / "test_spatial.png")
            print("   ✓ Success")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
    
    print(f"\nTest figures saved to {output_dir}")

if __name__ == "__main__":
    main()

