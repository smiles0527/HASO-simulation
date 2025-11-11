"""
HASO Live Evacuation Simulation Dashboard

Professional real-time visualization system for the HASO (Hierarchical Adaptive Search
and Optimization) evacuation framework. Provides comprehensive analytics, interactive
controls, and export capabilities for research and operational planning.

Features:
    - Real-time agent tracking with smooth interpolation
    - Multi-panel analytics dashboard
    - Flow dynamics visualization using electrical circuit modeling
    - Interactive speed controls (0.25x to 5x playback)
    - Hazard and clearance state visualization
    - Zone assignment monitoring
    - Video export for presentations
    - Frame capture for documentation

Usage:
    Interactive Mode:
        python run_live_simulation.py
    
    Video Export:
        python run_live_simulation.py --save-video --video-path output.mp4
    
    Custom Configuration:
        python run_live_simulation.py --map data/hospital.yaml --duration 600 --fps 15

Controls:
    [SPACE]     Pause/Resume simulation
    [R]         Reset to beginning
    [S]         Save current frame as PNG
    [1-5]       Set playback speed (0.25x, 0.5x, 1x, 2x, 5x)
    [H]         Toggle heat map overlay
    [F]         Toggle flow direction arrows
    [ESC]       Exit simulation

Requirements:
    - Python 3.7+
    - matplotlib, numpy, networkx, pyyaml
    - ffmpeg (for video export)
"""

import sys
import argparse
import os
from pathlib import Path
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Banner
print("\n" + "=" * 80)
print("    HASO EVACUATION SIMULATION - Live Dashboard")
print("    Real-Time Visualization & Analytics System")
print("=" * 80 + "\n")

# Load simulation framework
try:
    print("Loading simulation framework...")
    from notebooks import build_world
    from notebooks.src.animate_live import create_live_visualization
    print("[OK] Framework loaded successfully\n")
except ImportError as e:
    print(f"[ERROR] Error loading framework: {e}")
    print("\nPlease ensure all dependencies are installed:")
    print("  pip install -r requirements.txt\n")
    sys.exit(1)


def validate_file_path(path: str, file_type: str) -> bool:
    """Validate that a file exists and is readable."""
    if not os.path.exists(path):
        print(f"[ERROR] {file_type} file not found: {path}")
        return False
    if not os.path.isfile(path):
        print(f"[ERROR] {path} is not a file")
        return False
    return True


def parse_arguments():
    """Parse command-line arguments with validation."""
    parser = argparse.ArgumentParser(
        description='HASO Live Evacuation Simulation Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python run_live_simulation.py
  
  Export video:
    python run_live_simulation.py --save-video --video-path my_simulation.mp4
  
  Custom building and high FPS:
    python run_live_simulation.py --map data/hospital.yaml --fps 20 --duration 600
  
  Fast preview at 5x speed:
    python run_live_simulation.py --duration 180 --fps 15

For more information, see docs/LIVE_ANIMATION.md
        """
    )
    
    # Input files
    parser.add_argument('--map', 
                       default='notebooks/data/office_building_simple.yaml',
                       help='Building map YAML file (default: office_building_simple.yaml)')
    parser.add_argument('--config', 
                       default='notebooks/data/config_baseline.yaml',
                       help='Simulation configuration YAML (default: config_baseline.yaml)')
    
    # Simulation parameters
    parser.add_argument('--duration', type=float, default=300.0,
                       help='Simulation duration in seconds (default: 300)')
    parser.add_argument('--fps', type=int, default=20,
                       help='Frames per second (default: 20 for smooth playback)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Output options
    parser.add_argument('--save-video', action='store_true',
                       help='Render and save as video file instead of interactive display')
    parser.add_argument('--video-path', default='evacuation_simulation.mp4',
                       help='Output video filename (default: evacuation_simulation.mp4)')
    
    # Advanced options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed progress messages')
    
    return parser.parse_args()


def print_simulation_info(world, args):
    """Display comprehensive simulation information."""
    print("+" + "=" * 78 + "+")
    print("|  SIMULATION CONFIGURATION" + " " * 51 + "|")
    print("+" + "=" * 78 + "+")
    print()
    
    print(f"  Building Map:        {args.map}")
    print(f"  Configuration:       {args.config}")
    print(f"  Random Seed:         {args.seed}")
    print()
    
    cleared, total = world.G.get_cleared_count()
    exits = [n for n in world.G.nodes.values() if n.node_type.name == 'EXIT']
    hazards = [n for n in world.G.nodes.values() if hasattr(n, 'hazard') and n.hazard.name != 'NONE']
    
    print(f"  Total Nodes:         {len(world.G.nodes)}")
    print(f"  Total Edges:         {len(world.G.edges) // 2}")  # Bidirectional
    print(f"  Rooms to Clear:      {total}")
    print(f"  Exit Points:         {len(exits)}")
    print(f"  Active Hazards:      {len(hazards)}")
    print()
    
    print(f"  Responder Agents:    {len(world.agents)}")
    for role_name in ['SCOUT', 'SECURER', 'CHECKPOINTER', 'EVACUATOR']:
        count = sum(1 for a in world.agents if a.role.name == role_name)
        if count > 0:
            print(f"    - {role_name:12}   {count}")
    print()
    
    print(f"  Simulation Duration: {args.duration:.0f} seconds")
    print(f"  Frame Rate:          {args.fps} FPS")
    print(f"  Total Frames:        {int(args.duration * args.fps)}")
    print()
    
    if args.save_video:
        file_size_est = (args.duration * args.fps * 0.05)  # Rough estimate in MB
        print(f"  Output Mode:         VIDEO RENDERING")
        print(f"  Video File:          {args.video_path}")
        print(f"  Estimated Size:      ~{file_size_est:.1f} MB")
        print(f"  Estimated Time:      ~{args.duration / 10:.1f} minutes")
    else:
        print(f"  Output Mode:         INTERACTIVE DISPLAY")
    
    print()


def print_controls():
    """Display interactive control instructions."""
    print("+" + "=" * 78 + "+")
    print("|  INTERACTIVE CONTROLS" + " " * 55 + "|")
    print("+" + "=" * 78 + "+")
    print()
    print("  [SPACE BAR]    Pause / Resume simulation")
    print("  [R]            Reset to beginning")
    print("  [S]            Save current frame as PNG")
    print("  [1]            Set speed to 0.25x (slow motion)")
    print("  [2]            Set speed to 0.5x")
    print("  [3]            Set speed to 1.0x (normal)")
    print("  [4]            Set speed to 2.0x (fast)")
    print("  [5]            Set speed to 5.0x (very fast)")
    print("  [H]            Toggle heat map overlay")
    print("  [F]            Toggle flow direction arrows")
    print("  [ESC]          Exit simulation")
    print()
    print("  Time Slider:   Click and drag to jump to specific time")
    print()


def print_results(world, start_time, end_time):
    """Display final simulation results and statistics."""
    print()
    print("+" + "=" * 78 + "+")
    print("|  SIMULATION RESULTS" + " " * 59 + "|")
    print("+" + "=" * 78 + "+")
    print()
    
    final_cleared, final_total = world.G.get_cleared_count()
    pct_cleared = (final_cleared / final_total * 100) if final_total > 0 else 0
    
    print(f"  Simulation Time:     {world.time:.1f} seconds")
    print(f"  Rooms Cleared:       {final_cleared} / {final_total} ({pct_cleared:.1f}%)")
    print()
    
    # Agent statistics
    print("  Agent Performance:")
    for agent in world.agents:
        print(f"    Agent {agent.id} ({agent.role.name[:4]}):  "
              f"{agent.rooms_cleared} rooms cleared, "
              f"{len(agent.visited_nodes)} nodes visited, "
              f"Status: {agent.status.name}")
    print()
    
    # Evacuee statistics
    total_evacuees = sum(len(n.evacuees) for n in world.G.nodes.values())
    evacuees_at_exits = sum(len(n.evacuees) for n in world.G.nodes.values() 
                           if n.node_type.name == 'EXIT')
    
    if total_evacuees > 0:
        pct_safe = (evacuees_at_exits / total_evacuees * 100)
        print(f"  Evacuees:            {evacuees_at_exits} / {total_evacuees} reached exits ({pct_safe:.1f}%)")
        print()
    
    # Performance
    elapsed = end_time - start_time
    print(f"  Rendering Time:      {elapsed:.1f} seconds")
    print(f"  Average FPS:         {len(world.history) / elapsed:.1f}")
    print()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate input files
    if not validate_file_path(args.map, "Map"):
        sys.exit(1)
    if not validate_file_path(args.config, "Config"):
        sys.exit(1)
    
    # Validate video path directory exists
    if args.save_video:
        video_dir = os.path.dirname(args.video_path)
        if video_dir and not os.path.exists(video_dir):
            print(f"[ERROR] Output directory does not exist: {video_dir}")
            sys.exit(1)
    
    # Build simulation world
    if not args.quiet:
        print("Building simulation world...")
    
    try:
        world = build_world(args.map, config_path=args.config, seed=args.seed)
    except Exception as e:
        print(f"[ERROR] Error building world: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if not args.quiet:
        print("[OK] World built successfully\n")
        print_simulation_info(world, args)
    
    # No interactive controls output by default for cleaner UX
    
    # Launch visualization
    if args.save_video:
        print("Starting video rendering...")
        print("Progress will be shown in the console...")
    else:
        print("Launching interactive dashboard...")
    
    print()
    print("─" * 80)
    print()
    
    import time
    start_time = time.time()
    
    try:
        advanced_flag = getattr(args, 'simple', None)
        use_advanced = not advanced_flag if isinstance(advanced_flag, bool) else True

        dashboard = create_live_visualization(
            world,
            fps=args.fps,
            duration=args.duration,
            save_video=args.save_video,
            video_path=args.video_path,
            advanced=use_advanced,
            quiet=args.quiet
        )
        
        end_time = time.time()
        
        if not args.quiet:
            print()
            print("─" * 80)
            print()
            
            if args.save_video:
                print(f"[OK] Video saved successfully: {args.video_path}")
                file_size = os.path.getsize(args.video_path) / (1024 * 1024)
                print(f"  File size: {file_size:.2f} MB")
            
            print_results(world, start_time, end_time)
        
        print("+" + "=" * 78 + "+")
        print("|  SIMULATION COMPLETE" + " " * 57 + "|")
        print("+" + "=" * 78 + "+")
        print()
        
    except KeyboardInterrupt:
        print("\n\n[WARNING] Simulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
