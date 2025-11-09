# Live Evacuation Simulation Dashboard

## Overview

The HASO Live Simulation Dashboard provides a comprehensive real-time visualization environment for evacuation operations. This professional-grade tool enables researchers, emergency planners, and operational teams to observe, analyze, and optimize evacuation strategies through an interactive multi-panel interface.

## Key Capabilities

### Real-Time Visualization
- **Smooth Agent Tracking**: Continuous position updates with interpolated movement between nodes
- **Dynamic State Representation**: Real-time color-coding of room clearance, hazard levels, and agent status
- **Flow Dynamics**: Electrical circuit-based flow visualization showing evacuation pressure and bottlenecks
- **Historical Trails**: Visual path tracking showing agent movement patterns over time

### Analytics Dashboard

The dashboard consists of eight integrated panels providing comprehensive situational awareness:

1. **Building Layout** (Primary Display)
   - Live agent positions with role-based markers
   - Node state visualization (cleared, in-progress, hazardous, unknown)
   - Edge flow rates with color-coded intensity
   - Interactive legend and scale indicators

2. **Clearance Progress Graph**
   - Time-series tracking of cleared vs. discovered rooms
   - Progress rate indicators
   - Completion percentage metrics

3. **Flow Dynamics Analysis**
   - Total flow rate (current in amperes)
   - Average pressure levels (voltage)
   - Number of active evacuation paths
   - Real-time electrical circuit model metrics

4. **Agent Status & Performance**
   - Individual agent status (active, slowed, progressing, immobilized)
   - Rooms cleared per agent
   - Current location and activity
   - Role assignments and effectiveness

5. **Zone Coverage Monitor**
   - Zone assignment visualization
   - Per-zone clearance progress
   - Coverage percentage with visual indicators

6. **Active Hazards Display**
   - Hazard type distribution (fire, smoke, heat, chemical, etc.)
   - Severity levels and locations
   - Hazard propagation tracking

7. **Summary Statistics**
   - Simulation time and progress
   - Overall clearance percentage
   - Evacuee safety status
   - Flow metrics and system performance
   - Actual rendering FPS

8. **Interactive Controls**
   - Playback speed adjustment
   - Time slider for navigation
   - Keyboard shortcuts reference

## Usage

### Quick Start - Interactive Mode

Launch the dashboard with default settings:

```bash
python run_live_simulation.py
```

This opens an interactive window with all controls enabled and real-time visualization at 10 FPS.

### Video Export Mode

Render the entire simulation to a video file:

```bash
python run_live_simulation.py --save-video --video-path my_simulation.mp4
```

Video specifications:
- Format: MP4 (H.264 codec)
- Resolution: 2400×1400 pixels
- Bitrate: 2000 kbps
- Frame rate: Configurable (default 10 FPS)

### Advanced Configuration

Customize simulation parameters:

```bash
python run_live_simulation.py \
    --map notebooks/data/hospital_wing.yaml \
    --config notebooks/data/config_baseline.yaml \
    --duration 600 \
    --fps 15 \
    --seed 123
```

**Command-Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--map` | string | office_building_simple.yaml | Building map YAML file path |
| `--config` | string | config_baseline.yaml | Simulation configuration file |
| `--duration` | float | 300.0 | Total simulation time (seconds) |
| `--fps` | integer | 10 | Target frames per second |
| `--seed` | integer | 42 | Random seed for reproducibility |
| `--save-video` | flag | False | Enable video rendering mode |
| `--video-path` | string | evacuation_simulation.mp4 | Output video filename |
| `--no-advanced` | flag | False | Disable advanced features for performance |
| `--quiet` | flag | False | Suppress detailed console output |

### Python API

Integrate the dashboard into custom applications:

```python
from notebooks import build_world
from notebooks.src.animate_live import create_live_visualization

# Initialize simulation world
world = build_world(
    "notebooks/data/hospital_wing.yaml",
    config_path="notebooks/data/config_baseline.yaml",
    seed=42
)

# Create and launch dashboard
dashboard = create_live_visualization(
    world,
    fps=15,
    duration=300.0,
    save_video=False,
    advanced=True
)

# Access recorded simulation data
print(f"Simulation completed at t={dashboard.times[-1]:.1f}s")
print(f"Final clearance: {dashboard.cleared_counts[-1]} rooms")
print(f"Average FPS: {dashboard.actual_fps:.1f}")

# Extract agent trajectories
for agent_id, positions in dashboard.agent_positions_history.items():
    print(f"Agent {agent_id}: {len(positions)} recorded positions")
```

## Interactive Controls

### Keyboard Shortcuts

| Key | Function |
|-----|----------|
| **SPACE** | Pause/Resume simulation |
| **R** | Reset to beginning |
| **S** | Save current frame as high-resolution PNG |
| **1** | Set playback speed to 0.25× (slow motion) |
| **2** | Set playback speed to 0.5× |
| **3** | Set playback speed to 1.0× (normal) |
| **4** | Set playback speed to 2.0× (fast forward) |
| **5** | Set playback speed to 5.0× (very fast) |
| **H** | Toggle hazard heat map overlay |
| **F** | Toggle flow direction arrows |
| **ESC** | Exit simulation |

### Mouse Controls

- **Time Slider**: Click and drag to jump to specific simulation time
- **Panel Zoom**: Standard matplotlib zoom and pan (toolbar buttons)

## Visual Elements

### Node Representation

**Color Coding:**
- **Blue** (#0D6EFD): Exit points
- **Green** (#28A745): Cleared rooms
- **Yellow** (#FFC107): Rooms in progress (agent present)
- **Red** (#DC3545): Uncleared rooms
- **Gray** (#6C757D): Unknown areas (fog of war)
- **Dark Red** (#8B0000): High-hazard zones

**Size Scaling:**
- Node size proportional to room area
- Exit nodes displayed as squares
- Standard rooms displayed as circles

**Hazard Indicators:**
- Warning triangles for hazardous nodes
- Size indicates severity level
- Color gradient from orange (moderate) to dark red (critical)

### Agent Visualization

**Markers:**
- Star symbols for easy identification
- Color-coded by role:
  - Green: Scout
  - Blue: Securer
  - Yellow: Checkpointer
  - Purple: Evacuator

**Labels:**
- Agent ID and role abbreviation
- Position updates with smooth interpolation
- Semi-transparent background for readability

**Trails:**
- Colored path history (last 50 positions)
- Fading alpha for temporal depth
- Line thickness indicates movement speed

**Status Indicators:**
- Small circle below agent marker
- Color represents current status:
  - Green: Normal operation
  - Yellow: Slowed
  - Blue: Performing task
  - Red: Immobilized
  - Gray: Incapacitated

### Edge/Corridor Visualization

**Flow-Based Coloring:**
- **Blue**: High flow (>0.5 A) - primary evacuation routes
- **Green**: Moderate flow (0.1-0.5 A) - active paths
- **Gray**: Low flow (<0.1 A) - minimal usage
- **Dashed**: Non-traversable or blocked

**Line Properties:**
- Thickness proportional to conductance (1/resistance)
- Alpha (transparency) indicates flow activity
- Labels show physical distance in meters

## Technical Architecture

### Rendering System

**Frame Update Pipeline:**
```
1. Advance simulation time by (frame_time × speed_multiplier)
2. Update flow dynamics model
3. Calculate agent interpolated positions
4. Update node colors based on clearance state
5. Update edge visualization based on flow rates
6. Refresh all analytics graphs
7. Update information panels
8. Render frame to display or video buffer
```

**Performance Characteristics:**
- Target: 10-30 FPS (configurable)
- Actual FPS tracked and displayed in real-time
- Automatic frame timing adjustment
- Memory usage: ~2 MB per minute of simulation
- Scales efficiently to buildings with 100+ nodes

### Data Recording

The dashboard maintains comprehensive historical records:

**Time Series Data:**
- Frame timestamps
- Clearance counts (cleared and discovered)
- Flow metrics (total flow, average pressure, active paths)
- Agent positions with interpolation

**Storage Format:**
- In-memory deques with configurable size limits
- Efficient circular buffers for trail visualization
- Full history retained for time-slider functionality

**Export Capabilities:**
- Individual frame export (PNG, 200 DPI)
- Complete video export (MP4, H.264)
- Data access via Python API for post-processing

### Electrical Flow Model Integration

The dashboard integrates an advanced electrical circuit analogy for flow dynamics:

**Circuit Analogy Mapping:**
- Current (I) → Evacuee flow rate (people/second)
- Voltage (V) → Evacuation pressure (hazard severity + occupancy)
- Resistance (R) → Path impedance (width, hazards, visibility)
- Conductance (G) → Evacuation capacity (1/R)

**Flow Calculations:**
- Ohm's Law: I = V / R
- Kirchhoff's Current Law for node conservation
- Series resistance for sequential bottlenecks
- Parallel resistance for multiple exit paths

**Visualization:**
- Edge color intensity represents current flow
- Edge thickness represents conductance
- Real-time graphs show system-wide metrics
- Flow direction arrows (advanced mode)

## Application Scenarios

### 1. Research and Algorithm Development

**Use Case**: Testing and validating evacuation algorithms

**Benefits:**
- Visual debugging of agent behavior
- Real-time verification of pathfinding logic
- Zone assignment optimization
- Hazard response evaluation

**Example Workflow:**
1. Implement algorithm modification
2. Run simulation with visualization
3. Observe agent decision-making patterns
4. Identify and resolve inefficiencies
5. Export results for documentation

### 2. Emergency Response Training

**Use Case**: Training emergency responders on evacuation protocols

**Benefits:**
- Realistic scenario simulation
- Understanding of HASO methodology
- Coordination visualization
- Decision-making practice

**Training Features:**
- Pause capability for instruction
- Speed control for detailed analysis
- Multiple scenario configurations
- Performance metrics tracking

### 3. Building Safety Analysis

**Use Case**: Evaluating building evacuation capabilities

**Benefits:**
- Bottleneck identification
- Exit capacity assessment
- Hazard impact modeling
- Optimal responder placement

**Analysis Tools:**
- Flow dynamics visualization
- Clearance efficiency metrics
- Zone coverage analysis
- Comparative scenario testing

### 4. Academic Presentations and Publications

**Use Case**: Demonstrating research results

**Benefits:**
- High-quality video output
- Professional visualization aesthetics
- Comprehensive data display
- Customizable views

**Presentation Features:**
- Multiple export formats
- Configurable panel layouts
- Publication-ready graphics
- Supplementary video materials

## Performance Optimization

### Frame Rate Guidelines

**10 FPS** (Default)
- Smooth enough for most visualizations
- Good balance of performance and quality
- Suitable for buildings with 50-100 nodes
- Recommended for interactive mode

**15-20 FPS** (High Quality)
- Very smooth animation
- Requires more processing power
- Ideal for video export
- Best for presentations

**5-8 FPS** (Performance Mode)
- Acceptable for analysis
- Faster rendering
- Suitable for large buildings (100+ nodes)
- Recommended for initial testing

### System Requirements

**Minimum:**
- CPU: Dual-core processor, 2.0 GHz
- RAM: 4 GB
- Python: 3.7+
- Display: 1920×1080 resolution

**Recommended:**
- CPU: Quad-core processor, 3.0 GHz or higher
- RAM: 8 GB or more
- Python: 3.9+
- Display: 2560×1440 or higher resolution
- GPU: Dedicated graphics recommended for large simulations

**Software Dependencies:**
- matplotlib ≥ 3.3.0 (with animation support)
- numpy ≥ 1.19.0
- networkx ≥ 2.5
- PyYAML ≥ 5.3
- ffmpeg (for video export, installed separately)

### Optimization Strategies

**For Large Buildings (100+ nodes):**
```bash
python run_live_simulation.py --fps 5 --no-advanced
```

**For Extended Simulations (>600 seconds):**
```bash
python run_live_simulation.py --save-video --fps 10
```
(Video mode uses less memory than interactive mode)

**For Maximum Quality:**
```bash
python run_live_simulation.py --fps 20 --save-video
```

**For Quick Testing:**
```bash
python run_live_simulation.py --duration 120 --fps 8 --quiet
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Low Frame Rate / Choppy Animation

**Symptoms:** FPS drops below target, stuttering display

**Solutions:**
1. Reduce frame rate: `--fps 5`
2. Disable advanced features: `--no-advanced`
3. Shorten simulation: `--duration 180`
4. Close other applications
5. Use video export mode for offline rendering

#### Issue: Video Export Fails

**Symptoms:** Error message "ffmpeg not found" or codec errors

**Solutions:**

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from ffmpeg.org and add to PATH
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg

# Or for RPM-based systems:
sudo dnf install ffmpeg
```

**Verify Installation:**
```bash
ffmpeg -version
```

#### Issue: Memory Usage Grows Over Time

**Symptoms:** Simulation slows down after several minutes

**Solutions:**
1. Reduce simulation duration
2. Lower frame rate to reduce recorded frames
3. Increase system RAM
4. Use video export mode (releases memory during rendering)

#### Issue: Matplotlib Display Not Responding

**Symptoms:** Window freezes or becomes unresponsive

**Solutions:**
1. Try different matplotlib backend:
   ```python
   import matplotlib
   matplotlib.use('TkAgg')  # or 'Qt5Agg'
   ```
2. Use video export mode instead: `--save-video`
3. Update matplotlib: `pip install --upgrade matplotlib`
4. Check for conflicting GUI frameworks

#### Issue: Colors Don't Display Correctly

**Symptoms:** Nodes all appear the same color, flow not visible

**Solutions:**
1. Update matplotlib to latest version
2. Check color theme compatibility
3. Ensure proper node state initialization
4. Verify hazard data in building configuration

## Output Formats

### Interactive Window

**Features:**
- Live rendering at configured FPS
- Full keyboard and mouse control
- Real-time performance monitoring
- Immediate visual feedback

**Best For:**
- Algorithm development and debugging
- Interactive demonstrations
- Training scenarios
- Quick analysis

### Video Files

**Specifications:**
- Format: MP4 (MPEG-4 Part 14)
- Video Codec: H.264
- Resolution: 2400×1400 pixels (configurable)
- Frame Rate: As specified (typically 10-20 FPS)
- Bitrate: 2000 kbps
- Audio: None (silent)

**File Size Estimates:**
- 5 minutes at 10 FPS: ~6-8 MB
- 10 minutes at 15 FPS: ~15-20 MB
- 30 minutes at 10 FPS: ~25-35 MB

**Best For:**
- Presentations and conferences
- Documentation and reports
- Remote sharing and collaboration
- Archive and playback

### Frame Captures

**Specifications:**
- Format: PNG (Portable Network Graphics)
- Resolution: As displayed (typically 2400×1400)
- DPI: 200 (high quality)
- Color Depth: 24-bit RGB
- Compression: Lossless

**File Size:**
- Typical frame: 500-800 KB

**Best For:**
- Publication figures
- Slide presentations
- High-quality documentation
- Detailed analysis screenshots

## Advanced Features

### Hazard Heat Map Overlay (H Key)

Displays a color gradient overlay showing hazard intensity across the building:
- **Red zones**: Critical hazard areas (severity >0.7)
- **Orange zones**: Moderate hazards (severity 0.3-0.7)
- **Yellow zones**: Low hazards (severity <0.3)
- **Transparent**: No hazards

**Use Cases:**
- Hazard spread visualization
- Risk assessment
- Evacuation route planning
- Safety zone identification

### Flow Direction Arrows (F Key)

Shows directional arrows indicating evacuee flow:
- **Arrow size**: Proportional to flow rate
- **Arrow color**: Matches edge flow coloring
- **Arrow direction**: Points from high to low pressure

**Use Cases:**
- Bottleneck identification
- Flow optimization
- Exit usage analysis
- Path efficiency evaluation

### Speed Control (1-5 Keys)

Adjust playback speed for different analysis needs:
- **0.25× (Key 1)**: Slow motion for detailed observation
- **0.5× (Key 2)**: Moderate slow motion
- **1.0× (Key 3)**: Real-time speed
- **2.0× (Key 4)**: Fast preview
- **5.0× (Key 5)**: Very fast for long simulations

**Note:** Speed changes do not affect simulation accuracy, only playback rate.

## Integration with HASO Framework

### Zone Visualization

The dashboard displays HASO zone assignments in real-time:
- **Zone boundaries**: Implicit from clearance patterns
- **Zone coverage**: Progress bars in Zone Status panel
- **Agent-zone mapping**: Visible through agent trails

### Electrical Flow Model

Full integration with the electrical circuit-based flow model:
- **Real-time calculations**: Flow rates updated each frame
- **Visual feedback**: Edge colors show flow intensity
- **Analytics graphs**: Flow metrics tracked over time
- **Optimization insights**: Bottleneck identification

### Fog of War System

Unknown areas visualized through node colors:
- **Gray nodes**: Unknown (fog state 0)
- **Color transitions**: Real-time discovery updates
- **Vision radius**: Implicit from agent positions
- **Remote sensing**: Camera/radio connections shown

### Dynamic Hazard Propagation

Hazard spread visible in real-time:
- **Fire spread**: Node color changes
- **Smoke generation**: Adjacent node updates
- **Visibility decay**: Alpha transparency changes
- **Warning indicators**: Hazard markers appear

## Comparison with Static Visualization

| Feature | Static (haso_visualizer.py) | Live Dashboard |
|---------|----------------------------|----------------|
| Agent movement | End state only | Real-time tracking |
| Time evolution | Separate snapshots | Continuous animation |
| Interactivity | None | Full control |
| Flow visualization | Final state | Dynamic updates |
| Analysis depth | Post-simulation | During simulation |
| File size | ~1 MB (PNG) | ~6-20 MB (video) |
| Generation time | <10 seconds | 5-30 minutes |
| Best use case | Reports, papers | Demos, training |
| Data access | Static images | Full history API |

**Recommendation**: Use static visualization for final reports and publications; use live dashboard for development, debugging, presentations, and training.

## Future Enhancements

### Planned Features

**True Edge Interpolation**
- Smooth agent movement along edges
- Progress tracking between nodes
- Realistic speed visualization
- Expected: v3.0 release

**Multi-Floor 3D Visualization**
- Vertical building representation
- Floor switching controls
- Stairwell visualization
- 3D camera angles

**Rewind Functionality**
- Backward time navigation
- State restoration at any point
- Frame-by-frame stepping
- Comparison between time points

**Click Interactions**
- Agent info on click
- Node details panel
- Edge properties display
- Interactive tooltips

**Comparison Mode**
- Side-by-side scenario comparison
- A/B algorithm testing
- Performance differential visualization
- Synchronized playback

**Export Enhancements**
- GIF animation export
- SVG vector graphics
- Data export (CSV, JSON)
- Batch processing mode

### Performance Improvements

**GPU Acceleration**
- OpenGL-based rendering
- Shader support for effects
- Hardware-accelerated compositing
- 60+ FPS capability

**Adaptive Level of Detail**
- Automatic complexity reduction
- Distance-based simplification
- Dynamic agent marker sizing
- Intelligent label placement

**Incremental Rendering**
- Only update changed elements
- Smart dirty region tracking
- Partial frame updates
- Reduced CPU usage

## Examples and Tutorials

### Example 1: Basic Usage

```bash
# Start with default office building
python run_live_simulation.py

# Wait for dashboard to load
# Use SPACE to pause at interesting moments
# Press S to save frames
# Press 4 for 2x speed to preview faster
```

### Example 2: Hospital Evacuation Analysis

```bash
# Load hospital configuration
python run_live_simulation.py \
    --map notebooks/data/hospital_wing.yaml \
    --duration 600 \
    --fps 15

# Observe:
# - How scouts explore initially
# - How securers assist evacuees
# - How zones are covered systematically
# - Where bottlenecks occur
```

### Example 3: Algorithm Comparison

```python
# Run multiple scenarios programmatically
scenarios = [
    {'config': 'baseline.yaml', 'seed': 42},
    {'config': 'optimized.yaml', 'seed': 42},
]

for i, scenario in enumerate(scenarios):
    world = build_world('map.yaml', config_path=scenario['config'], 
                       seed=scenario['seed'])
    
    dashboard = create_live_visualization(
        world, save_video=True, 
        video_path=f'scenario_{i}.mp4'
    )
    
    # Compare videos side-by-side
```

### Example 4: Performance Benchmarking

```python
import time

# Test different configurations
configs = [
    {'fps': 5, 'advanced': False},
    {'fps': 10, 'advanced': False},
    {'fps': 10, 'advanced': True},
    {'fps': 20, 'advanced': True},
]

for config in configs:
    start = time.time()
    
    dashboard = create_live_visualization(
        world, fps=config['fps'], duration=180,
        advanced=config['advanced'], save_video=True,
        video_path=f"test_{config['fps']}fps.mp4"
    )
    
    elapsed = time.time() - start
    print(f"Config {config}: {elapsed:.1f}s, "
          f"Avg FPS: {dashboard.actual_fps:.1f}")
```

## Citation and Credits

### Citation

If you use the HASO Live Dashboard in your research, please cite:

```bibtex
@software{haso_dashboard,
  title = {HASO Live Evacuation Simulation Dashboard},
  author = {HASO Research Team},
  year = {2025},
  version = {2.0},
  url = {https://github.com/yourusername/haso}
}
```

### Technology Stack

- **Visualization**: matplotlib with FuncAnimation
- **Computation**: NumPy for numerical operations
- **Graph Processing**: NetworkX for layout algorithms
- **Configuration**: PyYAML for structured data
- **Video Encoding**: FFmpeg for high-quality rendering

### Acknowledgments

Electrical flow visualization inspired by circuit simulation tools and fluid dynamics modeling. Dashboard design follows modern operational monitoring systems used in command centers.

## Support and Resources

### Documentation

- **Getting Started**: See `docs/GETTING_STARTED.md`
- **HASO Algorithm**: See `docs/HASO_ALGORITHM.md`
- **Flow Model**: See `docs/ELECTRICAL_FLOW_MODEL.md`
- **API Reference**: See `docs/README.md`

### Issue Reporting

For bugs, feature requests, or questions:
1. Check existing documentation
2. Review troubleshooting section
3. Submit detailed issue report with:
   - Operating system and version
   - Python version
   - Command used
   - Error messages or unexpected behavior
   - Configuration files (if relevant)

### Performance Tuning Help

Contact the development team for:
- Large-scale simulations (>200 nodes)
- Custom visualization requirements
- Integration with other systems
- Performance optimization consultation

---

**Ready to visualize your evacuation strategy!**

Launch the dashboard:
```bash
python run_live_simulation.py
```

For video export:
```bash
python run_live_simulation.py --save-video --video-path my_evacuation.mp4
```
