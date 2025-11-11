# Quick Start - Improved Map-Style Animation

## What's New?

Your evacuation simulation now has a **smooth, map-like visualization** that looks like an actual building floor plan with moving dots representing responders!

## Visual Changes

### Before vs After

**BEFORE:**
- Rooms shown as scatter plot circles/markers
- Agents as large circles with complex styling
- Jumpy movement between positions
- Harder to read building layout

**AFTER:**
- ✅ Rooms drawn as actual **rectangles** (like a real floor plan)
- ✅ Corridors are elongated rectangles
- ✅ Exits are clearly marked blue squares
- ✅ Agents are **smooth-moving colored dots** with trails
- ✅ **Cubic easing interpolation** for natural movement
- ✅ Higher FPS (20 instead of 10) for fluid animation
- ✅ Map-like background with subtle grid

## How to Run

### Basic (Recommended)
```bash
py run_live_simulation.py
```

This will:
- Run at 20 FPS for smooth animation
- Show the office building layout
- Display 4 agents moving smoothly through rooms
- Run for 5 minutes (300 seconds)

### Custom Settings

**Shorter demo (1 minute):**
```bash
py run_live_simulation.py --duration 60
```

**Super smooth (30 FPS):**
```bash
py run_live_simulation.py --fps 30
```

**Hospital layout:**
```bash
py run_live_simulation.py --map notebooks/data/hospital_wing.yaml
```

**Save as video:**
```bash
py run_live_simulation.py --save-video --video-path my_animation.mp4
```

## Visual Elements

### Rooms (Rectangles)
- 🟦 **Blue squares** = EXIT points (safe zones)
- ⬜ **Light gray rectangles** = Corridors (elongated)
- 🟩 **Green squares** = Cleared rooms (with ✓)
- 🟨 **Yellow squares** = Room being cleared (agent inside)
- 🟥 **Red squares** = Uncleared rooms

### Agents (Smooth Moving Dots)
- 🟢 **Green dots** = SCOUT (fast reconnaissance)
- 🔵 **Blue dots** = SECURER (assist evacuation)
- 🟡 **Yellow dots** = CHECKPOINTER (secure areas)
- 🟣 **Purple dots** = EVACUATOR (final sweep)

Each agent has:
- Smooth interpolated movement (no jumping!)
- Colored trail showing their path
- Small badge below showing their role
- ID number inside the dot

## Interactive Controls

While animation is running:

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume |
| `1` | 0.25x speed (slow motion) |
| `2` | 0.5x speed |
| `3` | 1.0x speed (normal) |
| `4` | 2.0x speed (fast) |
| `5` | 5.0x speed (very fast) |
| `S` | Save current frame as PNG |
| `R` | Reset to beginning |
| `ESC` | Exit |

## Technical Improvements

### 1. Smooth Movement
- **Cubic easing interpolation** instead of linear
- Movement speed varies by agent status:
  - NORMAL: 0.12 units/frame (smooth)
  - SLOWED: 0.06 units/frame (slower)
  - PROGRESSING: 0.03 units/frame (clearing room)

### 2. Map-Like Visualization
- Actual `Rectangle` patches for rooms (not scatter markers)
- Room size based on `area` attribute from YAML
- Subtle dashed lines for connections
- Light grid background (#F8F9FA)

### 3. Performance Optimized
- Agents updated every frame (smooth movement)
- Room colors updated only when status changes
- Trails only add points when agent moves >0.05 units
- Edges updated every 10 frames
- Statistics updated every 30 frames

### 4. Higher Default FPS
- Increased from 10 FPS to 20 FPS
- Results in much smoother animation
- Still performs well on most systems

## Example Output

The animation will show:
- **Main Panel (left)**: Map-style floor plan with moving dots
- **Progress Graph (top right)**: Clearance progress over time
- **Team Status (middle right)**: Live agent information
- **Statistics Bar (bottom)**: Key metrics

## Troubleshooting

**Animation looks choppy:**
- Try increasing FPS: `--fps 30`
- Close other programs to free up CPU

**Animation too fast:**
- Press `1` or `2` to slow down playback
- Or use `--fps 10` for lower framerate

**Can't see agents:**
- They're smaller now (1.2 unit radius dots)
- Look for colored circles with numbers inside
- They leave colored trails behind them

**Rooms look weird:**
- Make sure your YAML file has `x`, `y`, and `area` attributes
- Check that coordinates make sense for your building

## Files Modified

- `notebooks/src/animate_live.py` - Main visualization improvements
- `run_live_simulation.py` - Updated default FPS to 20
- `ANIMATION_IMPROVEMENTS.md` - Detailed technical documentation

## Next Steps

1. **Run the animation**: `py run_live_simulation.py`
2. **Try different buildings**: Use `--map` to load different layouts
3. **Export videos**: Use `--save-video` for presentations
4. **Experiment with FPS**: Find the best balance for your system

Enjoy your smooth, map-like evacuation animation! 🎉

