# Performance Improvements Summary

## Changes Made to Fix Lag and Visual Issues

### 1. **Real Building Layout** ✓
- Now uses actual x,y coordinates from YAML files  
- No more circular fallback layout
- Looks like a real building floor plan (hospital/office)

### 2. **Performance Optimizations** ✓
- Removed time slider (major lag source)
- Reduced update frequency:
  - Agents: Every frame (smooth movement)
  - Nodes: Every frame (lightweight)
  - Edges: Every 10 frames
  - Graphs/Text: Every 30 frames
- Limited agent trail length to 100 points
- Only update when significant movement occurs
- Smaller figure size (18x10 instead of 22x12)
- Simplified node update logic

### 3. **Visual Style** ✓
- Corridors shown as thick dark lines
- Open doors as green connections
- Closed doors as red dotted lines
- Rooms as squares, Exits as diamonds
- Hazards with warning symbols
- Color-coded room statuses

### 4. **Informative Labels** ✓
- Clear panel titles explaining purpose
- Room clearance progress tracking
- Team status with zone assignments
- Evacuation metrics (rooms/min, flow rates)
- Real-time time display

## How to Test

Run with hospital layout for best visual effect:

```bash
python run_live_simulation.py --map notebooks/data/hospital_wing.yaml --duration 120 --fps 25
```

For office layout:

```bash
python run_live_simulation.py --map notebooks/data/office_building_simple.yaml --duration 90 --fps 30
```

## Current Status

✅ Building layout is correct (uses YAML coordinates)
✅ Visual design is informative and professional
⚠️  Performance improved but still needs work (FPS still low)
✅ Simulation logic is working (agents scheduled correctly)

## Known Issues

1. FPS is still around 0.3-0.5 (target is 15-30)
2. Matplotlib is inherently slow for real-time animation
3. May need to consider alternative rendering (Pygame, Pand3D, or web-based)

## Recommendations

For production HiMCM paper:
- Use static visualizations (existing haso_visualizer.py works great)
- Export video at low FPS (5-10) and speed up in post-processing
- Consider web-based visualization for interactive demos

