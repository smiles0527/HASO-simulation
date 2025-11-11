# NOTE: THIS FILE IS AI GENERATED

# How to Run the Simulation - Step by Step

## Method 0: Scenario Launcher (interactive menu)

```bash
python launch_simulation.py
```

This opens a start-menu style interface where you can pick a curated scenario,
tweak responder counts, scale hazards or evacuee density, and choose between a
live dashboard or a headless run with automatic reporting. It is the fastest
way to explore the new Innovation Hub, Medical Pavilion, and Transit Atrium
layouts.

---

## Method 1: Quick Test (30 seconds)

**Open your terminal in this folder and run:**

```bash
python test_simulation.py
```

This will:
- Test all imports
- Load the office building map
- Run a 30-second simulation
- Show results

**Expected output:** You'll see agent movement and simulation results.

---

## Method 2: Full Simulation (5 minutes)

```bash
python test_full_simulation.py
```

This will:
- Run a complete 5-minute evacuation
- Show detailed agent performance
- Display which rooms were cleared

---

## Method 3: Visual Demo (with plots)

```bash
python demo_visual.py
```

This will:
- Run the simulation
- Create 4 visualization files in `demo_results/`
- Show an interactive dashboard
- Display graphs of progress

**Output files:**
- `1_building_layout.png` - Building map
- `2_clearance_progress.png` - Progress over time
- `3_agent_paths.png` - Agent movements
- `4_complete_dashboard.png` - Full summary

---

## Method 4: Interactive Jupyter Notebook (Recommended)

This is the recommended way to explore the simulation.

### Step 1: Install Jupyter Lab
```bash
pip install jupyterlab
```

### Step 2: Launch Jupyter
```bash
jupyter lab
```

### Step 3: Open the Demo Notebook
In Jupyter Lab, navigate to:
```
notebooks/simulation_demo.ipynb
```

### Step 4: Run the Cells
Click **Cell → Run All** or press `Shift+Enter` on each cell

You'll see:
- Building visualizations
- Agent movements
- Performance graphs
- Statistical analysis

---

## Method 5: Python Script (Custom)

Create a file called `my_simulation.py`:

```python
from notebooks import simulate
from notebooks.src import generate_summary_report

# Run simulation
results = simulate(
    map_path="notebooks/data/office_building_simple.yaml",
    config_path="notebooks/data/config_baseline.yaml",
    tmax=600,  # 10 minutes
    seed=42
)

# Get results
world = results['world']
print(f"Simulation time: {world.time:.1f}s")
print(f"All rooms cleared: {results['all_cleared']}")

# Generate report
report = generate_summary_report(world)
print(report)
```

Then run:
```bash
python my_simulation.py
```

---

## Method 6: Python Interactive Shell

```bash
python
```

Then type:
```python
>>> from notebooks import simulate
>>> results = simulate("notebooks/data/office_building_simple.yaml", tmax=300)
>>> world = results['world']
>>> print(f"Time: {world.time:.1f}s")
>>> cleared, total = world.G.get_cleared_count()
>>> print(f"Cleared {cleared}/{total} rooms")
```

---

## Recommended Path for Beginners

1. **First time?** Run: `python test_simulation.py`
2. **Want visuals?** Run: `python demo_visual.py`
3. **Want to explore?** Run: `jupyter lab` → open `simulation_demo.ipynb`
4. **Ready to experiment?** Create your own maps and run simulations!

---

## Files You Can Modify

### Building Maps:
- `notebooks/data/office_building_simple.yaml` - Small office (11 nodes, 7 rooms)
- `notebooks/data/hospital_wing.yaml` - Hospital with patients (15 nodes, 9 rooms)

### Configuration:
- `notebooks/data/config_baseline.yaml` - Agent setup and parameters

### Try editing:
- Number of agents
- Agent roles (Scout, Securer, Checkpointer, Evacuator)
- Sweep strategies (right, left, corridor)
- Room priorities
- Hazard locations

---

## Troubleshooting

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Error: "matplotlib not found" (for visual demo)
```bash
pip install matplotlib
```

### Error: Import errors
Make sure you're in the project root directory:
```bash
cd "C:\Users\iscur\OneDrive\My Surface docs\Desktop\HIMCM-Topic-A-2025"
```

### Simulation runs but no rooms cleared
This is normal for very short simulations (< 60 seconds). Try:
- Increase `tmax` to 300 or 600 seconds
- Check agent logs to see what they're doing

---

## What You'll See

When you run a simulation, you'll see output like:

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

---

## Next Steps

After running your first simulation:

1. **Analyze Results**: Look at agent logs, clearance rates, distances
2. **Try Different Scenarios**: Hospital, office, custom buildings
3. **Experiment**: Change sweep strategies, team composition
4. **Optimize**: Find the fastest way to clear all rooms
5. **Visualize**: Create charts and animations
6. **Extend**: Add new features, hazard models, evacuee behaviors

---

## Quick Tips

- **Start small**: Use `office_building_simple.yaml` first
- **Use Jupyter**: Best for visualization and experimentation
- **Check logs**: Agent logs show what they're doing
- **Adjust time**: Use `tmax=600` (10 min) for complete clearance
- **Save results**: Visualizations save to `demo_results/` folder

---

## Ready to Go

You have everything you need. Pick a method above and start simulating.

**Fastest start:**
```bash
python test_simulation.py
```

**Best experience:**
```bash
jupyter lab
# Then open: notebooks/simulation_demo.ipynb
```

**Need help?** See `README.md` or `GETTING_STARTED.md`

