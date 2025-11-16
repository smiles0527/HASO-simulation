# Electrical Flow Model for Evacuation Dynamics

The electrical flow model applies circuit theory to quantify evacuee movement and congestion. Physical quantities map to electrical analogs: current I (people/s) represents flow rate, voltage V represents evacuation pressure, and resistance R represents bottleneck constraints.

## Fundamental Principles

Ohm's Law governs flow rate:

\[I = \frac{\Delta V}{R}\]

where \(\Delta V = V_{src} - V_{dst}\) is the pressure gradient driving evacuees from high-pressure areas (hazardous rooms) toward low-pressure areas (exits).

Kirchhoff's Current Law enforces flow conservation at nodes:

\[\sum I_{in} = \sum I_{out} + \frac{dN}{dt}\]

where \(N\) is the number of evacuees in the node.

Series resistance for sequential bottlenecks:

\[R_{total} = \sum_{i=1}^{n} R_i\]

Parallel resistance for alternative routes:

\[\frac{1}{R_{total}} = \sum_{i=1}^{n} \frac{1}{R_i}\]

## Pressure Initialization

Node pressures combine multiple factors:

\[V_i = V_{hazard} + V_{occupancy} + V_{fog}\]

where:
- \(V_{hazard} = 100 \cdot h_i\) (hazard severity \(h_i \in [0,1]\))
- \(V_{occupancy} = 50 \cdot p_i\) (occupancy probability \(p_i\))
- \(V_{fog} = 30 \cdot (1 - f_i/3)\) (fog state \(f_i \in \{0,1,2,3\}\))

Exits serve as ground nodes with \(V_{exit} = 0\), creating pressure gradients that drive evacuee movement.

## Resistance Calculation

Edge resistance follows:

\[R = \rho \cdot \frac{L}{A} \cdot f_{hazard} \cdot f_{visibility}\]

where:
- \(\rho = 1.0\) (base resistivity)
- \(L\) is corridor length
- \(A = w \cdot h\) is cross-sectional area (width × height)
- \(f_{hazard} = 1.0 + \frac{h_{src} + h_{dst}}{2}\) (hazard factor)
- \(f_{visibility} = \frac{1}{\max(0.1, \frac{v_{src} + v_{dst}}{2})}\) (visibility factor)

Non-traversable edges have \(R = \infty\) until cleared.

## Flow Rate Computation

Flow rate between nodes \(i\) and \(j\):

\[I_{ij} = \max\left(0, \min\left(\frac{V_i - V_j}{R_{ij}}, N_i\right)\right)\]

where \(N_i\) is the number of evacuees at node \(i\). Flow is unidirectional (high to low pressure) and bounded by available evacuees.

## Parallel Path Analysis

The model identifies multiple paths between source and destination using NetworkX shortest path algorithms. For each path, total resistance is calculated by summing edge resistances along the path (series combination). Effective resistance for parallel paths uses the reciprocal sum formula, enabling flow distribution across alternative routes.

The implementation limits path enumeration to three parallel paths to balance accuracy and computational efficiency. Paths are ranked by total resistance, with the lowest-resistance path designated as primary route.

## Flow Distribution Optimization

Flow distribution to exit \(k\):

\[p_k = \frac{G_k}{\sum_{j} G_j}\]

where \(G_k = 1/R_k\) is conductance (inverse resistance) for exit \(k\). Higher conductance paths receive proportionally more flow.

## Pressure Dynamics

Node pressures evolve via capacitor discharge:

\[V(t) = V_0 \cdot e^{-t/\tau}\]

where \(\tau = RC\) is the time constant, \(R\) is effective resistance, and \(C\) is node capacity. The time constant:

\[\tau = \frac{1}{I_{out} \cdot C}\]

Higher outflow rates or larger capacities accelerate pressure decay. Exits maintain \(V_{exit} = 0\) as constant sinks.

## Capacity Modeling

Room capacity (capacitance):

\[C_i = \frac{A_i}{2.0}\]

where \(A_i\) is room area in square meters. The 2 m²/person density follows standard occupancy guidelines.

## Bottleneck Identification

Bottleneck factors identify the narrowest point along evacuation paths. The calculation finds the minimum conductance (inverse resistance) across all edges in a path, identifying constraints that limit overall throughput.

Bottlenecks with width less than 1.0 meter are flagged for responder attention, as they significantly restrict flow rates. The model can suggest alternative routes or prioritize bottleneck clearance to improve evacuation efficiency.

## Route Recommendations

The suggest_optimal_routes function provides comprehensive evacuation guidance. It calculates flow distribution percentages for each exit, identifies primary paths with lowest resistance, enumerates backup routes for redundancy, and flags bottlenecks requiring intervention.

Recommendations include expected flow rates, path resistances, bottleneck locations, and parallel path counts. This information supports responder decision-making and automated route assignment for evacuees.

## Integration with Simulation

The FlowDynamics class integrates with the simulation through the graph model. Flow calculations update during evacuee movement phases. Visualization overlays show pressure gradients as color maps and flow rates as arrow thickness. Analytical outputs inform HASO zone partitioning and task allocation.

## Limitations and Assumptions

The model assumes quasi-static flow conditions over short intervals, approximating dynamic evacuation as a series of steady-state snapshots. Evacuee behavior is modeled as fluid flow rather than individual decisions, suitable for large-scale analysis but less accurate for small groups. Resistance calculations use simplified formulas that may not capture psychological effects or complex obstacle interactions.
