# HASO Algorithm: Hierarchical Adaptive Search and Optimization

The Hierarchical Adaptive Search and Optimization (HASO) algorithm provides a systematic framework for coordinating multi-responder evacuation sweeps. HASO partitions buildings into responder zones, assigns agents to zones based on workload and capabilities, and dynamically adapts to changing conditions such as hazard propagation and agent failures.

## Overview

HASO operates through four main steps: responder assessment, zone assignment, hierarchical partitioning, and dynamic adaptation. The algorithm minimizes expected completion time while ensuring balanced workloads and complete building coverage.

## Step 1: Responder Assessment

The algorithm begins by evaluating each responder's capabilities, including base speed, role specialization, and priority heuristic. Scouts receive higher priority for exploration tasks, while Securers are prioritized for hazard mitigation. Each agent's efficiency is calculated as the product of base speed and priority heuristic, which determines workload allocation.

## Step 2: Zone Assignment

Responders are assigned using ILP optimization or greedy assignment. The ILP minimizes:

\[\min \sum_{i,j} t_{ij} \cdot x_{ij}\]

subject to zone coverage and agent capacity constraints, where \(t_{ij}\) is expected completion time for agent \(i\) in zone \(j\), and \(x_{ij} \in \{0,1\}\) indicates assignment.

The greedy fallback assigns agent \(i\) to zone \(j^*\) where:

\[j^* = \arg\max_j \frac{W_j}{L_j + E_i}\]

where \(W_j\) is zone workload, \(L_j\) is current load, and \(E_i = s_i \cdot h_i\) is agent efficiency (speed × priority heuristic).

## Step 3: Hierarchical Partitioning

Building graphs are partitioned using community detection (greedy modularity) to minimize inter-zone edges. Node importance:

\[w_i = (6 - p_i) \cdot A_i\]

where \(p_i \in [1,5]\) is room priority and \(A_i\) is area. Edge weights represent traversal time: \(w_{ij} = L_{ij} / T_{ij}\) where \(L_{ij}\) is length and \(T_{ij}\) is traversability.

If fewer communities than zones are detected, largest zones are recursively split. Balancing redistributes nodes to equalize workloads using balance weight \(\alpha \in [0,1]\).

## Step 4: Dynamic Adaptation

HASO adapts to changing conditions through dynamic zone reallocation. When an agent fails or slows significantly, their assigned zone is reallocated to the nearest active agent. The algorithm calculates the centroid of the failed zone and identifies the closest active agent based on Euclidean distance.

Zone reallocation merges the failed zone into the closest agent's existing zone, ensuring continuous coverage without gaps. This mechanism handles scenarios such as agent incapacitation, communication failures, or unexpected delays.

## Hazard Propagation

Fire spreads to neighbor \(j\) with probability:

\[P_{spread} = p_{base} \cdot h_i(t)\]

where \(p_{base}\) is base spread probability and \(h_i(t)\) is fire severity. Fire intensity grows:

\[h_i(t+\Delta t) = \min\left(1.0, h_i(t) + \Delta t \cdot \frac{k_i}{100}\right)\]

where \(k_i\) is fire spread rate. Smoke visibility decays:

\[v_i(t+\Delta t) = v_i(t) \cdot e^{-\gamma \Delta t / 10}\]

where \(\gamma\) is smoke decay rate.

## Path Planning Integration

Modified A* cost function:

\[c_{ij} = d_{ij} + \beta \cdot h_j + \lambda \cdot (1 - v_j)\]

where \(d_{ij}\) is distance, \(h_j\) is hazard severity, \(v_j\) is visibility, and \(\beta, \lambda\) are penalty weights. Agents prefer intra-zone paths but can traverse inter-zone edges when necessary.

## Zone Balancing

Zone workload:

\[W_j = \sum_{i \in Z_j} t_i \cdot A_i \cdot (1 + 2h_i) \cdot (1 + p_i)\]

where \(t_i\) is base search time, \(A_i\) is area, \(h_i\) is hazard severity, and \(p_i\) is occupancy probability. Balancing moves nodes from zones with \(W_j > \bar{W}(1+\alpha)\) to zones with \(W_j < \bar{W}(1-\alpha)\), where \(\bar{W}\) is average workload and \(\alpha \in [0,1]\) is balance weight.

## Implementation Details

The zone_optimizer module provides the core HASO implementation. The partition_building_zones function performs hierarchical partitioning, while assign_responders_to_zones handles agent assignment. The reallocate_zone_dynamic function implements dynamic reallocation.

When NetworkX is unavailable, the algorithm falls back to simple spatial partitioning based on node coordinates. This ensures the simulator remains functional even without optional dependencies, though with reduced partitioning quality.

Zone assignments are stored in the world's zones dictionary (zone_id to node list) and agent_zones dictionary (agent_id to zone_id). Each node maintains a zone_id attribute for efficient lookup during path planning and task allocation.

## Configuration

HASO is enabled by default in the simulator configuration. The use_haso flag in the configuration YAML controls whether zone partitioning is performed. When disabled, agents operate without zone constraints, using global task allocation instead.

Zone partitioning occurs during world initialization via the init_zones method. The number of zones defaults to the number of agents but can be explicitly specified. Initialization failures are handled gracefully, with warnings logged and simulation continuing without zone partitioning.

## Performance Considerations

Community detection has complexity \(O(n \log n)\) for \(n\) nodes. Zone balancing converges within ~10 iterations. ILP assignment provides optimal solutions but requires ortools or pulp. The greedy fallback runs in \(O(a \cdot z)\) time for \(a\) agents and \(z\) zones.
