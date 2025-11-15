"""
Electrical Circuit-Based Flow Dynamics for Evacuation Modeling

Applies electrical circuit principles to model evacuee flow:
- Current (I) = Evacuee flow rate (people/second)
- Voltage (V) = Evacuation pressure (hazard severity + urgency)
- Resistance (R) = Bottleneck width, congestion, obstacles

Key principles:
1. Ohm's Law: I = V / R
2. Kirchhoff's Current Law: Flow conservation at nodes
3. Series resistance: R_total = R1 + R2 + ... (sequential bottlenecks)
4. Parallel resistance: 1/R_total = 1/R1 + 1/R2 + ... (parallel paths)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .graph_model import Graph, Node, Edge


class FlowDynamics:
    """
    Electrical circuit model for evacuee flow dynamics.
    """
    
    def __init__(self, graph: Graph):
        self.graph = graph
        
        # Flow state
        self.node_pressure: Dict[int, float] = {}  # Voltage at each node
        self.edge_flow: Dict[Tuple[int, int], float] = {}  # Current through each edge
        self.node_capacity: Dict[int, float] = {}  # Capacitance (room capacity)
        
        # Initialize
        self._initialize_pressures()
        self._initialize_capacities()
    
    def _initialize_pressures(self):
        """
        Initialize evacuation pressure (voltage) at each node.
        Exits have 0V (ground), hazardous areas have high V (source).
        """
        for node_id, node in self.graph.nodes.items():
            if node.node_type.name == 'EXIT':
                # Exits are ground (0V)
                self.node_pressure[node_id] = 0.0
            else:
                # Base pressure from hazard severity
                hazard_pressure = node.hazard_severity * 100.0  # Scale to voltage
                
                # Add pressure from occupancy
                occupancy_pressure = node.occupancy_probability * 50.0
                
                # Add urgency based on fog of war (unknown areas = higher pressure)
                fog_pressure = (1 - node.fog_state / 3.0) * 30.0 if node.fog_state >= 0 else 0.0
                
                # Total pressure (voltage)
                self.node_pressure[node_id] = hazard_pressure + occupancy_pressure + fog_pressure
    
    def _initialize_capacities(self):
        """
        Initialize room capacity (capacitance).
        Larger rooms can hold more people temporarily.
        """
        for node_id, node in self.graph.nodes.items():
            # Capacity in people (based on area)
            area = node.area if hasattr(node, 'area') and node.area > 0 else 10.0
            self.node_capacity[node_id] = area / 2.0  # ~2 m² per person
    
    def calculate_edge_resistance(self, edge: Edge, src_node: Node, dst_node: Node) -> float:
        """
        Calculate resistance of an edge (corridor/doorway).
        
        R = ρ * L / A
        where:
        - ρ (rho) = resistivity (base congestion factor)
        - L = length of corridor
        - A = effective width (cross-sectional area)
        """
        # Base resistivity (congestion factor)
        rho = 1.0
        
        # Hazard increases resistance (smoke, fire)
        hazard_factor = 1.0 + (src_node.hazard_severity + dst_node.hazard_severity) / 2.0
        rho *= hazard_factor
        
        # Visibility affects resistance
        visibility_factor = 1.0 / max(0.1, (src_node.visibility + dst_node.visibility) / 2.0)
        rho *= visibility_factor
        
        # Length (longer corridors = more resistance)
        length = edge.length if hasattr(edge, 'length') else 10.0
        
        # Width (narrower = more resistance)
        width = edge.width if hasattr(edge, 'width') else 1.5
        cross_section = width * 2.5  # Assume 2.5m height
        
        # Calculate resistance: R = ρ * L / A
        resistance = rho * length / cross_section
        
        # Non-traversable edges have infinite resistance
        if not edge.traversable:
            resistance = float('inf')
        
        return max(0.1, resistance)  # Minimum resistance
    
    def calculate_flow_rate(self, src_id: int, dst_id: int) -> float:
        """
        Calculate flow rate between two nodes using Ohm's Law.
        
        I = ΔV / R
        where:
        - I = flow rate (people/second)
        - ΔV = pressure difference (voltage drop)
        - R = edge resistance
        """
        # Get edge
        edge = self.graph.edges.get((src_id, dst_id))
        if not edge:
            edge = self.graph.edges.get((dst_id, src_id))
        if not edge:
            return 0.0
        
        # Get nodes
        src_node = self.graph.get_node(src_id)
        dst_node = self.graph.get_node(dst_id)
        if not src_node or not dst_node:
            return 0.0
        
        # Voltage difference (pressure gradient)
        delta_V = self.node_pressure.get(src_id, 0) - self.node_pressure.get(dst_id, 0)
        
        # Resistance
        R = self.calculate_edge_resistance(edge, src_node, dst_node)
        
        # Ohm's Law: I = V / R
        if R == float('inf') or R <= 0:
            return 0.0
        
        flow = delta_V / R
        
        # Flow is directional (only positive flow in direction of decreasing pressure)
        flow = max(0.0, flow)
        
        # Limit by number of evacuees available at source
        evacuees_at_src = len(src_node.evacuees) if src_node.evacuees else 0
        flow = min(flow, evacuees_at_src)
        
        return flow
    
    def calculate_parallel_paths(self, src_id: int, dst_id: int) -> List[Tuple[List[int], float]]:
        """
        Find parallel paths between source and destination.
        Calculate effective resistance using parallel resistance formula.
        
        For parallel paths: 1/R_eff = 1/R1 + 1/R2 + ... + 1/Rn
        """
        import networkx as nx
        
        # Build NetworkX graph for pathfinding
        G_nx = nx.Graph()
        for (u, v), edge in self.graph.edges.items():
            if edge.traversable:
                src_node = self.graph.get_node(u)
                dst_node = self.graph.get_node(v)
                if src_node and dst_node:
                    R = self.calculate_edge_resistance(edge, src_node, dst_node)
                    G_nx.add_edge(u, v, resistance=R)
        
        # Find multiple paths (up to 3 parallel paths)
        try:
            paths = []
            for path in nx.shortest_simple_paths(G_nx, src_id, dst_id, weight='resistance'):
                if len(paths) >= 3:
                    break
                
                # Calculate total resistance for this path (series)
                path_resistance = 0.0
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    edge_data = G_nx.get_edge_data(u, v)
                    if edge_data:
                        path_resistance += edge_data['resistance']
                
                paths.append((path, path_resistance))
            
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def calculate_effective_resistance(self, paths: List[Tuple[List[int], float]]) -> float:
        """
        Calculate effective resistance for parallel paths.
        
        1/R_eff = 1/R1 + 1/R2 + ... + 1/Rn
        """
        if not paths:
            return float('inf')
        
        # Sum of reciprocals
        reciprocal_sum = sum(1.0 / R for _, R in paths if R > 0 and R != float('inf'))
        
        if reciprocal_sum == 0:
            return float('inf')
        
        return 1.0 / reciprocal_sum
    
    def optimize_flow_distribution(self, src_id: int, exit_ids: List[int]) -> Dict[int, float]:
        """
        Optimize evacuee distribution to multiple exits using parallel resistance.
        
        Returns: Dictionary mapping exit_id -> proportion of evacuees to send there
        """
        # Find paths to each exit
        exit_paths = {}
        exit_resistances = {}
        
        for exit_id in exit_ids:
            paths = self.calculate_parallel_paths(src_id, exit_id)
            if paths:
                R_eff = self.calculate_effective_resistance(paths)
                exit_paths[exit_id] = paths
                exit_resistances[exit_id] = R_eff
        
        if not exit_resistances:
            return {}
        
        # Calculate conductance (inverse of resistance)
        # Higher conductance = easier path = more flow
        conductances = {exit_id: 1.0/R for exit_id, R in exit_resistances.items() 
                       if R > 0 and R != float('inf')}
        
        total_conductance = sum(conductances.values())
        
        if total_conductance == 0:
            return {}
        
        # Distribute flow proportionally to conductance
        flow_distribution = {exit_id: G / total_conductance 
                            for exit_id, G in conductances.items()}
        
        return flow_distribution
    
    def update_pressures(self, dt: float):
        """
        Update node pressures based on evacuee movement (like charge dissipation).
        
        Uses capacitor discharge model: V(t) = V0 * e^(-t/RC)
        """
        for node_id, node in self.graph.nodes.items():
            if node.node_type.name == 'EXIT':
                continue
            
            # Calculate discharge rate based on outgoing flow
            neighbors = self.graph.neighbors(node_id)
            total_outflow = sum(self.calculate_flow_rate(node_id, neighbor) 
                              for neighbor in neighbors)
            
            # Capacitance (room capacity)
            C = self.node_capacity.get(node_id, 1.0)
            
            # Voltage decay (pressure relief as evacuees leave)
            if total_outflow > 0:
                # Time constant: τ = RC
                tau = 1.0 / (total_outflow * C) if C > 0 else 1.0
                decay_factor = np.exp(-dt / max(tau, 0.1))
                
                current_pressure = self.node_pressure.get(node_id, 0)
                self.node_pressure[node_id] = current_pressure * decay_factor
    
    def get_flow_metrics(self) -> Dict[str, float]:
        """
        Calculate flow performance metrics.
        """
        total_flow = sum(abs(flow) for flow in self.edge_flow.values())
        max_flow = max(self.edge_flow.values()) if self.edge_flow else 0
        avg_pressure = np.mean(list(self.node_pressure.values()))
        
        return {
            'total_flow': total_flow,
            'max_flow': max_flow,
            'avg_pressure': avg_pressure,
            'num_active_paths': len([f for f in self.edge_flow.values() if f > 0.1])
        }


def calculate_bottleneck_factor(graph: Graph, path: List[int]) -> float:
    """
    Calculate bottleneck factor for a path (minimum conductance).
    
    Like finding the narrowest point in a series circuit.
    """
    min_conductance = float('inf')
    
    for i in range(len(path) - 1):
        src_id, dst_id = path[i], path[i+1]
        edge = graph.edges.get((src_id, dst_id)) or graph.edges.get((dst_id, src_id))
        
        if edge:
            # Conductance = 1 / Resistance
            width = edge.width if hasattr(edge, 'width') else 1.5
            conductance = width * 2.0  # Simplified conductance
            min_conductance = min(min_conductance, conductance)
    
    return min_conductance if min_conductance != float('inf') else 0.0


def suggest_optimal_routes(graph: Graph, src_id: int, exit_ids: List[int]) -> Dict[str, any]:
    """
    Suggest optimal evacuation routes using electrical flow analysis.
    
    Returns recommendations for:
    - Primary route (lowest resistance)
    - Backup routes (parallel paths)
    - Flow distribution percentages
    """
    flow_model = FlowDynamics(graph)
    
    # Get flow distribution to exits
    distribution = flow_model.optimize_flow_distribution(src_id, exit_ids)
    
    # Find best paths
    recommendations = {
        'flow_distribution': distribution,
        'routes': {},
        'bottlenecks': []
    }
    
    for exit_id, proportion in distribution.items():
        paths = flow_model.calculate_parallel_paths(src_id, exit_id)
        
        if paths:
            # Primary path (lowest resistance)
            primary_path, primary_R = min(paths, key=lambda x: x[1])
            
            # Calculate bottleneck
            bottleneck = calculate_bottleneck_factor(graph, primary_path)
            
            recommendations['routes'][exit_id] = {
                'primary_path': primary_path,
                'resistance': primary_R,
                'expected_flow': proportion,
                'bottleneck_width': bottleneck,
                'parallel_paths': len(paths)
            }
            
            if bottleneck < 1.0:
                recommendations['bottlenecks'].append({
                    'exit': exit_id,
                    'width': bottleneck,
                    'path': primary_path
                })
    
    return recommendations

