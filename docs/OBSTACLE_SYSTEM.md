# Obstacle and Hazard System

The obstacle and hazard system models physical barriers, dynamic obstacles, and environmental hazards that affect responder movement, visibility, and evacuee safety. The system supports multiple edge types with distinct traversal properties, dynamic state changes, and hazard propagation mechanisms.

## Edge Types and Properties

The simulator defines several edge types representing different physical connections between building nodes. Hallways are standard traversable connections with full visibility and normal speed. Corridors function similarly but may have different width or length characteristics affecting flow capacity.

Doors come in three variants: open doors are immediately traversable with no delay, closed doors require opening actions taking approximately 5 seconds, and locked doors require breaking through taking approximately 10 seconds. Once opened or broken, doors become traversable but may retain reduced speed modifiers.

Walls are non-traversable barriers that block both movement and vision. Stairs are traversable but impose a 30% speed reduction to model vertical movement challenges. Radio camera edges provide remote vision without physical traversal, enabling responders to observe connected nodes without entering them.

## Dynamic State Management

Edges maintain dynamic state that changes during simulation. The is_open flag indicates whether doors are currently open, while is_broken indicates whether obstacles have been forcibly cleared. The opened_by field tracks which agent performed the opening or breaking action, and opened_at records the timestamp for coordination purposes.

Edge properties are initialized from edge type definitions but can be modified during simulation. The can_traverse method checks current traversability by evaluating base properties and dynamic state. For doors, traversability depends on whether they have been opened or broken, while walls remain permanently non-traversable.

## Traversal Mechanics

Responders can interact with obstacles through open and break actions. Opening closed doors requires the agent to remain at the door location for the specified open_time duration. Breaking through locked doors or other obstacles requires the break_time duration and may consume agent resources.

Once an edge is opened or broken, it becomes permanently traversable for all agents, modeling shared access. The system tracks which agent performed the action for logging and coordination purposes, though the benefit applies globally.

Speed modifiers affect traversal time even after obstacles are cleared. A door that was previously locked may retain a 0.9 speed modifier, representing damage or residual difficulty. These modifiers are applied during path cost calculations in the A* search algorithm.

## Vision and Fog of War

Edge properties control vision propagation through the fog of war system. The blocks_vision flag determines whether an edge prevents line-of-sight observation. Closed and locked doors block vision until opened, while walls permanently block vision. Radio camera edges provide remote vision without requiring physical traversal.

The gives_vision property indicates whether traversing an edge reveals connected nodes. Most traversable edges provide vision, enabling agents to update fog of war state when moving. Non-traversable edges that block vision prevent agents from learning about nodes beyond the barrier until the obstacle is cleared.

## Hazard Types

The system models multiple hazard types affecting node safety and responder effectiveness. Smoke reduces visibility and increases traversal difficulty. Fire poses immediate danger and can spread to adjacent nodes. Heat creates uncomfortable conditions that slow movement and increase search time.

Biohazard, explosive, chemical, and radioactive hazards represent specialized threat types requiring specific response protocols. Each hazard type can have associated severity levels ranging from 0.0 (none) to 1.0 (maximum), affecting various simulation parameters.

## Hazard Severity Effects

Search time scales with hazard severity:

\[t_{search} = t_{base} \cdot (1 + 2h_i)\]

where \(h_i \in [0,1]\) is hazard severity. Maximum severity triples base search duration. Pathfinding cost includes hazard penalties: \(c_{ij} = d_{ij} + \beta \cdot h_j\) where \(\beta\) is hazard penalty weight.

## Hazard Propagation

Fire spreads to neighbor \(j\) with probability:

\[P_{spread} = p_{base} \cdot h_i(t)\]

where \(p_{base} = 0.15\) is base spread probability. Fire intensity grows:

\[h_i(t+\Delta t) = \min\left(1.0, h_i(t) + \Delta t \cdot \frac{k_i}{100}\right)\]

where \(k_i\) is fire spread rate. Smoke visibility decays:

\[v_i(t+\Delta t) = v_i(t) \cdot e^{-\gamma \Delta t / 10}\]

where \(\gamma\) is smoke decay rate.

## Obstacle Interaction Policies

Different responder roles have varying capabilities for obstacle interaction. Securers are typically assigned to open doors and clear obstacles as part of their role. Scouts may bypass obstacles to reach exploration targets, while Checkpointers focus on maintaining open paths at strategic locations.

The task allocation system considers obstacle states when ranking room priorities. Rooms behind locked doors receive lower immediate priority unless they contain high-value targets, encouraging systematic obstacle clearance before deep exploration.

## Edge Property Lookup

Edge properties are defined in the EDGE_TYPE_PROPERTIES dictionary, providing centralized configuration for all edge behaviors. Each edge type maps to a dictionary containing base_traversable, blocks_vision, speed_modifier, can_open, can_break, open_time, break_time, and provides_remote_vision flags.

The get_properties method on Edge instances retrieves these definitions, enabling policy code to query edge capabilities. This design allows easy extension with new edge types by adding entries to the properties dictionary.

## Integration with Pathfinding

The A* pathfinding algorithm respects edge traversability when computing shortest paths. Non-traversable edges are excluded from the search graph until opened or broken. Path costs include traversal time based on edge length, speed modifiers, and hazard penalties.

The shortest_path_haso method incorporates obstacle states into cost calculations, ensuring paths account for doors that must be opened and obstacles that must be cleared. Agents receive tasks to open doors when necessary for reaching target rooms.

## Visualization

Obstacles are visualized in the building layout with distinct styling. Walls appear as solid barriers, doors show open or closed states with appropriate icons, and locked doors are highlighted to indicate breaking requirements. The visualization updates dynamically as obstacles are cleared during simulation.

Hazard overlays show severity levels using color gradients, with red indicating fire, gray indicating smoke, and specialized colors for other hazard types. Severity intensity is represented by color saturation, providing immediate visual feedback on threat levels.

## Configuration

Edge types are specified in building YAML files using the edge_type field. The system supports string names that map to EdgeType enum values, enabling human-readable configuration. Default edge type is HALLWAY for unspecified connections.

Hazard types and severities are specified per node in the building definition. Initial hazard states can be set, and the simulation updates them dynamically based on propagation rules. Configuration parameters control fire spread probability, smoke decay rate, and other propagation characteristics.

## Performance Considerations

Obstacle state updates occur during discrete event processing, with minimal computational overhead. The system uses efficient lookups for edge properties and state checks, ensuring pathfinding performance remains acceptable even with many obstacles.

Hazard propagation uses neighbor iteration with probabilistic checks, scaling linearly with the number of nodes. The update_hazards method is called periodically rather than every timestep, reducing computational cost while maintaining realistic dynamics.

## Future Enhancements

Potential improvements include multi-stage obstacle clearing requiring multiple agents, timed obstacles that close automatically after opening, and conditional obstacles that appear based on simulation events. Integration with the electrical flow model could provide congestion-aware obstacle prioritization, and machine learning could optimize obstacle clearing sequences for minimum completion time.
