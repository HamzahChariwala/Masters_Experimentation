"""
Behaviour specification module for analyzing and generating navigation paths in MiniGrid environments.
"""

# Import and expose the log module functions
from Behaviour_Specification.log import (
    analyze_navigation_graphs, 
    export_path_data_to_json,
    generate_env_tensor_from_file
)

# Import from graph_analysis for backward compatibility
from Behaviour_Specification.graph_analysis import analyze_navigation_graphs as analyze_navigation_graphs_legacy

# Import from state_generation module
from Behaviour_Specification.state_generation import (
    State,
    analyze_state_nodes,
    filter_state_nodes
)

# Import from dijkstras module
from Behaviour_Specification.dijkstras import (
    create_graphs_from_nodes,
    compute_shortest_paths
)

__all__ = [
    # Main log functions
    'analyze_navigation_graphs',
    'export_path_data_to_json',
    'generate_env_tensor_from_file',
    
    # State generation
    'State',
    'analyze_state_nodes',
    'filter_state_nodes',
    
    # Dijkstra's algorithm
    'create_graphs_from_nodes',
    'compute_shortest_paths'
] 