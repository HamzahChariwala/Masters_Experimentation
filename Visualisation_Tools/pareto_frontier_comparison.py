#!/usr/bin/env python3
"""
Generate Pareto frontier comparison plot between Standard and NoDeath agents.
Uses individual agent performance data to find Pareto optimal points.
X-axis: Goal reached proportion
Y-axis: 1 - Next cell lava proportion
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import magma
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


def find_agent_performance_files(agent_base_dir, agent_types, category):
    """Find all performance_all_states.json files for specified agent types."""
    performance_files = []
    
    if category == 'Standard':
        # Standard agents are in Standard, Standard2, Standard3 directories
        for agent_type in agent_types:
            agent_dir = os.path.join(agent_base_dir, agent_type)
            
            if not os.path.exists(agent_dir):
                print(f"Warning: Agent directory {agent_dir} not found")
                continue
                
            # Look for version directories
            for item in os.listdir(agent_dir):
                version_dir = os.path.join(agent_dir, item)
                if os.path.isdir(version_dir) and item.startswith(f"{agent_type}-v"):
                    performance_file = os.path.join(version_dir, "evaluation_summary", "performance_all_states.json")
                    
                    if os.path.exists(performance_file):
                        performance_files.append({
                            'path': performance_file,
                            'agent_id': item,
                            'agent_type': agent_type,
                            'category': category
                        })
    
    elif category == 'NoDeath':
        # NoDeath agents are in NoDeath/penalty_value directories
        nodeath_dir = os.path.join(agent_base_dir, 'NoDeath')
        if not os.path.exists(nodeath_dir):
            print(f"Warning: NoDeath directory {nodeath_dir} not found")
            return performance_files
        
        for penalty_folder in os.listdir(nodeath_dir):
            penalty_dir = os.path.join(nodeath_dir, penalty_folder)
            if os.path.isdir(penalty_dir) and penalty_folder.endswith('_penalty'):
                # Look for version directories within each penalty folder
                for item in os.listdir(penalty_dir):
                    version_dir = os.path.join(penalty_dir, item)
                    if os.path.isdir(version_dir) and item.startswith(f"{penalty_folder}-v"):
                        performance_file = os.path.join(version_dir, "evaluation_summary", "performance_all_states.json")
                        
                        if os.path.exists(performance_file):
                            performance_files.append({
                                'path': performance_file,
                                'agent_id': item,
                                'agent_type': f'NoDeath/{penalty_folder}',
                                'category': category
                            })
    
    return performance_files


def extract_metrics_from_file(performance_file_info):
    """Extract goal_reached_proportion and next_cell_lava_proportion from a performance file."""
    try:
        with open(performance_file_info['path'], 'r') as f:
            data = json.load(f)
        
        if 'overall_summary' not in data:
            return None
            
        overall_summary = data['overall_summary']
        goal_reached = overall_summary.get('goal_reached_proportion')
        next_cell_lava = overall_summary.get('next_cell_lava_proportion')
        
        if goal_reached is None or next_cell_lava is None:
            return None
        
        return {
            'category': performance_file_info['category'],
            'agent_type': performance_file_info['agent_type'],
            'agent_id': performance_file_info['agent_id'],
            'goal_reached_proportion': goal_reached,
            'next_cell_lava_proportion': next_cell_lava,
            'y_value': 1 - next_cell_lava  # Transform for y-axis
        }
        
    except Exception as e:
        print(f"Error processing {performance_file_info['path']}: {e}")
        return None


def find_pareto_frontier(points):
    """
    Find Pareto optimal points where we want to maximize both x and y.
    Returns indices of Pareto optimal points.
    """
    points = np.array(points)
    n_points = len(points)
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_pareto[i]:
            # For maximization: point j dominates point i if j >= i in all dimensions and j > i in at least one
            # So we check if any other point dominates point i
            dominates_i = ((points >= points[i]).all(axis=1) & 
                          (points > points[i]).any(axis=1))
            
            # If any point dominates point i, then point i is not Pareto optimal
            if dominates_i.any():
                is_pareto[i] = False
    
    return np.where(is_pareto)[0]


def create_pareto_frontier_comparison():
    """Create Pareto frontier comparison plot between Standard and NoDeath agents."""
    agent_base_dir = os.path.join(project_root, "Agent_Storage", "LavaTests")
    
    # Define agent types
    standard_types = ['Standard', 'Standard2', 'Standard3']
    
    # Find all performance files
    print("Searching for Standard agent performance files...")
    standard_files = find_agent_performance_files(agent_base_dir, standard_types, 'Standard')
    
    print("Searching for NoDeath agent performance files...")
    nodeath_files = find_agent_performance_files(agent_base_dir, [], 'NoDeath')
    
    print(f"Found {len(standard_files)} Standard agent files")
    print(f"Found {len(nodeath_files)} NoDeath agent files")
    
    # Extract metrics from all files
    all_metrics = []
    
    for file_info in standard_files + nodeath_files:
        metrics = extract_metrics_from_file(file_info)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid metrics data found. Cannot generate plot.")
        return
    
    print(f"Successfully extracted metrics from {len(all_metrics)} agent trials")
    
    # Separate by category
    standard_metrics = [m for m in all_metrics if m['category'] == 'Standard']
    nodeath_metrics = [m for m in all_metrics if m['category'] == 'NoDeath']
    
    print(f"Standard agents: {len(standard_metrics)}")
    print(f"NoDeath agents: {len(nodeath_metrics)}")
    
    # Extract points for Pareto analysis
    standard_points = [(m['goal_reached_proportion'], m['y_value']) for m in standard_metrics]
    nodeath_points = [(m['goal_reached_proportion'], m['y_value']) for m in nodeath_metrics]
    
    # Find Pareto frontiers
    standard_pareto_indices = find_pareto_frontier(standard_points)
    nodeath_pareto_indices = find_pareto_frontier(nodeath_points)
    
    # Colors from magma colormap
    standard_color = magma(0.2)  # Dark purple
    nodeath_color = magma(0.6)   # More distinct orange/red color
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot all points
    standard_x = [p[0] for p in standard_points]
    standard_y = [p[1] for p in standard_points]
    nodeath_x = [p[0] for p in nodeath_points]
    nodeath_y = [p[1] for p in nodeath_points]
    
    # Scatter plots for all points
    ax.scatter(standard_x, standard_y, 
               c=[standard_color], 
               label=f'Standard Agents (n={len(standard_metrics)})',
               alpha=0.6, s=65, edgecolors='white', linewidth=0.5)
    
    ax.scatter(nodeath_x, nodeath_y, 
               c=[nodeath_color], 
               label=f'NoDeath Agents (n={len(nodeath_metrics)})',
               alpha=0.6, s=65, edgecolors='white', linewidth=0.5)
    
    # Highlight Pareto optimal points
    pareto_standard_x = [standard_points[i][0] for i in standard_pareto_indices]
    pareto_standard_y = [standard_points[i][1] for i in standard_pareto_indices]
    pareto_nodeath_x = [nodeath_points[i][0] for i in nodeath_pareto_indices]
    pareto_nodeath_y = [nodeath_points[i][1] for i in nodeath_pareto_indices]
    
    # Plot Pareto optimal points with different markers
    ax.scatter(pareto_standard_x, pareto_standard_y, 
               c=[standard_color], 
               label=f'Standard Pareto Optimal (n={len(standard_pareto_indices)})',
               alpha=1.0, s=140, marker='s', edgecolors='none', linewidth=0)
    
    ax.scatter(pareto_nodeath_x, pareto_nodeath_y, 
               c=[nodeath_color], 
               label=f'NoDeath Pareto Optimal (n={len(nodeath_pareto_indices)})',
               alpha=1.0, s=140, marker='^', edgecolors='none', linewidth=0)
    
    # Sort Pareto points by x-coordinate for line plotting
    if len(pareto_standard_x) > 1:
        standard_pareto_sorted = sorted(zip(pareto_standard_x, pareto_standard_y))
        ax.plot([p[0] for p in standard_pareto_sorted], [p[1] for p in standard_pareto_sorted], 
                color=standard_color, linewidth=4, alpha=0.7, linestyle='--')
    
    if len(pareto_nodeath_x) > 1:
        nodeath_pareto_sorted = sorted(zip(pareto_nodeath_x, pareto_nodeath_y))
        ax.plot([p[0] for p in nodeath_pareto_sorted], [p[1] for p in nodeath_pareto_sorted], 
                color=nodeath_color, linewidth=4, alpha=0.7, linestyle='--')
    
    # Customize the plot
    ax.set_xlabel('Goal Reached Proportion', fontsize=14, fontweight='bold')
    ax.set_ylabel('1 - Next Cell Lava Proportion', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier Comparison: Standard vs NoDeath Agents', fontsize=16, fontweight='bold')
    
    # Set axis limits
    all_x = standard_x + nodeath_x
    all_y = standard_y + nodeath_y
    x_margin = (max(all_x) - min(all_x)) * 0.05
    y_margin = (max(all_y) - min(all_y)) * 0.05
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), "standard_vs_nodeath")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    save_path = os.path.join(save_dir, "pareto_frontier_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPareto frontier comparison plot saved: {save_path}")
    print(f"Standard Pareto optimal points: {len(standard_pareto_indices)}")
    print(f"NoDeath Pareto optimal points: {len(nodeath_pareto_indices)}")
    
    # Print some statistics
    print(f"\nStandard agents performance range:")
    if standard_x:
        print(f"  Goal reached: {min(standard_x):.3f} - {max(standard_x):.3f}")
        print(f"  1-Lava proportion: {min(standard_y):.3f} - {max(standard_y):.3f}")
    else:
        print("  No Standard agents found")
    
    print(f"NoDeath agents performance range:")
    if nodeath_x:
        print(f"  Goal reached: {min(nodeath_x):.3f} - {max(nodeath_x):.3f}")
        print(f"  1-Lava proportion: {min(nodeath_y):.3f} - {max(nodeath_y):.3f}")
    else:
        print("  No NoDeath agents found")
    
    return save_path


if __name__ == "__main__":
    print("Generating Pareto frontier comparison plot...")
    print("Comparing Standard vs NoDeath agents")
    print("X-axis: Goal reached proportion")
    print("Y-axis: 1 - Next cell lava proportion")
    print("Using individual agent performance data\n")
    
    save_path = create_pareto_frontier_comparison()
    
    if save_path:
        print("\nPareto frontier comparison completed!")
    else:
        print("Failed to generate Pareto frontier comparison.") 