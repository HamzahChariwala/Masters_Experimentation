#!/usr/bin/env python3
"""
Generate Pareto frontier comparison between Linear, Exponential, and Sigmoid reward function agents.
X-axis: Goal reached proportion
Y-axis: 1 - Next cell lava proportion
Each reward function type gets its own color and Pareto frontier.
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


def find_reward_function_performance_files(agent_base_dir):
    """Find all reward function performance_all_states.json files organized by type."""
    performance_files = {}
    
    episode_end_dir = os.path.join(agent_base_dir, 'EpisodeEnd')
    if not os.path.exists(episode_end_dir):
        print(f"Warning: EpisodeEnd directory {episode_end_dir} not found")
        return performance_files
    
    reward_types = ['Linear', 'Exponential', 'Sigmoid']
    
    for reward_type in reward_types:
        performance_files[reward_type] = []
        reward_dir = os.path.join(episode_end_dir, reward_type)
        
        if not os.path.exists(reward_dir):
            print(f"Warning: {reward_type} directory {reward_dir} not found")
            continue
        
        # Look through penalty folders
        for penalty_folder in os.listdir(reward_dir):
            penalty_dir = os.path.join(reward_dir, penalty_folder)
            if os.path.isdir(penalty_dir) and penalty_folder.endswith('_penalty'):
                
                # Look for version directories within each penalty folder
                for item in os.listdir(penalty_dir):
                    version_dir = os.path.join(penalty_dir, item)
                    if os.path.isdir(version_dir) and item.startswith(f"{penalty_folder}-v"):
                        performance_file = os.path.join(version_dir, "evaluation_summary", "performance_all_states.json")
                        
                        if os.path.exists(performance_file):
                            performance_files[reward_type].append({
                                'path': performance_file,
                                'agent_id': item,
                                'penalty': penalty_folder.replace('_penalty', ''),
                                'reward_type': reward_type
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
            'reward_type': performance_file_info['reward_type'],
            'agent_id': performance_file_info['agent_id'],
            'penalty': performance_file_info['penalty'],
            'goal_reached_proportion': goal_reached,
            'next_cell_lava_proportion': next_cell_lava,
            'y_value': 1 - next_cell_lava  # Transform for y-axis
        }
        
    except Exception as e:
        print(f"Error processing {performance_file_info['path']}: {e}")
        return None


def compute_pareto_frontier(points):
    """
    Compute the Pareto frontier for a set of 2D points.
    Assumes we want to maximize both x and y coordinates.
    """
    if len(points) == 0:
        return np.array([]), np.array([])
    
    points = np.array(points)
    n_points = points.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Point j dominates point i if j is better in both dimensions
                if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and 
                    (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    is_pareto[i] = False
                    break
    
    pareto_points = points[is_pareto]
    
    if len(pareto_points) == 0:
        return np.array([]), np.array([])
    
    # Sort by x-coordinate for plotting
    sorted_indices = np.argsort(pareto_points[:, 0])
    pareto_points = pareto_points[sorted_indices]
    
    return pareto_points[:, 0], pareto_points[:, 1]


def create_reward_function_pareto_comparison():
    """Create Pareto frontier comparison plot for reward function types."""
    agent_base_dir = os.path.join(project_root, "Agent_Storage", "LavaTests")
    
    # Find all reward function performance files
    print("Searching for reward function agent performance files...")
    reward_files = find_reward_function_performance_files(agent_base_dir)
    
    # Extract metrics from all files
    all_metrics = {}
    reward_counts = {}
    
    for reward_type, files in reward_files.items():
        all_metrics[reward_type] = []
        reward_counts[reward_type] = len(files)
        
        for file_info in files:
            metrics = extract_metrics_from_file(file_info)
            if metrics:
                all_metrics[reward_type].append(metrics)
    
    # Check if we have data
    total_agents = sum(len(metrics) for metrics in all_metrics.values())
    if total_agents == 0:
        print("No valid metrics data found. Cannot generate plot.")
        return
    
    print(f"Successfully extracted metrics from {total_agents} reward function agent trials")
    for reward_type, metrics in all_metrics.items():
        print(f"  {reward_type}: {len(metrics)} agents")
    
    # Colors matching the reward function plots
    reward_colors = {
        'Linear': magma(0.2),
        'Exponential': magma(0.5), 
        'Sigmoid': magma(0.8)
    }
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot each reward function type
    for reward_type, metrics in all_metrics.items():
        if not metrics:
            continue
            
        # Extract x and y coordinates
        x_coords = [m['goal_reached_proportion'] for m in metrics]
        y_coords = [m['y_value'] for m in metrics]
        points = list(zip(x_coords, y_coords))
        
        color = reward_colors[reward_type]
        
        # Plot all points
        ax.scatter(x_coords, y_coords, 
                   c=[color], alpha=0.6, s=65, 
                   label=f'{reward_type} (n={len(metrics)})',
                   edgecolors='white', linewidth=0.5)
        
        # Compute and plot Pareto frontier
        pareto_x, pareto_y = compute_pareto_frontier(points)
        
        if len(pareto_x) > 0:
            # Plot Pareto optimal points with higher visibility
            ax.scatter(pareto_x, pareto_y, 
                      c=[color], s=140, alpha=1.0,
                      edgecolors='none', marker='o', zorder=10)
            
            # Plot Pareto frontier line
            if len(pareto_x) > 1:
                ax.plot(pareto_x, pareto_y, color=color, linewidth=4, 
                       alpha=0.8, linestyle='-', zorder=9)
            
            print(f"{reward_type} Pareto frontier: {len(pareto_x)} optimal points")
            print(f"  Goal reached range: {min(pareto_x):.3f} - {max(pareto_x):.3f}")
            print(f"  1-Lava proportion range: {min(pareto_y):.3f} - {max(pareto_y):.3f}")
        else:
            print(f"{reward_type}: No Pareto optimal points found")
    
    # Customize the plot
    ax.set_xlabel('Goal Reached Proportion', fontsize=14, fontweight='bold')
    ax.set_ylabel('1 - Next Cell Lava Proportion', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier Comparison: Reward Function Types', fontsize=16, fontweight='bold')
    
    # Set axis limits
    all_x = []
    all_y = []
    for metrics in all_metrics.values():
        if metrics:
            all_x.extend([m['goal_reached_proportion'] for m in metrics])
            all_y.extend([m['y_value'] for m in metrics])
    
    if all_x and all_y:
        x_margin = (max(all_x) - min(all_x)) * 0.05
        y_margin = (max(all_y) - min(all_y)) * 0.05
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), "custom_rewards")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    save_path = os.path.join(save_dir, "reward_function_pareto_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nReward function Pareto comparison plot saved: {save_path}")
    print(f"Total reward types: {len([t for t in all_metrics.keys() if all_metrics[t]])}")
    print(f"Total data points: {total_agents}")
    
    return save_path


if __name__ == "__main__":
    print("Generating reward function Pareto frontier comparison...")
    print("X-axis: Goal reached proportion")
    print("Y-axis: 1 - Next cell lava proportion")
    print("Colors: Linear (magma 0.2), Exponential (magma 0.5), Sigmoid (magma 0.8)\n")
    
    save_path = create_reward_function_pareto_comparison()
    
    if save_path:
        print("\nReward function Pareto comparison completed!")
    else:
        print("Failed to generate reward function Pareto comparison plot.") 