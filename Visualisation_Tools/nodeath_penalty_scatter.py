#!/usr/bin/env python3
"""
Generate scatter plot showing individual NoDeath agent trials colored by penalty value.
X-axis: Goal reached proportion
Y-axis: 1 - Next cell lava proportion
Each penalty gets its own color from the magma colormap with convex hull regions.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import magma
from scipy.spatial import ConvexHull
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


def find_nodeath_performance_files(agent_base_dir):
    """Find all NoDeath performance_all_states.json files organized by penalty."""
    performance_files = {}
    
    nodeath_dir = os.path.join(agent_base_dir, 'NoDeath')
    if not os.path.exists(nodeath_dir):
        print(f"Warning: NoDeath directory {nodeath_dir} not found")
        return performance_files
    
    for penalty_folder in os.listdir(nodeath_dir):
        penalty_dir = os.path.join(nodeath_dir, penalty_folder)
        if os.path.isdir(penalty_dir) and penalty_folder.endswith('_penalty'):
            penalty_value = penalty_folder.replace('_penalty', '')
            performance_files[penalty_value] = []
            
            # Look for version directories within each penalty folder
            for item in os.listdir(penalty_dir):
                version_dir = os.path.join(penalty_dir, item)
                if os.path.isdir(version_dir) and item.startswith(f"{penalty_folder}-v"):
                    performance_file = os.path.join(version_dir, "evaluation_summary", "performance_all_states.json")
                    
                    if os.path.exists(performance_file):
                        performance_files[penalty_value].append({
                            'path': performance_file,
                            'agent_id': item,
                            'penalty': penalty_value
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
            'penalty': performance_file_info['penalty'],
            'agent_id': performance_file_info['agent_id'],
            'goal_reached_proportion': goal_reached,
            'next_cell_lava_proportion': next_cell_lava,
            'y_value': 1 - next_cell_lava  # Transform for y-axis
        }
        
    except Exception as e:
        print(f"Error processing {performance_file_info['path']}: {e}")
        return None


def create_nodeath_penalty_scatter():
    """Create scatter plot of NoDeath agents colored by penalty value with convex hull regions."""
    agent_base_dir = os.path.join(project_root, "Agent_Storage", "LavaTests")
    
    # Find all NoDeath performance files organized by penalty
    print("Searching for NoDeath agent performance files...")
    penalty_files = find_nodeath_performance_files(agent_base_dir)
    
    # Extract metrics from all files
    all_metrics = []
    penalty_counts = {}
    
    for penalty, files in penalty_files.items():
        penalty_counts[penalty] = len(files)
        for file_info in files:
            metrics = extract_metrics_from_file(file_info)
            if metrics:
                all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid metrics data found. Cannot generate plot.")
        return
    
    print(f"Successfully extracted metrics from {len(all_metrics)} NoDeath agent trials")
    for penalty, count in sorted(penalty_counts.items(), key=lambda x: float(x[0])):
        print(f"  Penalty {penalty}: {count} agents")
    
    # Group metrics by penalty
    penalty_data = {}
    for metrics in all_metrics:
        penalty = metrics['penalty']
        if penalty not in penalty_data:
            penalty_data[penalty] = {'x': [], 'y': []}
        penalty_data[penalty]['x'].append(metrics['goal_reached_proportion'])
        penalty_data[penalty]['y'].append(metrics['y_value'])
    
    # Sort penalties for consistent coloring
    sorted_penalties = sorted(penalty_data.keys(), key=lambda x: float(x))
    
    # Create colors from magma colormap - spread across the range
    n_penalties = len(sorted_penalties)
    colors = [magma(i / (n_penalties - 1)) for i in range(n_penalties)]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # First, plot convex hull regions for each penalty group
    for i, penalty in enumerate(sorted_penalties):
        data = penalty_data[penalty]
        
        if len(data['x']) >= 3:  # Need at least 3 points for a hull
            points = np.column_stack((data['x'], data['y']))
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                ax.fill(hull_points[:, 0], hull_points[:, 1], 
                       color=colors[i], alpha=0.3, 
                       label=f'Penalty {penalty} region')
            except Exception as e:
                print(f"Could not create convex hull for penalty {penalty}: {e}")
    
    # Then plot the scatter points on top
    for i, penalty in enumerate(sorted_penalties):
        data = penalty_data[penalty]
        ax.scatter(data['x'], data['y'], 
                   c=[colors[i]], 
                   label=f'Penalty {penalty} (n={len(data["x"])})',
                   alpha=0.8, s=65, edgecolors='white', linewidth=0.5,
                   zorder=10)  # Higher zorder to plot on top of hull regions
        
        print(f"Penalty {penalty}: {len(data['x'])} data points")
        print(f"  Goal reached range: {min(data['x']):.3f} - {max(data['x']):.3f}")
        print(f"  1-Lava proportion range: {min(data['y']):.3f} - {max(data['y']):.3f}")
    
    # Customize the plot
    ax.set_xlabel('Goal Reached Proportion', fontsize=14, fontweight='bold')
    ax.set_ylabel('1 - Next Cell Lava Proportion', fontsize=14, fontweight='bold')
    ax.set_title('NoDeath Agent Performance by Penalty Value (with Convex Hull Regions)', fontsize=16, fontweight='bold')
    
    # Set axis limits
    all_x = [m['goal_reached_proportion'] for m in all_metrics]
    all_y = [m['y_value'] for m in all_metrics]
    x_margin = (max(all_x) - min(all_x)) * 0.05
    y_margin = (max(all_y) - min(all_y)) * 0.05
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    
    ax.grid(True, alpha=0.3)
    
    # Create legend with both points and regions
    handles, labels = ax.get_legend_handles_labels()
    # Filter to only show the scatter point labels (not the region labels)
    point_handles = [h for h, l in zip(handles, labels) if 'region' not in l]
    point_labels = [l for l in labels if 'region' not in l]
    ax.legend(handles=point_handles, labels=point_labels, fontsize=10, loc='best', ncol=2)
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), "standard_vs_nodeath")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    save_path = os.path.join(save_dir, "nodeath_penalty_scatter_with_hulls.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nNoDeath penalty scatter plot with convex hulls saved: {save_path}")
    print(f"Total penalties plotted: {len(sorted_penalties)}")
    print(f"Total data points: {len(all_metrics)}")
    
    # Print performance ranges by penalty
    print(f"\nPerformance ranges by penalty:")
    for penalty in sorted_penalties:
        data = penalty_data[penalty]
        print(f"  Penalty {penalty}:")
        print(f"    Goal reached: {min(data['x']):.3f} - {max(data['x']):.3f}")
        print(f"    1-Lava proportion: {min(data['y']):.3f} - {max(data['y']):.3f}")
    
    return save_path


if __name__ == "__main__":
    print("Generating NoDeath penalty scatter plot with convex hull regions...")
    print("X-axis: Goal reached proportion")
    print("Y-axis: 1 - Next cell lava proportion")
    print("Colors: Individual penalties using magma colormap")
    print("Regions: Convex hull areas for each penalty group\n")
    
    save_path = create_nodeath_penalty_scatter()
    
    if save_path:
        print("\nNoDeath penalty scatter plot with convex hulls completed!")
    else:
        print("Failed to generate NoDeath penalty scatter plot.") 