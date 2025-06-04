#!/usr/bin/env python3
"""
Generate scatterplots showing Standard, Standard2, and Standard3 agent performance.
X-axis: goal reached proportion
Y-axis: 1 - next cell lava proportion
Labels: Training timesteps from config files
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import magma
import yaml
import sys
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


def extract_timesteps_from_config(config_path):
    """Extract total_timesteps from agent config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Navigate to the timesteps field
        timesteps = config.get('experiment', {}).get('output', {}).get('total_timesteps')
        return timesteps
    except Exception as e:
        print(f"Error reading config {config_path}: {e}")
        return None


def find_agent_performance_files(agent_base_dir, agent_types):
    """Find all performance_all_states.json files and extract timesteps."""
    performance_files = {}
    
    for agent_type in agent_types:
        performance_files[agent_type] = []
        agent_dir = os.path.join(agent_base_dir, agent_type)
        
        if not os.path.exists(agent_dir):
            print(f"Warning: Agent directory {agent_dir} not found")
            continue
            
        # Look for version directories
        for item in os.listdir(agent_dir):
            version_dir = os.path.join(agent_dir, item)
            if os.path.isdir(version_dir) and item.startswith(f"{agent_type}-v"):
                performance_file = os.path.join(version_dir, "evaluation_summary", "performance_all_states.json")
                config_file = os.path.join(version_dir, "config.yaml")
                
                if os.path.exists(performance_file) and os.path.exists(config_file):
                    timesteps = extract_timesteps_from_config(config_file)
                    performance_files[agent_type].append({
                        'path': performance_file,
                        'agent_id': item,
                        'agent_type': agent_type,
                        'timesteps': timesteps
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
            'agent_type': performance_file_info['agent_type'],
            'agent_id': performance_file_info['agent_id'],
            'timesteps': performance_file_info['timesteps'],
            'goal_reached_proportion': goal_reached,
            'next_cell_lava_proportion': next_cell_lava,
            'y_value': 1 - next_cell_lava  # Transform for y-axis
        }
        
    except Exception as e:
        print(f"Error processing {performance_file_info['path']}: {e}")
        return None


def generate_standard_scatter_plots():
    """Generate three scatterplots for Standard, Standard2, and Standard3 agents."""
    agent_base_dir = os.path.join(project_root, "Agent_Storage", "LavaTests")
    agent_types = ['Standard', 'Standard2', 'Standard3']
    
    # Find all performance files
    print("Searching for agent performance files...")
    performance_files = find_agent_performance_files(agent_base_dir, agent_types)
    
    # Check what we found
    for agent_type in agent_types:
        count = len(performance_files.get(agent_type, []))
        print(f"Found {count} performance files for {agent_type}")
    
    # Extract metrics from all files
    all_metrics = []
    for agent_type in agent_types:
        for file_info in performance_files.get(agent_type, []):
            metrics = extract_metrics_from_file(file_info)
            if metrics:
                all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid metrics data found. Cannot generate plot.")
        return None
    
    print(f"Successfully extracted metrics from {len(all_metrics)} agent trials")
    
    # Group by timesteps to create proper labels
    timestep_groups = {}
    for metrics in all_metrics:
        timesteps = metrics['timesteps']
        if timesteps not in timestep_groups:
            timestep_groups[timesteps] = {'x': [], 'y': [], 'agent_type': metrics['agent_type']}
        
        timestep_groups[timesteps]['x'].append(metrics['goal_reached_proportion'])
        timestep_groups[timesteps]['y'].append(metrics['y_value'])
    
    # Sort by timesteps for consistent coloring
    sorted_timesteps = sorted(timestep_groups.keys())
    
    # Define colors using magma colormap
    colors = [magma(0.2), magma(0.5), magma(0.8)]
    
    # Calculate axis limits
    all_x = [m['goal_reached_proportion'] for m in all_metrics]
    all_y = [m['y_value'] for m in all_metrics]
    x_margin = (max(all_x) - min(all_x)) * 0.1 if max(all_x) != min(all_x) else 0.05
    y_margin = (max(all_y) - min(all_y)) * 0.1 if max(all_y) != min(all_y) else 0.05
    xlim = (max(0, min(all_x) - x_margin), min(1, max(all_x) + x_margin))
    ylim = (max(0, min(all_y) - y_margin), min(1, max(all_y) + y_margin))
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), "standard")
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Larger dots
    print("\nGenerating Plot 1: Large dots...")
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    for i, timesteps in enumerate(sorted_timesteps):
        data = timestep_groups[timesteps]
        label = f"{timesteps:,} timesteps"  # Format with commas
        
        ax1.scatter(data['x'], data['y'], 
                   c=[colors[i]], 
                   label=label,
                   alpha=0.7,
                   s=150,  # Large dots
                   edgecolors='black',
                   linewidth=1.0)
        
        print(f"{label}: {len(data['x'])} data points")
        print(f"  Goal reached range: {min(data['x']):.3f} - {max(data['x']):.3f}")
        print(f"  1-Lava proportion range: {min(data['y']):.3f} - {max(data['y']):.3f}")
    
    # Customize plot 1
    ax1.set_xlabel('Goal Reached Proportion', fontsize=12, fontweight='bold')
    ax1.set_ylabel('1 - Next Cell Lava Proportion', fontsize=12, fontweight='bold')
    ax1.set_title('Agent Performance: Goal Reached vs Lava Avoidance (Large Dots)', fontsize=14, fontweight='bold')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='best')
    
    # Save plot 1
    save_path1 = os.path.join(save_dir, "standard_agents_scatter_large_dots.png")
    plt.tight_layout()
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved large dots plot: {save_path1}")
    
    # Plot 2: Regular sized dots
    print("\nGenerating Plot 2: Regular dots...")
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    
    for i, timesteps in enumerate(sorted_timesteps):
        data = timestep_groups[timesteps]
        label = f"{timesteps:,} timesteps"
        
        # Plot the points
        ax2.scatter(data['x'], data['y'], 
                   c=[colors[i]], 
                   label=label,
                   alpha=0.7,
                   s=100,  # Regular sized dots
                   edgecolors='black',
                   linewidth=0.5)
    
    # Customize plot 2
    ax2.set_xlabel('Goal Reached Proportion', fontsize=12, fontweight='bold')
    ax2.set_ylabel('1 - Next Cell Lava Proportion', fontsize=12, fontweight='bold')
    ax2.set_title('Agent Performance: Goal Reached vs Lava Avoidance (Regular Dots)', fontsize=14, fontweight='bold')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='best')
    
    # Save plot 2
    save_path2 = os.path.join(save_dir, "standard_agents_scatter_regular_dots.png")
    plt.tight_layout()
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved regular dots plot: {save_path2}")
    
    # Plot 3: Ashby-style with shaded regions
    print("\nGenerating Plot 3: Ashby-style with shaded regions...")
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
    
    for i, timesteps in enumerate(sorted_timesteps):
        data = timestep_groups[timesteps]
        label = f"{timesteps:,} timesteps"
        
        # Create points array for convex hull
        points = np.column_stack((data['x'], data['y']))
        
        # Only create convex hull if we have at least 3 points
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                # Get hull vertices in order
                hull_points = points[hull.vertices]
                
                # Create polygon and add to plot
                polygon = Polygon(hull_points, 
                                facecolor=colors[i], 
                                alpha=0.3, 
                                edgecolor=colors[i], 
                                linewidth=2,
                                label=f"{label} region")
                ax3.add_patch(polygon)
                
            except Exception as e:
                print(f"Warning: Could not create convex hull for {label}: {e}")
        
        # Plot the points on top of the shaded regions
        ax3.scatter(data['x'], data['y'], 
                   c=[colors[i]], 
                   alpha=0.8,
                   s=80,
                   edgecolors='white',
                   linewidth=1.5,
                   zorder=10)  # Ensure points are on top
    
    # Customize plot 3
    ax3.set_xlabel('Goal Reached Proportion', fontsize=12, fontweight='bold')
    ax3.set_ylabel('1 - Next Cell Lava Proportion', fontsize=12, fontweight='bold')
    ax3.set_title('Agent Performance: Goal Reached vs Lava Avoidance (Ashby-style Regions)', fontsize=14, fontweight='bold')
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.grid(True, alpha=0.3)
    
    # Create custom legend combining regions and points
    legend_elements = []
    for i, timesteps in enumerate(sorted_timesteps):
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor=colors[i], alpha=0.3, edgecolor=colors[i], 
                                   label=f"{timesteps:,} timesteps"))
    ax3.legend(handles=legend_elements, fontsize=11, loc='best')
    
    # Save plot 3
    save_path3 = os.path.join(save_dir, "standard_agents_scatter_ashby_regions.png")
    plt.tight_layout()
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved Ashby-style plot: {save_path3}")
    
    print(f"\nAll three plots saved to standard/ directory")
    print(f"Total data points plotted: {len(all_metrics)}")
    
    return [save_path1, save_path2, save_path3]


if __name__ == "__main__":
    print("Generating Standard agents scatterplots...")
    print("X-axis: Goal reached proportion")
    print("Y-axis: 1 - Next cell lava proportion")
    print("Labels: Training timesteps from config files")
    print("Colors: Magma colormap\n")
    
    save_paths = generate_standard_scatter_plots()
    
    if save_paths:
        print("\nScatterplot generation completed!")
        print("Generated plots:")
        for i, path in enumerate(save_paths, 1):
            print(f"  {i}. {path}")
    else:
        print("Failed to generate scatterplots.") 