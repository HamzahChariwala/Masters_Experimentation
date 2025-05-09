#!/usr/bin/env python3
"""
Distribution Visualizer

This script visualizes the different distribution types defined in config.yaml
for spawn positions in MiniGrid environments, using the magma colormap.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import matplotlib as mpl

# Import our distribution map class
from EnvironmentEdits.BespokeEdits.SpawnDistribution import DistributionMap

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Visualize spawn distributions from config.yaml")
    parser.add_argument("--grid-size", type=int, default=11,
                        help="Size of the grid (default: 11x11)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("--output-dir", type=str, default="./distribution_vis",
                        help="Directory to save visualizations (default: ./distribution_vis)")
    parser.add_argument("--show", action="store_true",
                        help="Show visualizations (default: False)")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def visualize_distribution(dist_map, title, output_path=None, show=False):
    """Visualize a distribution map using the magma colormap"""
    plt.figure(figsize=(10, 8))
    
    # Use magma colormap
    cmap = plt.cm.magma
    
    # Plot the distribution
    plt.imshow(dist_map.probabilities, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Probability')
    
    # Add grid lines
    plt.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set ticks at each grid cell
    plt.xticks(np.arange(-.5, dist_map.width, 1), [])
    plt.yticks(np.arange(-.5, dist_map.height, 1), [])
    
    # Add labels for coordinates
    for i in range(dist_map.width):
        plt.text(i, -0.8, str(i), ha='center', va='center', color='black')
    for j in range(dist_map.height):
        plt.text(-0.8, j, str(j), ha='center', va='center', color='black')
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    
    plt.close()

def apply_distribution_from_config(dist_map, dist_type, params):
    """Apply distribution from config to the distribution map"""
    if dist_type == "uniform":
        dist_map.uniform_distribution()
    
    elif dist_type == "gaussian_2d":
        center = params.get("center", [1.0, 1.0])
        std = params.get("std", [0.2, 0.2])
        directional = params.get("directional", False)
        angle = params.get("angle", 0)
        dist_map.gaussian_2d(center, std, directional, angle)
    
    elif dist_type == "corners":
        corner_size = params.get("corner_size", 2)
        dist_map.corners_distribution(corner_size)
    
    elif dist_type == "border":
        border_width = params.get("border_width", 1)
        dist_map.border_distribution(border_width)
    
    elif dist_type == "composite":
        distributions = params.get("distributions", [])
        dist_map.composite_distribution(distributions)
    
    return dist_map

def visualize_stages_from_config(config, grid_size, output_dir, show=False):
    """Visualize all stages defined in the config"""
    # Check if spawn configuration exists
    if 'spawn' not in config or 'stage_training' not in config['spawn']:
        print("No stage training configuration found in config file")
        return
    
    # Get stage training configuration
    stage_config = config['spawn']['stage_training']
    
    if not stage_config.get('enabled', False) or 'distributions' not in stage_config:
        print("Stage training is not enabled or no distributions defined")
        return
    
    # Get distributions
    distributions = stage_config['distributions']
    num_stages = len(distributions)
    
    print(f"Visualizing {num_stages} stages from config")
    
    # Create visualizations for each stage
    for i, dist_config in enumerate(distributions):
        # Get distribution type and parameters
        dist_type = dist_config.get('type')
        params = dist_config.get('params', {})
        description = dist_config.get('description', f"Stage {i+1}")
        
        # Skip if no distribution type
        if not dist_type:
            continue
        
        # Create distribution map
        dist_map = DistributionMap(grid_size, grid_size)
        
        # Apply distribution
        apply_distribution_from_config(dist_map, dist_type, params)
        
        # Output path
        output_path = os.path.join(output_dir, f"stage_{i+1}_{dist_type}.png")
        
        # Title
        title = f"Stage {i+1}: {description}"
        
        # Visualize
        visualize_distribution(dist_map, title, output_path, show)
    
    print(f"Generated {num_stages} stage visualizations in {output_dir}")

def visualize_transitions(config, grid_size, output_dir, show=False):
    """Visualize transitions between stages"""
    # Check if spawn configuration exists
    if 'spawn' not in config or 'stage_training' not in config['spawn']:
        print("No stage training configuration found in config file")
        return
    
    # Get stage training configuration
    stage_config = config['spawn']['stage_training']
    
    if not stage_config.get('enabled', False) or 'distributions' not in stage_config:
        print("Stage training is not enabled or no distributions defined")
        return
    
    # Get distributions
    distributions = stage_config['distributions']
    num_stages = len(distributions)
    
    # Only process if there are at least 2 stages
    if num_stages < 2:
        return
    
    # Create transitions between each pair of adjacent stages
    for i in range(num_stages - 1):
        # Get distribution types and parameters for current and next stage
        current_type = distributions[i].get('type')
        current_params = distributions[i].get('params', {})
        next_type = distributions[i+1].get('type')
        next_params = distributions[i+1].get('params', {})
        
        # Skip if no distribution type
        if not current_type or not next_type:
            continue
        
        # Create distribution maps
        current_map = DistributionMap(grid_size, grid_size)
        next_map = DistributionMap(grid_size, grid_size)
        
        # Apply distributions
        apply_distribution_from_config(current_map, current_type, current_params)
        apply_distribution_from_config(next_map, next_type, next_params)
        
        # Create transition visualizations (25%, 50%, 75%)
        transition_points = [0.25, 0.5, 0.75]
        
        for progress in transition_points:
            # Create transition map
            transition_map = DistributionMap(grid_size, grid_size)
            transition_map.from_existing_distribution(current_map.probabilities)
            transition_map.temporal_interpolation(next_map, progress)
            
            # Output path
            output_path = os.path.join(output_dir, f"transition_{i}_to_{i+1}_{int(progress*100)}pct.png")
            
            # Title
            title = f"Transition from Stage {i+1} to {i+2} ({int(progress*100)}%)"
            
            # Visualize
            visualize_distribution(transition_map, title, output_path, show)
    
    print(f"Generated transition visualizations in {output_dir}")

def main():
    """Main function to visualize distributions from config.yaml"""
    args = parse_args()
    
    # Load config file
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set the default colormap to magma
    plt.rcParams['image.cmap'] = 'magma'
    
    # Visualize all stages from config
    visualize_stages_from_config(config, args.grid_size, args.output_dir, args.show)
    
    # Visualize transitions between stages
    visualize_transitions(config, args.grid_size, args.output_dir, args.show)

if __name__ == "__main__":
    main() 