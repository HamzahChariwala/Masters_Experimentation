#!/usr/bin/env python

"""
Utility script to visualize different spawn distribution configurations.
This script creates and visualizes all supported distribution types for spawn positions.

Run this script to generate visualization images of the different distribution types.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .spawn_distributions import DistributionMap

def visualize_all_distributions(output_dir="distribution_visualizations", grid_size=(8, 8), goal_pos=(7, 7)):
    """
    Visualize all supported distribution types with different parameters.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save visualizations
    grid_size : tuple
        Size of the grid (height, width)
    goal_pos : tuple
        Position of the goal (y, x)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure to hold all distribution visualizations
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Spawn Distribution Types", fontsize=16)
    
    # 1. Uniform distribution
    print("Generating uniform distribution...")
    uniform_dist = DistributionMap(grid_size)
    uniform_dist.uniform()
    
    # 2. Poisson distribution (favor near)
    print("Generating Poisson distribution (near goal)...")
    poisson_near = DistributionMap(grid_size)
    poisson_near.poisson_from_point(goal_pos, lambda_param=1.0, favor_near=True)
    
    # 3. Poisson distribution (favor far)
    print("Generating Poisson distribution (far from goal)...")
    poisson_far = DistributionMap(grid_size)
    poisson_far.poisson_from_point(goal_pos, lambda_param=1.0, favor_near=False)
    
    # 4. Gaussian distribution (favor near)
    print("Generating Gaussian distribution (near goal)...")
    gaussian_near = DistributionMap(grid_size)
    gaussian_near.gaussian_from_point(goal_pos, sigma=2.0, favor_near=True)
    
    # 5. Gaussian distribution (favor far)
    print("Generating Gaussian distribution (far from goal)...")
    gaussian_far = DistributionMap(grid_size)
    gaussian_far.gaussian_from_point(goal_pos, sigma=2.0, favor_near=False)
    
    # 6. Distance-based distribution (favor near)
    print("Generating Distance-based distribution (near goal)...")
    distance_near = DistributionMap(grid_size)
    distance_near.distance_from_point(goal_pos, power=1, favor_near=True)
    
    # 7. Distance-based distribution (favor far)
    print("Generating Distance-based distribution (far from goal)...")
    distance_far = DistributionMap(grid_size)
    distance_far.distance_from_point(goal_pos, power=1, favor_near=False)
    
    # 8. Multi-point distribution
    print("Generating Multi-point distribution...")
    points = [(1, 1), goal_pos, (grid_size[0]//2, grid_size[1]//2)]
    weights = [0.5, 0.3, 0.2]
    multi_point = DistributionMap(grid_size)
    multi_point.multi_point_gaussian(points, weights, sigma=1.5)
    
    # 9. Masked distribution (exclude center region)
    print("Generating Masked distribution...")
    masked_dist = DistributionMap(grid_size)
    masked_dist.uniform()
    
    # Create a mask to exclude the center region
    mask = np.ones(grid_size, dtype=bool)
    center_y, center_x = grid_size[0] // 2, grid_size[1] // 2
    radius = min(grid_size) // 4
    
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            if ((y - center_y) ** 2 + (x - center_x) ** 2) < radius ** 2:
                mask[y, x] = False
    
    masked_dist.apply_mask(mask)
    
    # Plot all distributions
    distributions = [
        (uniform_dist, "Uniform", axes[0, 0]),
        (poisson_near, "Poisson (Near Goal)", axes[0, 1]),
        (poisson_far, "Poisson (Far from Goal)", axes[0, 2]),
        (gaussian_near, "Gaussian (Near Goal)", axes[1, 0]),
        (gaussian_far, "Gaussian (Far from Goal)", axes[1, 1]),
        (distance_near, "Distance-based (Near Goal)", axes[1, 2]),
        (distance_far, "Distance-based (Far from Goal)", axes[2, 0]),
        (multi_point, "Multi-point Gaussian", axes[2, 1]),
        (masked_dist, "Masked Uniform", axes[2, 2])
    ]
    
    for dist, title, ax in distributions:
        dist.visualize(title=title, ax=ax, show=False)
        # Mark goal position
        ax.plot(goal_pos[1], goal_pos[0], 'r*', markersize=10)
        
        # For multi-point, mark all reference points
        if title == "Multi-point Gaussian":
            for i, (y, x) in enumerate(points):
                if (y, x) != goal_pos:  # Skip goal position as it's already marked
                    ax.plot(x, y, 'go', markersize=8)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    combined_path = os.path.join(output_dir, "all_distributions.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined visualization to: {combined_path}")
    
    # Also save individual visualizations
    for dist, title, _ in distributions:
        plt.figure(figsize=(6, 6))
        dist.visualize(title=title, show=False)
        # Mark goal position
        plt.plot(goal_pos[1], goal_pos[0], 'r*', markersize=10)
        
        # For multi-point, mark all reference points
        if title == "Multi-point Gaussian":
            for i, (y, x) in enumerate(points):
                if (y, x) != goal_pos:  # Skip goal position as it's already marked
                    plt.plot(x, y, 'go', markersize=8)
        
        # Save individual plot
        clean_title = title.replace(" ", "_").replace("(", "").replace(")", "").lower()
        individual_path = os.path.join(output_dir, f"{clean_title}.png")
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {title} visualization to: {individual_path}")
    
    print(f"\nAll visualizations saved to: {output_dir}")
    return combined_path


if __name__ == "__main__":
    # Create output directory
    output_dir = "distribution_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all distribution visualizations
    combined_path = visualize_all_distributions(output_dir)
    
    # Show the combined visualization if running in an interactive environment
    try:
        plt.figure(figsize=(12, 12))
        img = plt.imread(combined_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Spawn Distribution Types")
        plt.show()
    except Exception as e:
        print(f"Note: Could not display the image in this environment: {e}")
        print(f"The visualization has been saved to: {combined_path}") 