#!/usr/bin/env python3
"""
Test script to visualize spawn distributions in isolation.
This script focuses solely on creating and visualizing distributions
without the complexity of agent training.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# Create the output directory
OUTPUT_DIR = "./spawn_distribution_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_empty_grid(width=8, height=8):
    """Create an empty grid with zeros."""
    return np.zeros((height, width))

def create_lava_grid(width=8, height=8, lava_type="gap"):
    """Create a grid with lava obstacles."""
    grid = create_empty_grid(width, height)
    lava_positions = []
    
    if lava_type == "gap":
        # Create a horizontal lava strip with a gap in the middle
        mid_y = height // 2
        for x in range(width):
            if x != width // 2:  # Gap in the middle
                lava_positions.append((x, mid_y))
    
    elif lava_type == "crossing":
        # Create a diagonal line of lava cells
        for i in range(min(width-2, height-2)):
            lava_positions.append((i+1, i+1))
    
    return grid, lava_positions

def apply_uniform_distribution(grid, lava_positions=None, goal_pos=None):
    """Apply a uniform distribution to the grid."""
    height, width = grid.shape
    grid.fill(1.0)
    
    # Zero out obstacles and goal
    if goal_pos:
        grid[goal_pos[1], goal_pos[0]] = 0.0
    
    if lava_positions:
        for x, y in lava_positions:
            if 0 <= y < height and 0 <= x < width:
                grid[y, x] = 0.0
    
    # Normalize
    if np.sum(grid) > 0:
        grid = grid / np.sum(grid)
    
    return grid

def apply_poisson_distribution(grid, goal_pos, lava_positions=None, lambda_param=1.0, favor_near=True):
    """Apply a Poisson distribution based on distance from goal."""
    height, width = grid.shape
    
    for y in range(height):
        for x in range(width):
            # Calculate Manhattan distance from goal
            distance = abs(x - goal_pos[0]) + abs(y - goal_pos[1])
            
            if favor_near:
                # Higher probability near the goal
                prob = np.exp(-lambda_param * distance)
            else:
                # Higher probability far from the goal
                max_distance = abs(0 - width) + abs(0 - height)
                prob = np.exp(-lambda_param * (max_distance - distance))
            
            grid[y, x] = prob
    
    # Zero out obstacles and goal
    grid[goal_pos[1], goal_pos[0]] = 0.0
    
    if lava_positions:
        for x, y in lava_positions:
            if 0 <= y < height and 0 <= x < width:
                grid[y, x] = 0.0
    
    # Normalize
    if np.sum(grid) > 0:
        grid = grid / np.sum(grid)
    
    return grid

def apply_gaussian_distribution(grid, goal_pos, lava_positions=None, sigma=2.0, favor_near=True):
    """Apply a Gaussian distribution based on distance from goal."""
    height, width = grid.shape
    
    for y in range(height):
        for x in range(width):
            # Calculate squared Euclidean distance from goal
            dist_squared = (x - goal_pos[0])**2 + (y - goal_pos[1])**2
            
            if favor_near:
                # Higher probability near the goal
                prob = np.exp(-dist_squared / (2 * sigma**2))
            else:
                # Higher probability far from the goal
                # Use maximum possible squared distance for inversion
                center_x, center_y = width // 2, height // 2
                max_dist_sq = (center_x - goal_pos[0])**2 + (center_y - goal_pos[1])**2
                prob = np.exp(-(max_dist_sq - dist_squared) / (2 * sigma**2))
            
            grid[y, x] = prob
    
    # Zero out obstacles and goal
    grid[goal_pos[1], goal_pos[0]] = 0.0
    
    if lava_positions:
        for x, y in lava_positions:
            if 0 <= y < height and 0 <= x < width:
                grid[y, x] = 0.0
    
    # Normalize
    if np.sum(grid) > 0:
        grid = grid / np.sum(grid)
    
    return grid

def apply_distance_distribution(grid, goal_pos, lava_positions=None, power=1.0, favor_near=True):
    """Apply a distance-based distribution."""
    height, width = grid.shape
    max_distance = abs(0 - width) + abs(0 - height)
    
    for y in range(height):
        for x in range(width):
            # Calculate Manhattan distance from goal
            distance = abs(x - goal_pos[0]) + abs(y - goal_pos[1])
            norm_distance = distance / max_distance
            
            if favor_near:
                prob = 1.0 - (norm_distance ** power)
            else:
                prob = norm_distance ** power
            
            grid[y, x] = prob
    
    # Zero out obstacles and goal
    grid[goal_pos[1], goal_pos[0]] = 0.0
    
    if lava_positions:
        for x, y in lava_positions:
            if 0 <= y < height and 0 <= x < width:
                grid[y, x] = 0.0
    
    # Normalize
    if np.sum(grid) > 0:
        grid = grid / np.sum(grid)
    
    return grid

def plot_distribution(grid, goal_pos=None, lava_positions=None, title="Spawn Distribution", 
                      filename=None):
    """Plot a spawn distribution."""
    height, width = grid.shape
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(grid, origin='upper', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Probability')
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Show coordinates
    ax.set_xticks(np.arange(0, width, 1))
    ax.set_yticks(np.arange(0, height, 1))
    
    # Add markers for goal and lava
    if goal_pos:
        ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Goal')
    
    if lava_positions and len(lava_positions) > 0:
        lava_x, lava_y = zip(*lava_positions)
        ax.scatter(lava_x, lava_y, c='red', marker='x', s=100, label='Lava')
    
    # Annotate cells with probability values
    for y in range(height):
        for x in range(width):
            # Skip cells with zero probability
            if grid[y, x] > 0:
                text = ax.text(x, y, f"{grid[y, x]:.4f}",
                           ha="center", va="center", color="w", fontsize=8)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Add legend if we have special markers
    if goal_pos or (lava_positions and len(lava_positions) > 0):
        ax.legend(loc='upper right')
    
    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {filename}")
    
    plt.close()

def print_ascii_visualization(grid, goal_pos=None, lava_positions=None, title=None):
    """Print an ASCII visualization of the spawn distribution."""
    height, width = grid.shape
    
    if title:
        print(f"\n=== {title} ===")
    
    # Define characters for different probability ranges
    # From lowest to highest probability
    chars = " .,:;+*#@"
    max_prob = np.max(grid) if np.max(grid) > 0 else 1.0
    
    # Print column headers
    header = "    "
    for x in range(width):
        header += f"{x}"
    print(header)
    
    # Print horizontal line
    print("   " + "-" * width)
    
    # Print the grid with ASCII characters
    for y in range(height):
        row = f"{y:2d} |"
        for x in range(width):
            if goal_pos and (x, y) == goal_pos:
                # Goal position
                row += "G"
            elif lava_positions and (x, y) in lava_positions:
                # Lava position
                row += "L"
            else:
                # Regular cell with probability indicator
                prob = grid[y, x]
                if prob <= 0:
                    row += " "  # Zero probability
                else:
                    # Map probability to character
                    char_idx = min(int(prob / max_prob * (len(chars) - 1)), len(chars) - 1)
                    row += chars[char_idx]
        print(row)
    
    # Print legend
    print("\nLegend:")
    print("G = Goal position (zero probability)")
    print("L = Lava cell (zero probability)")
    print(f"{chars} = Increasing probability (left to right)")

def print_numeric_distribution(grid, goal_pos=None, lava_positions=None, title=None, summary=True):
    """Print the numeric probability distribution."""
    if title:
        print(f"\n=== {title} ===")
    
    height, width = grid.shape
    
    print("\n  Probability Distribution Array:")
    
    # Print column headers
    header = "    "
    for x in range(width):
        header += f"{x:7d} "
    print(header)
    
    # Print horizontal line
    print("    " + "-" * (8 * width))
    
    # Format the probabilities with special symbols for goal and lava
    for y in range(height):
        row = f"{y:2d} |"
        for x in range(width):
            if goal_pos and (x, y) == goal_pos:
                # Goal position
                row += "  GOAL  "
            elif lava_positions and (x, y) in lava_positions:
                # Lava position
                row += "  LAVA  "
            else:
                # Regular cell with probability value
                prob = grid[y, x]
                row += f" {prob:.5f}"
        print(row)
    
    # Print legend
    print("\n  Legend:")
    print("  - GOAL: Goal position (zero probability)")
    print("  - LAVA: Lava cell (zero probability)")
    print("  - Values represent spawn probabilities (sum to 1.0)")
    
    # Print summary statistics
    if summary:
        nonzero_cells = np.count_nonzero(grid)
        total_cells = width * height
        print(f"\n  Summary:")
        print(f"  - Valid spawn cells: {nonzero_cells} of {total_cells} ({nonzero_cells/total_cells:.1%})")
        if nonzero_cells > 0:
            print(f"  - Highest probability: {np.max(grid):.5f}")
            print(f"  - Average probability (non-zero cells): {np.sum(grid)/nonzero_cells:.5f}")

def main():
    """Main function to test spawn distributions."""
    # Define grid size
    width, height = 7, 7
    
    # Define goal position (typically bottom-right corner)
    goal_pos = (width-2, height-2)  # (5, 5) for a 7x7 grid
    
    print("\n====== SPAWN DISTRIBUTION TEST ======")
    print(f"Grid size: {width}x{height}")
    print(f"Goal position: {goal_pos}")
    print("====================================\n")
    
    # Test different distribution types
    distribution_types = [
        {"name": "Uniform", "type": "uniform"},
        {"name": "Poisson (near goal)", "type": "poisson", "favor_near": True},
        {"name": "Poisson (far from goal)", "type": "poisson", "favor_near": False},
        {"name": "Gaussian (near goal)", "type": "gaussian", "favor_near": True},
        {"name": "Gaussian (far from goal)", "type": "gaussian", "favor_near": False},
        {"name": "Distance (near goal)", "type": "distance", "favor_near": True},
        {"name": "Distance (far from goal)", "type": "distance", "favor_near": False}
    ]
    
    # Test with different obstacle configurations
    obstacle_types = [
        {"name": "Empty", "type": None},
        {"name": "Lava Gap", "type": "gap"},
        {"name": "Lava Crossing", "type": "crossing"}
    ]
    
    # Generate and visualize each combination
    for obstacle in obstacle_types:
        print(f"\n==== {obstacle['name']} Environment ====")
        
        # Create grid with obstacles
        if obstacle["type"] is None:
            grid = create_empty_grid(width, height)
            lava_positions = []
        else:
            grid, lava_positions = create_lava_grid(width, height, obstacle["type"])
        
        for dist in distribution_types:
            print(f"\n--- {dist['name']} Distribution ---")
            
            # Apply the distribution
            if dist["type"] == "uniform":
                dist_grid = apply_uniform_distribution(grid.copy(), lava_positions, goal_pos)
            elif dist["type"] == "poisson":
                dist_grid = apply_poisson_distribution(
                    grid.copy(), goal_pos, lava_positions, 
                    lambda_param=1.0, favor_near=dist.get("favor_near", True)
                )
            elif dist["type"] == "gaussian":
                dist_grid = apply_gaussian_distribution(
                    grid.copy(), goal_pos, lava_positions,
                    sigma=2.0, favor_near=dist.get("favor_near", True)
                )
            elif dist["type"] == "distance":
                dist_grid = apply_distance_distribution(
                    grid.copy(), goal_pos, lava_positions,
                    power=1.0, favor_near=dist.get("favor_near", True)
                )
            
            # Print ASCII visualization
            print_ascii_visualization(
                dist_grid, goal_pos, lava_positions, 
                title=f"{dist['name']} Distribution - {obstacle['name']} Environment"
            )
            
            # Print numeric values
            print_numeric_distribution(
                dist_grid, goal_pos, lava_positions,
                title=f"{dist['name']} Distribution Values"
            )
            
            # Save visualization
            filename = os.path.join(
                OUTPUT_DIR, 
                f"{dist['type']}_{dist.get('favor_near', True)}_{obstacle['type'] or 'empty'}.png"
            )
            plot_distribution(
                dist_grid, goal_pos, lava_positions,
                title=f"{dist['name']} Distribution - {obstacle['name']} Environment",
                filename=filename
            )
    
    print("\n====== TEST COMPLETE ======")
    print(f"Visualizations saved to {OUTPUT_DIR}")
    print("============================\n")

if __name__ == "__main__":
    main() 