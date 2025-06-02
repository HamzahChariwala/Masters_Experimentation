import json
import numpy as np
import os

# Analyze the JSON file for a specific environment
def analyze_json(json_file):
    print(f"Analyzing {json_file}...")
    
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get the environment key
    env_key = list(data.keys())[0]
    
    # Extract information
    grid = data[env_key]["grid"]
    width = data[env_key]["width"]
    height = data[env_key]["height"]
    walls = data[env_key]["walls"]
    lava = data[env_key]["lava"]
    goals = data[env_key]["goal"]
    
    print(f"Environment: {env_key}")
    print(f"Dimensions: {width}x{height}")
    print(f"Walls: {len(walls)}")
    print(f"Lava cells: {len(lava)}")
    print(f"Goal cells: {len(goals)}")
    
    # Count cells with finite costs
    cells_with_costs = 0
    
    # Track cells with diagonal costs that are better than straight costs
    diagonal_better_cells = []
    
    # Create a grid to track cell costs
    cost_grid = np.full((height, width, 4), np.inf)
    for y in range(height):
        for x in range(width):
            if y < len(grid) and x < len(grid[y]):
                costs = grid[y][x]
                for dir_idx in range(min(4, len(costs))):
                    if dir_idx < len(costs):
                        cost_grid[y, x, dir_idx] = costs[dir_idx]
    
    # Create a grid to track minimum costs regardless of orientation
    min_cost_grid = np.full((height, width), np.inf)
    best_dir_grid = np.full((height, width), -1)
    
    for y in range(height):
        for x in range(width):
            costs = cost_grid[y, x]
            if any(cost != np.inf for cost in costs):
                cells_with_costs += 1
                
                # Find minimum cost for this cell
                min_cost = np.inf
                best_dir = -1
                for dir_idx, cost in enumerate(costs):
                    if cost < min_cost:
                        min_cost = cost
                        best_dir = dir_idx
                
                min_cost_grid[y, x] = min_cost
                best_dir_grid[y, x] = best_dir
    
    print(f"Cells with finite costs: {cells_with_costs} out of {width * height - len(walls)}")
    
    # Check for cells that have costs but are drawn as white (unreachable)
    cells_with_costs_coords = []
    for y in range(height):
        for x in range(width):
            if min_cost_grid[y, x] != np.inf:
                cells_with_costs_coords.append((x, y, min_cost_grid[y, x], best_dir_grid[y, x]))
    
    # Determine if there are potential diagonal moves
    diagonal_moves_possible = []
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up
    
    for x, y, cost, dir_idx in cells_with_costs_coords:
        if dir_idx != -1:
            dx, dy = directions[dir_idx]
            target_x, target_y = x + dx, y + dy
            
            # Check if this is a potential diagonal move
            if 0 <= target_x < width and 0 <= target_y < height:
                # Check if diagonals would be better
                # This is a very simplified check and may not match the actual algorithm
                for diag_idx, (diag_dx, diag_dy) in enumerate([
                    (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal directions
                ]):
                    diag_x, diag_y = x + diag_dx, y + diag_dy
                    
                    # Check if diagonal target is valid
                    if 0 <= diag_x < width and 0 <= diag_y < height:
                        # Check if diagonal target has a cost
                        if min_cost_grid[diag_y, diag_x] != np.inf:
                            # Check if diagonal path is better than going around
                            if min_cost_grid[diag_y, diag_x] + 1.4 < cost:
                                diagonal_moves_possible.append((x, y, diag_x, diag_y, 
                                                              min_cost_grid[diag_y, diag_x] + 1.4, cost))
    
    if diagonal_moves_possible:
        print("\nPotential diagonal moves that would be better:")
        for x, y, diag_x, diag_y, diag_cost, current_cost in diagonal_moves_possible[:10]:
            print(f"  From ({x},{y}) to ({diag_x},{diag_y}): {diag_cost:.1f} vs current {current_cost:.1f}")
        if len(diagonal_moves_possible) > 10:
            print(f"  ... and {len(diagonal_moves_possible) - 10} more")
    else:
        print("\nNo potential diagonal moves found that would be better than current paths.")
    
    # Check for cells that have arrows pointing into lava
    arrows_to_lava = []
    lava_coords = set((x, y) for x, y in lava)
    
    for x, y, cost, dir_idx in cells_with_costs_coords:
        if dir_idx != -1:
            dx, dy = directions[dir_idx]
            target_x, target_y = x + dx, y + dy
            
            if (target_x, target_y) in lava_coords:
                arrows_to_lava.append((x, y, target_x, target_y))
    
    if arrows_to_lava:
        print("\nCells with arrows pointing into lava:")
        for x, y, lava_x, lava_y in arrows_to_lava[:10]:
            print(f"  From ({x},{y}) to lava at ({lava_x},{lava_y})")
        if len(arrows_to_lava) > 10:
            print(f"  ... and {len(arrows_to_lava) - 10} more")
    else:
        print("\nNo arrows pointing into lava found.")
    
    return {
        "data": data,
        "cells_with_costs": cells_with_costs,
        "min_cost_grid": min_cost_grid,
        "best_dir_grid": best_dir_grid,
        "diagonal_moves_possible": diagonal_moves_possible,
        "arrows_to_lava": arrows_to_lava
    }

# Main function
def main():
    # List of modes to analyze
    modes = ["normal", "costly", "blocked"]
    
    # Analyze LavaCrossing environment for seed 1
    for mode in modes:
        json_file = f"results/MiniGrid_LavaCrossingS9N1_v0_1_{mode}.json"
        if os.path.exists(json_file):
            print("\n" + "="*50)
            analyze_json(json_file)
            print("="*50)

if __name__ == "__main__":
    main() 