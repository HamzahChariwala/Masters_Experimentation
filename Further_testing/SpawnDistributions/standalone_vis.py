"""
Standalone script to demonstrate ASCII visualizations of spawn distributions.
This script doesn't require imports from main.py.
"""
import os
import numpy as np
from pprint import pprint

# Define example distribution types to visualize
DISTRIBUTIONS = [
    {
        "name": "Uniform Distribution",
        "type": "uniform"
    },
    {
        "name": "Poisson Distribution (near goal)",
        "type": "poisson_goal",
        "favor_near": True 
    },
    {
        "name": "Poisson Distribution (far from goal)",
        "type": "poisson_goal",
        "favor_near": False
    },
    {
        "name": "Gaussian Distribution (near goal)",
        "type": "gaussian_goal",
        "favor_near": True
    },
    {
        "name": "Gaussian Distribution (far from goal)",
        "type": "gaussian_goal",
        "favor_near": False
    },
    {
        "name": "Distance-based Distribution (near goal)",
        "type": "distance_goal",
        "favor_near": True
    },
    {
        "name": "Distance-based Distribution (far from goal)",
        "type": "distance_goal",
        "favor_near": False
    },
]

# Example stage-based training configurations
STAGE_CONFIGS = [
    {
        "name": "Near to Far Training (4 stages)",
        "num_stages": 4,
        "total_timesteps": 1_000_000,
        "distributions": [
            {"type": "poisson_goal", "params": {"lambda_param": 1.0, "favor_near": True}},
            {"type": "distance_goal", "params": {"favor_near": True, "power": 1}},
            {"type": "gaussian_goal", "params": {"sigma": 2.0, "favor_near": False}},
            {"type": "uniform"}
        ]
    },
    {
        "name": "Simple 2-Stage Training",
        "num_stages": 2,
        "total_timesteps": 100_000,
        "distributions": [
            {"type": "poisson_goal", "params": {"lambda_param": 1.0, "favor_near": True}},
            {"type": "poisson_goal", "params": {"lambda_param": 1.0, "favor_near": False}}
        ]
    }
]

# Example continuous transition configurations
CONTINUOUS_CONFIGS = [
    {
        "name": "Linear Near to Far Transition",
        "initial_type": "poisson_goal",
        "initial_params": {"lambda_param": 1.0, "favor_near": True},
        "target_type": "uniform",
        "rate": 1.0,
        "total_timesteps": 1_000_000
    },
    {
        "name": "Slow-start Transition (Cubic)",
        "initial_type": "gaussian_goal",
        "initial_params": {"sigma": 1.0, "favor_near": True},
        "target_type": "gaussian_goal",
        "target_params": {"sigma": 1.0, "favor_near": False},
        "rate": 3.0,
        "total_timesteps": 500_000
    }
]

def generate_numeric_distribution(dist_type, favor_near=True, width=7, height=7, env_type="empty"):
    """
    Generate a numeric probability distribution.
    
    Parameters:
    ----------
    dist_type : str
        Type of distribution (uniform, poisson_goal, gaussian_goal, distance_goal)
    favor_near : bool
        Whether to favor positions near the goal
    width, height : int
        Grid dimensions
    env_type : str
        Environment type ("empty", "lava_gap", "lava_crossing")
    
    Returns:
    -------
    numpy.ndarray
        Probability distribution array
    dict
        Environment information (goal position, lava positions)
    """
    # Initialize grid and environment information
    grid = np.zeros((height, width))
    goal_pos = (width-2, height-2)  # Typical goal position (bottom right)
    lava_positions = []
    
    # Configure environment based on type
    if env_type == "lava_gap":
        # Create a horizontal lava strip with a gap in the middle
        mid_y = height // 2
        for x in range(width):
            if x != width // 2:  # Gap in the middle
                lava_positions.append((x, mid_y))
    
    elif env_type == "lava_crossing":
        # Create a diagonal line of lava cells
        for i in range(min(width-2, height-2)):
            lava_positions.append((i+1, i+1))
    
    # Generate initial distribution
    if dist_type == "uniform":
        grid.fill(1.0 / (width * height))
    
    else:
        # Get goal position
        goal_x, goal_y = goal_pos
        
        # Fill grid with appropriate distribution values
        for y in range(height):
            for x in range(width):
                # Calculate Manhattan distance from goal
                distance = abs(x - goal_x) + abs(y - goal_y)
                max_distance = abs(0 - width) + abs(0 - height)
                
                # Normalized distance [0, 1]
                norm_distance = distance / max_distance
                
                # Calculate density based on distribution type and favor_near
                if dist_type == "poisson_goal":
                    lambda_param = 1.0
                    prob = np.exp(-lambda_param * distance)
                    
                    if not favor_near:
                        # Invert to favor positions far from goal
                        prob = np.exp(-lambda_param * (max_distance - distance))
                        
                elif dist_type == "gaussian_goal":
                    sigma = 2.0
                    dist_squared = (x - goal_x)**2 + (y - goal_y)**2
                    prob = np.exp(-dist_squared / (2 * sigma**2))
                    
                    if not favor_near:
                        # Calculate distance from center to use for inversion
                        center_x, center_y = width // 2, height // 2
                        max_dist_sq = (center_x - goal_x)**2 + (center_y - goal_y)**2
                        # Invert to favor positions far from goal
                        prob = np.exp(-(max_dist_sq - dist_squared) / (2 * sigma**2))
                        
                elif dist_type == "distance_goal":
                    power = 1.0
                    
                    if favor_near:
                        prob = 1.0 - (norm_distance ** power)
                    else:
                        prob = norm_distance ** power
                
                grid[y, x] = prob
    
    # Zero out goal position (can't spawn on goal)
    grid[goal_pos[1], goal_pos[0]] = 0.0
    
    # Zero out lava positions
    for x, y in lava_positions:
        if 0 <= y < height and 0 <= x < width:
            grid[y, x] = 0.0
    
    # Normalize to make a valid probability distribution
    if np.sum(grid) > 0:
        grid = grid / np.sum(grid)
    
    # Create environment info dictionary
    env_info = {
        "goal_pos": goal_pos,
        "lava_positions": lava_positions,
        "distribution_type": dist_type,
        "favor_near": favor_near
    }
    
    return grid, env_info

def print_numeric_distribution(grid, env_info, show_legend=True):
    """
    Print the actual numeric probability distribution with formatting.
    
    Parameters:
    ----------
    grid : numpy.ndarray
        Probability distribution
    env_info : dict
        Environment information (goal position, lava positions)
    show_legend : bool
        Whether to show the legend for special cells
    """
    height, width = grid.shape
    goal_pos = env_info["goal_pos"]
    lava_positions = env_info["lava_positions"]
    
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
            if (x, y) == goal_pos:
                # Goal position
                row += "  GOAL  "
            elif (x, y) in lava_positions:
                # Lava position
                row += "  LAVA  "
            else:
                # Regular cell with probability value
                prob = grid[y, x]
                row += f" {prob:.5f}"
        print(row)
    
    # Print legend
    if show_legend:
        print("\n  Legend:")
        print("  - GOAL: Goal position (zero probability)")
        print("  - LAVA: Lava cell (zero probability)")
        print("  - Values represent spawn probabilities (sum to 1.0)")
        
    # Print summary statistics
    nonzero_cells = np.count_nonzero(grid)
    total_cells = width * height
    print(f"\n  Summary:")
    print(f"  - Valid spawn cells: {nonzero_cells} of {total_cells} ({nonzero_cells/total_cells:.1%})")
    print(f"  - Highest probability: {np.max(grid):.5f}")
    print(f"  - Distribution type: {env_info['distribution_type']}")
    if env_info['distribution_type'] != "uniform":
        near_far = "near goal" if env_info['favor_near'] else "far from goal"
        print(f"  - Direction: {near_far}")

def print_ascii_visualization(grid, env_info, show_legend=True):
    """
    Print an ASCII visualization of the spawn distribution.
    
    Parameters:
    ----------
    grid : numpy.ndarray
        Probability distribution
    env_info : dict
        Environment information (goal position, lava positions)
    show_legend : bool
        Whether to show the legend
    """
    height, width = grid.shape
    goal_pos = env_info["goal_pos"]
    lava_positions = env_info["lava_positions"]
    
    print("\n  ASCII Visualization:")
    
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
            if (x, y) == goal_pos:
                # Goal position
                row += "G"
            elif (x, y) in lava_positions:
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
    if show_legend:
        print("\n  Legend:")
        print("  G = Goal position (zero probability)")
        print("  L = Lava cell (zero probability)")
        print(f"  {chars} = Increasing probability (left to right)")

def print_transition_timeline(rate, total_timesteps):
    """
    Print a simple visualization of transition speed.
    
    Parameters:
    ----------
    rate : float
        Power to which progress is raised (higher = slower start)
    total_timesteps : int
        Total number of timesteps
    """
    print("\n  Transition speed:")
    points = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("  Progress |", end="")
    for p in points:
        print(f" {p:.0%} |", end="")
    print("\n  ---------+-----+-----+-----+-----+-----")
    
    print("  Timestep |", end="")
    for p in points:
        # Solve for timestep: (t/total)^rate = p
        # t = total * p^(1/rate)
        if rate > 0:
            timestep = int(total_timesteps * (p ** (1.0 / rate)))
        else:
            timestep = int(total_timesteps * p)
        print(f" {timestep:,} |", end="")
    print()

def print_spawn_distribution_info(config_type, config):
    """
    Print numeric and ASCII visualization for a spawn distribution configuration.
    
    Parameters:
    ----------
    config_type : str
        Type of configuration: 'distribution', 'stage', or 'continuous'
    config : dict
        Configuration parameters
    """
    if config_type == 'distribution':
        print(f"\n=== {config['name']} ===")
        
        # Generate distribution for empty and lava environments
        empty_grid, empty_info = generate_numeric_distribution(
            config['type'], 
            config.get('favor_near', True), 
            env_type="empty"
        )
        
        lava_grid, lava_info = generate_numeric_distribution(
            config['type'], 
            config.get('favor_near', True),
            env_type="lava_gap"
        )
        
        # Print distributions
        print("\nEmpty Environment:")
        print_ascii_visualization(empty_grid, empty_info)
        print_numeric_distribution(empty_grid, empty_info)
        
        print("\nLava Gap Environment:")
        print_ascii_visualization(lava_grid, lava_info)
        print_numeric_distribution(lava_grid, lava_info)
        
        print("-" * 50)
    
    elif config_type == 'stage':
        print(f"\n=== {config['name']} ===")
        print(f"Using stage-based curriculum with {config['num_stages']} stages")
        timesteps_per_stage = config['total_timesteps'] / config['num_stages']
        
        for i, dist in enumerate(config['distributions']):
            start_step = int(i * timesteps_per_stage)
            end_step = int((i+1) * timesteps_per_stage)
            
            # Format parameters nicely
            param_str = ""
            if "params" in dist:
                param_parts = []
                for k, v in dist["params"].items():
                    param_parts.append(f"{k}={v}")
                if param_parts:
                    param_str = ", " + ", ".join(param_parts)
            
            print(f"\n  Stage {i+1}: {dist['type']}{param_str}")
            print(f"  Timesteps: {start_step:,} to {end_step:,}")
            
            # Generate and print for lava environment only
            favor_near = dist.get("params", {}).get("favor_near", True)
            grid, info = generate_numeric_distribution(
                dist['type'], 
                favor_near,
                env_type="lava_gap"
            )
            
            print_ascii_visualization(grid, info, show_legend=(i==0))
            print_numeric_distribution(grid, info, show_legend=(i==0))
        
        print("-" * 50)
    
    elif config_type == 'continuous':
        print(f"\n=== {config['name']} ===")
        print(f"Using continuous curriculum learning")
        
        # Print initial distribution
        favor_near = config.get("initial_params", {}).get("favor_near", True)
        init_grid, init_info = generate_numeric_distribution(
            config['initial_type'], 
            favor_near,
            env_type="lava_gap"
        )
        
        print("\nInitial Distribution:")
        print_ascii_visualization(init_grid, init_info)
        print_numeric_distribution(init_grid, init_info)
        
        # Generate target distribution
        target_favor_near = config.get("target_params", {}).get("favor_near", False)
        target_grid, target_info = generate_numeric_distribution(
            config['target_type'], 
            target_favor_near,
            env_type="lava_gap"
        )
        
        print("\nTarget Distribution:")
        print_ascii_visualization(target_grid, target_info)
        print_numeric_distribution(target_grid, target_info)
        
        print(f"\nTransition rate: {config['rate']}")
        print_transition_timeline(config['rate'], config['total_timesteps'])
        
        print("-" * 50)

def main():
    print("\n====== SPAWN DISTRIBUTION VISUALIZATIONS ======")
    print("Showing ASCII and numeric visualizations including obstacles (lava)")
    print("=================================================")
    
    # Show selected distributions
    print("\n== DISTRIBUTION EXAMPLES WITH LAVA ==")
    # Limit to a few key examples to keep output manageable
    selected_dists = [DISTRIBUTIONS[0], DISTRIBUTIONS[1], DISTRIBUTIONS[2]]
    for dist in selected_dists:
        print_spawn_distribution_info('distribution', dist)
    
    # Show one stage-based example
    print("\n== STAGE-BASED TRAINING EXAMPLE WITH LAVA ==")
    print_spawn_distribution_info('stage', STAGE_CONFIGS[0])
    
    # Show one continuous example
    print("\n== CONTINUOUS TRANSITION EXAMPLE WITH LAVA ==")
    print_spawn_distribution_info('continuous', CONTINUOUS_CONFIGS[0])
    
    print("\n===============================================")
    print("These visualizations show both ASCII representations")
    print("and actual numeric probability values.")
    print("Note that goals and lava cells have zero probability.")
    print("===============================================")

if __name__ == "__main__":
    main() 