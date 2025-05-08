def print_spawn_distribution_info(env_params, total_timesteps):
    """
    Print a text representation of spawn distribution information.
    
    Parameters:
    ----------
    env_params : dict
        Environment parameters including spawn distribution settings
    total_timesteps : int
        Total training timesteps
    """
    # Import visualization functions from SpawnDistributions
    try:
        from SpawnDistributions.distribution_inspector import print_numeric_distribution
        from SpawnDistributions.standalone_vis import generate_numeric_distribution
        use_numeric_display = True
    except ImportError:
        use_numeric_display = False
    
    # Check if flexible spawn is enabled
    if not env_params.get("use_flexible_spawn", False):
        print("\n====== SPAWN DISTRIBUTION ======")
        print("Flexible spawn distribution is disabled")
        print("Using default MiniGrid random spawn")
        print("===============================\n")
        return
        
    print("\n====== SPAWN DISTRIBUTION CONFIGURATION ======")
    print(f"Exclude goal-adjacent spawns: {env_params.get('exclude_goal_adjacent', False)}")
    
    # Handle stage-based training
    if env_params.get("use_stage_training", False) and env_params.get("stage_training_config"):
        config = env_params["stage_training_config"]
        num_stages = config["num_stages"]
        print(f"Using stage-based curriculum with {num_stages} stages:")
        
        # Calculate approximate timesteps per stage
        timesteps_per_stage = total_timesteps / num_stages
        
        for i, dist in enumerate(config["distributions"]):
            dist_type = dist["type"]
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
            
            print(f"  Stage {i+1}: {dist_type}{param_str}")
            print(f"    Timesteps: {start_step:,} to {end_step:,}")
            
            # Use numeric distribution visualization if available
            if use_numeric_display and dist_type in ["poisson_goal", "gaussian_goal", "distance_goal", "uniform"]:
                favor_near = dist.get("params", {}).get("favor_near", True)
                
                # Generate grid for visualization 
                grid, info = generate_numeric_distribution(
                    dist_type, 
                    favor_near=favor_near,
                    env_type="empty"  # Use empty environment for simplicity
                )
                
                # Print stage number as title
                title = f"Stage {i+1} Distribution ({dist_type})"
                print_numeric_distribution(grid, info["goal_pos"], info["lava_positions"], title=title)
            else:
                # Use simple ASCII visualization as fallback
                if dist_type in ["poisson_goal", "gaussian_goal", "distance_goal"]:
                    favor_near = dist.get("params", {}).get("favor_near", True)
                    direction = "near goal" if favor_near else "far from goal"
                    print(f"    Direction: {direction}")
                    _print_simple_distribution_visual(dist_type, favor_near)
    
    # Handle continuous transition
    elif env_params.get("use_continuous_transition", False) and env_params.get("continuous_transition_config"):
        config = env_params["continuous_transition_config"]
        initial_type = env_params["spawn_distribution_type"]
        target_type = config["target_type"]
        rate = config.get("rate", 1.0)
        
        print(f"Using continuous curriculum learning:")
        print(f"  Initial distribution: {initial_type}")
        
        # Format parameters nicely
        if env_params.get("spawn_distribution_params"):
            param_parts = []
            for k, v in env_params["spawn_distribution_params"].items():
                param_parts.append(f"{k}={v}")
            if param_parts:
                print(f"    Parameters: {', '.join(param_parts)}")
                
        # Use numeric distribution visualization if available
        if use_numeric_display and initial_type in ["poisson_goal", "gaussian_goal", "distance_goal", "uniform"]:
            favor_near = env_params.get("spawn_distribution_params", {}).get("favor_near", True)
            
            # Generate initial distribution
            init_grid, init_info = generate_numeric_distribution(
                initial_type, 
                favor_near=favor_near,
                env_type="empty"  # Use empty environment for simplicity
            )
            
            # Print visualization
            print_numeric_distribution(init_grid, init_info["goal_pos"], init_info["lava_positions"], 
                                     title="Initial Distribution")
            
            # Generate target distribution
            target_grid, target_info = generate_numeric_distribution(
                target_type, 
                favor_near=favor_near,  # Use same favor_near setting
                env_type="empty"  # Use empty environment for simplicity
            )
            
            # Print visualization
            print_numeric_distribution(target_grid, target_info["goal_pos"], target_info["lava_positions"], 
                                     title="Target Distribution")
        else:
            # Use simple ASCII visualization as fallback
            if initial_type in ["poisson_goal", "gaussian_goal", "distance_goal"]:
                favor_near = env_params.get("spawn_distribution_params", {}).get("favor_near", True)
                direction = "near goal" if favor_near else "far from goal"
                print(f"    Direction: {direction}")
                _print_simple_distribution_visual(initial_type, favor_near)
            
        print(f"\n  Target distribution: {target_type}")
        print(f"  Transition rate: {rate}")
        
        # Print a simple transition timeline
        _print_transition_timeline(rate, total_timesteps)
        
    # Handle fixed distribution
    else:
        dist_type = env_params.get("spawn_distribution_type", "uniform")
        print(f"Using fixed spawn distribution: {dist_type}")
        
        # Format parameters nicely
        if env_params.get("spawn_distribution_params"):
            param_parts = []
            for k, v in env_params["spawn_distribution_params"].items():
                param_parts.append(f"{k}={v}")
            if param_parts:
                print(f"  Parameters: {', '.join(param_parts)}")
                
        # Use numeric distribution visualization if available
        if use_numeric_display and dist_type in ["poisson_goal", "gaussian_goal", "distance_goal", "uniform"]:
            favor_near = env_params.get("spawn_distribution_params", {}).get("favor_near", True)
            
            # Generate grid for visualization
            grid, info = generate_numeric_distribution(
                dist_type, 
                favor_near=favor_near,
                env_type="empty"  # Use empty environment for simplicity
            )
            
            # Print visualization
            print_numeric_distribution(grid, info["goal_pos"], info["lava_positions"], 
                                     title=f"Fixed Distribution ({dist_type})")
        else:
            # Use simple ASCII visualization as fallback
            if dist_type in ["poisson_goal", "gaussian_goal", "distance_goal"]:
                favor_near = env_params.get("spawn_distribution_params", {}).get("favor_near", True)
                direction = "near goal" if favor_near else "far from goal"
                print(f"  Direction: {direction}")
                _print_simple_distribution_visual(dist_type, favor_near)
    
    # Print additional statistics about visualization
    if env_params.get("spawn_vis_frequency", 0) > 0:
        print(f"\nVisualization frequency: Every {env_params.get('spawn_vis_frequency'):,} steps")
        print(f"Visualization directory: {env_params.get('spawn_vis_dir', './spawn_vis')}")
    
    print("===========================================\n")


def _print_simple_distribution_visual(dist_type, favor_near):
    """Print a simple ASCII visualization of the distribution."""
    size = 11  # Grid size (odd number to have a center)
    center = size // 2
    
    print("\n    Distribution Pattern (G = goal):")
    
    # Create a grid with empty spaces
    grid = [[' ' for _ in range(size)] for _ in range(size)]
    
    # Place a G in the center to represent the goal
    grid[center][center] = 'G'
    
    # Define a set of characters to represent density
    characters = ' .,:;+*#@'  # From low to high density
    
    # Fill the grid with density indicators
    for y in range(size):
        for x in range(size):
            if x == center and y == center:
                continue  # Skip the goal position
                
            # Calculate Manhattan distance from goal
            distance = abs(x - center) + abs(y - center)
            max_distance = 2 * center
            
            # Normalized distance [0, 1]
            norm_distance = distance / max_distance
            
            # Calculate density based on distribution type and favor_near
            density = 0
            if dist_type == "uniform":
                density = 0.5  # Uniform density
            elif favor_near:
                # Higher density near the goal
                if dist_type == "poisson_goal":
                    density = max(0, 1 - norm_distance * 1.5)
                elif dist_type == "gaussian_goal":
                    density = max(0, 1 - (norm_distance * 1.2) ** 2)
                elif dist_type == "distance_goal":
                    density = max(0, 1 - norm_distance ** 1.5)
            else:
                # Higher density far from the goal
                if dist_type == "poisson_goal":
                    density = min(1, norm_distance * 1.5)
                elif dist_type == "gaussian_goal":
                    density = min(1, (norm_distance * 1.2) ** 2)
                elif dist_type == "distance_goal":
                    density = min(1, norm_distance ** 1.5)
            
            # Convert density to character (skip the goal position)
            if not (x == center and y == center):
                # Map density [0,1] to character index
                char_idx = min(len(characters) - 1, int(density * len(characters)))
                grid[y][x] = characters[char_idx]
    
    # Print the grid
    for row in grid:
        print("    " + ''.join(row))
    print()


def _print_transition_timeline(rate, total_timesteps):
    """Print a simple timeline showing the continuous transition."""
    width = 50  # Width of the timeline
    print("\n    Transition Timeline:")
    print("    Start " + "-" * width + " End")
    
    # Calculate marker positions for different percentages
    markers = []
    for pct in [0.25, 0.5, 0.75]:
        # Apply the rate function to calculate the actual position
        # rate > 1 means slow start, fast finish
        # rate < 1 means fast start, slow finish
        position = int(width * (pct ** rate))
        markers.append((position, f"{int(pct * 100)}%"))
    
    # Print the markers
    marker_line = "         "
    for pos, label in markers:
        space = " " * (pos - len(marker_line))
        marker_line += space + "↓"
    print(marker_line)
    
    # Print the percentages
    pct_line = "         "
    for pos, label in markers:
        space = " " * (pos - len(pct_line))
        pct_line += space + label
    print(pct_line)
    print()

# Explicitly add the function to the module's namespace
__all__ = ["print_spawn_distribution_info"]