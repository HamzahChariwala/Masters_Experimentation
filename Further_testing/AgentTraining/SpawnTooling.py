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
        
    print("\n====== SPAWN DISTRIBUTION ======")
    print(f"Flexible spawn distribution is enabled")
    print(f"Distribution: {env_params.get('spawn_distribution_type', 'undefined')}")
    
    # Get environment ID if available
    env_id = env_params.get("env_id", None)
    
    # Extract grid size from environment ID
    width, height = 8, 8  # Default size for most MiniGrid environments
    
    if env_id:
        # Check if it's a sized environment (like S11N5)
        import re
        size_match = re.search(r"S(\d+)", env_id)
        if size_match:
            size = int(size_match.group(1))
            width, height = size, size
        else:
            # Try to extract size from format like 8x8
            size_match = re.search(r"(\d+)x(\d+)", env_id)
            if size_match:
                width = int(size_match.group(1))
                height = int(size_match.group(2))
    
    # Determine environment type based on environment ID
    env_type = "empty"  # Default
    if env_id:
        if "LavaCrossing" in env_id:
            env_type = "lava_crossing"
        elif "LavaGap" in env_id:
            env_type = "lava_gap"
    
    # Check if stage training is enabled 
    if env_params.get("use_stage_training", False):
        stage_config = env_params.get("stage_training_config", {})
        num_stages = stage_config.get("num_stages", 1)
        distributions = stage_config.get("distributions", [])
        
        print(f"Using stage-based curriculum with {num_stages} stages")
        timesteps_per_stage = total_timesteps / num_stages
        
        # Display each stage
        for i, dist_config in enumerate(distributions):
            start_step = int(i * timesteps_per_stage)
            end_step = int((i+1) * timesteps_per_stage)
            
            dist_type = dist_config.get("type", "uniform")
            params = dist_config.get("params", {})
            favor_near = params.get("favor_near", True)
            
            print(f"\n  Stage {i+1}: {dist_type}")
            if params:
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                print(f"  Parameters: {param_str}")
            print(f"  Timesteps: {start_step:,} to {end_step:,}")
            
            if use_numeric_display:
                # Generate and display distribution for this stage
                grid, env_info = generate_numeric_distribution(
                    dist_type, 
                    favor_near=favor_near,
                    width=width,
                    height=height,
                    env_type=env_type,
                    env_id=env_id
                )
                
                # Display detailed numeric distribution
                print_numeric_distribution(
                    grid, 
                    env_info["goal_pos"],
                    env_info["lava_positions"],
                    env_info["wall_positions"],
                    title=f"Stage {i+1} Distribution"
                )
            else:
                # Text-only fallback
                print(f"  {dist_type} distribution {'near goal' if favor_near else 'far from goal'}")
    
    # Check if continuous transition is enabled
    elif env_params.get("use_continuous_transition", False):
        transition_config = env_params.get("continuous_transition_config", {})
        rate = transition_config.get("rate", 1.0)
        
        initial_type = transition_config.get("initial_type", "uniform")
        initial_params = transition_config.get("initial_params", {})
        initial_favor_near = initial_params.get("favor_near", True)
        
        target_type = transition_config.get("target_type", "uniform")
        target_params = transition_config.get("target_params", {})
        target_favor_near = target_params.get("favor_near", False)
        
        print(f"Using continuous curriculum transition")
        print(f"Initial: {initial_type} {'near goal' if initial_favor_near else 'far from goal'}")
        print(f"Target: {target_type} {'near goal' if target_favor_near else 'far from goal'}")
        print(f"Transition rate: {rate}")
        
        if use_numeric_display:
            # Show initial distribution
            initial_grid, initial_info = generate_numeric_distribution(
                initial_type, 
                favor_near=initial_favor_near,
                width=width,
                height=height,
                env_type=env_type,
                env_id=env_id
            )
            
            print_numeric_distribution(
                initial_grid, 
                initial_info["goal_pos"],
                initial_info["lava_positions"],
                initial_info["wall_positions"],
                title="Initial Distribution"
            )
            
            # Show target distribution
            target_grid, target_info = generate_numeric_distribution(
                target_type, 
                favor_near=target_favor_near,
                width=width,
                height=height,
                env_type=env_type,
                env_id=env_id
            )
            
            print_numeric_distribution(
                target_grid, 
                target_info["goal_pos"],
                target_info["lava_positions"],
                target_info["wall_positions"],
                title="Target Distribution"
            )
    
    # Regular fixed distribution
    else:
        dist_type = env_params.get("spawn_distribution_type", "uniform")
        dist_params = env_params.get("spawn_distribution_params", {})
        favor_near = dist_params.get("favor_near", True)
        
        print(f"Using fixed distribution: {dist_type}")
        if dist_params:
            param_str = ", ".join([f"{k}={v}" for k, v in dist_params.items()])
            print(f"Parameters: {param_str}")
        
        if use_numeric_display:
            # Generate and display the distribution
            grid, env_info = generate_numeric_distribution(
                dist_type, 
                favor_near=favor_near,
                width=width,
                height=height,
                env_type=env_type,
                env_id=env_id
            )
            
            # Display detailed numeric distribution
            print_numeric_distribution(
                grid, 
                env_info["goal_pos"],
                env_info["lava_positions"],
                env_info["wall_positions"],
                title="Spawn Distribution"
            )
        else:
            # Text-only fallback
            print(f"{dist_type} distribution {'near goal' if favor_near else 'far from goal'}")
            
        # Extra info about valid spawn positions
        valid_cells = (grid > 0).sum()
        total_cells = width * height
        print(f"\nValid spawn positions: {valid_cells} out of {total_cells} cells")
        print(f"Walls and obstacles: {total_cells - valid_cells} cells (including goal, lava, and walls)")
    
    print("===============================\n")


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