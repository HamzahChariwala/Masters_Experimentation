import numpy as np
import random
import re

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
    # Import visualization functions from SpawnDistribution
    try:
        from Environment_Tooling.BespokeEdits.SpawnDistribution import print_numeric_distribution
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
    
    # Extract parameters from environment configuration
    dist_type = env_params.get("spawn_distribution_type", "uniform")
    params = env_params.get("spawn_distribution_params", {})
    favor_near = params.get("favor_near", True)
    
    # Find environment dimensions from environment ID
    width, height = 8, 8  # Default
    env_id = env_params.get("env_id", "")
    if "Empty" in env_id:
        match = re.search(r"Empty-(\d+)x(\d+)", env_id)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
    elif "LavaCrossingS" in env_id:
        match = re.search(r"LavaCrossingS(\d+)", env_id)
        if match:
            size = int(match.group(1))
            width, height = size, size
    elif "LavaGapS" in env_id:
        match = re.search(r"LavaGapS(\d+)", env_id)
        if match:
            size = int(match.group(1))
            width, height = size, size
    
    # Determine environment type
    env_type = "empty"
    if "LavaCrossing" in env_id:
        env_type = "lava_crossing"
    elif "LavaGap" in env_id:
        env_type = "lava_gap"
    
    print(f"\n====== SPAWN DISTRIBUTION: {dist_type.upper()} ======")
    
    if use_numeric_display:
        # Stage-based training info
        if env_params.get("use_stage_training", False):
            stage_config = env_params.get("stage_training_config", {})
            
            print("Using stage-based curriculum training:")
            print(f"Number of stages: {stage_config.get('num_stages', 0)}")
            
            # Curriculum proportion
            curriculum_prop = stage_config.get('curriculum_proportion', 1.0)
            curriculum_timesteps = int(total_timesteps * curriculum_prop)
            print(f"Curriculum ends at: {curriculum_timesteps}/{total_timesteps} timesteps ({curriculum_prop:.1%})")
            
            # Smooth transitions
            smooth_config = stage_config.get('smooth_transitions', {})
            if smooth_config.get('enabled', False):
                print("Using smooth transitions between stages")
                print(f"Transition proportion: {smooth_config.get('transition_proportion', 0.2):.1%} of each stage")
            
            # Display stage information
            distributions = stage_config.get('distributions', [])
            total_relative_duration = stage_config.get('total_relative_duration', 0)
            
            # Calculate total_relative_duration if it's zero
            if total_relative_duration == 0 and distributions:
                total_relative_duration = sum(dist.get('relative_duration', 1.0) for dist in distributions)
                if total_relative_duration == 0:
                    total_relative_duration = len(distributions)  # Fallback to equal durations
            
            print("\nStage progression:")
            for i, dist in enumerate(distributions):
                rel_duration = dist.get('relative_duration', 1.0)
                stage_duration = int((rel_duration / total_relative_duration) * curriculum_timesteps)
                
                print(f"\nStage {i+1}:")
                print(f"  Distribution type: {dist.get('type', 'uniform')}")
                if 'description' in dist:
                    print(f"  Description: {dist['description']}")
                print(f"  Duration: {stage_duration} timesteps ({rel_duration/total_relative_duration:.1%} of curriculum)")
                
                # Generate and display the distribution
                stage_params = dist.get('params', {})
                try:
                    # Try to generate a numeric representation of this distribution
                    grid, env_info = generate_numeric_distribution(
                        dist.get('type', 'uniform'),
                        stage_params.get('favor_near', True),
                        width,
                        height,
                        env_type,
                        env_id,
                        stage_params
                    )
                    
                    # Display detailed numeric distribution
                    print_numeric_distribution(
                        grid, 
                        env_info["goal_pos"],
                        env_info["lava_positions"],
                        env_info["wall_positions"],
                        title=f"Stage {i+1} Distribution"
                    )
                except Exception as e:
                    print(f"  Error generating distribution visualization: {e}")
            
            # After curriculum
            print("\nAfter curriculum:")
            print("  Distribution type: uniform")
            print(f"  Duration: {total_timesteps - curriculum_timesteps} timesteps ({1.0 - curriculum_prop:.1%} of total)")
        
        # Continuous transition info
        elif env_params.get("use_continuous_transition", False):
            transition_config = env_params.get("continuous_transition_config", {})
            
            print("Using continuous transition:")
            print(f"Initial distribution: {dist_type}")
            print(f"Target distribution: {transition_config.get('target_type', 'uniform')}")
            print(f"Transition rate: {transition_config.get('rate', 1.0)}")
            
            # Generate and display initial and target distributions
            try:
                # Initial distribution
                initial_grid, initial_info = generate_numeric_distribution(
                    dist_type,
                    favor_near,
                    width,
                    height,
                    env_type,
                    env_id,
                    params
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
                    transition_config.get("target_type", "uniform"),
                    not favor_near,
                    width,
                    height,
                    env_type,
                    env_id,
                    transition_config.get("target_params", {})
                )
                
                print_numeric_distribution(
                    target_grid, 
                    target_info["goal_pos"],
                    target_info["lava_positions"],
                    target_info["wall_positions"],
                    title="Target Distribution"
                )
                
                # Show timeline of transition progress
                rate = transition_config.get("rate", 1.0)
                print("\nTransition Timeline:")
                
                # Print timeline markers at 25%, 50%, 75%, 100%
                for progress in [0.25, 0.5, 0.75, 1.0]:
                    actual_progress = progress ** rate
                    timestep = int(total_timesteps * progress)
                    print(f"  {timestep} timesteps ({progress:.0%}): {actual_progress:.1%} transition progress")
                
            except Exception as e:
                print(f"Error generating distribution visualizations: {e}")
        
        # Fixed distribution info
        else:
            print(f"Using fixed distribution: {dist_type}")
            print(f"Parameters: {params}")
            
            # Generate and display the distribution
            try:
                grid, env_info = generate_numeric_distribution(
                    dist_type,
                    favor_near,
                    width,
                    height,
                    env_type,
                    env_id,
                    params
                )
                
                # Display detailed numeric distribution
                print_numeric_distribution(
                    grid, 
                    env_info["goal_pos"],
                    env_info["lava_positions"],
                    env_info["wall_positions"],
                    title="Spawn Distribution"
                )
            except Exception as e:
                print(f"Error generating distribution visualization: {e}")
        
    else:
        # Text-only fallback
        print("Detailed distribution visualization not available.")
        print(f"Distribution type: {dist_type}")
        print(f"Favor near goal: {favor_near}")
        
        # Stage info without visualization
        if env_params.get("use_stage_training", False):
            stage_config = env_params.get("stage_training_config", {})
            print(f"Using stage-based training with {stage_config.get('num_stages', 0)} stages")
        elif env_params.get("use_continuous_transition", False):
            transition_config = env_params.get("continuous_transition_config", {})
            print(f"Using continuous transition to {transition_config.get('target_type', 'uniform')}")
    
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


def generate_numeric_distribution(dist_type, favor_near=True, width=7, height=7, env_type="empty", env_id=None, params=None):
    """
    Generate a numeric probability distribution.
    
    Parameters:
    ----------
    dist_type : str
        Type of distribution (uniform, poisson_goal, gaussian_goal, distance_goal, gaussian_2d)
    favor_near : bool
        Whether to favor positions near the goal (for legacy distribution types)
    width, height : int
        Grid dimensions
    env_type : str
        Environment type ("empty", "lava_gap", "lava_crossing")
    env_id : str or None
        If provided, use this environment ID to determine the actual layout
    params : dict
        Distribution parameters (for newer distribution types like gaussian_2d)
    
    Returns:
    -------
    numpy.ndarray
        Probability distribution array
    dict
        Environment information (goal position, lava positions)
    """
    # Use empty dict if params is None
    params = params or {}
    
    # Initialize grid and environment information
    grid = np.zeros((height, width))
    goal_pos = (width-2, height-2)  # Typical goal position (bottom right)
    lava_positions = []
    wall_positions = []
    
    # Add walls for the border - all MiniGrid environments have walls around the border
    for x in range(width):
        for y in range(height):
            # Cells on the border are walls
            if x == 0 or x == width-1 or y == 0 or y == height-1:
                wall_positions.append((x, y))
    
    # If an env_id is provided, try to extract the actual layout
    if env_id:
        # Parse environment information from the ID
        import re
        
        # For LavaCrossing environments
        if "LavaCrossing" in env_id:
            # Extract size and number of crossings from environment ID
            # Example: MiniGrid-LavaCrossingS11N5-v0 -> Size 11, 5 crossings
            match = re.search(r"LavaCrossingS(\d+)N(\d+)", env_id)
            if match:
                size = int(match.group(1))
                num_crossings = int(match.group(2))
                
                # Check if size matches our width/height
                if size != width or size != height:
                    width = size
                    height = size
                    grid = np.zeros((height, width))
                    # Recalculate wall positions for new size
                    wall_positions = []
                    for x in range(width):
                        for y in range(height):
                            if x == 0 or x == width-1 or y == 0 or y == height-1:
                                wall_positions.append((x, y))
                
                # Different lava patterns based on environment
                if "LavaCrossingS9N1" in env_id:
                    # Single diagonal crossing
                    for i in range(1, width-1):
                        lava_positions.append((i, i))
                
                elif "LavaCrossingS11N5" in env_id:
                    # 5 vertical crossings at x=2,4,6,8 with row 2 filled
                    # Row 2 is filled
                    for x in range(2, width-1):
                        lava_positions.append((x, 2))
                    
                    # Vertical crossings
                    for x in [2, 4, 6, 8]:
                        if x < width - 1:
                            for y in range(1, height-1):
                                if y != 4 and (x != 8 or y != 7):  # Gaps in columns
                                    lava_positions.append((x, y))
                
                else:
                    # For other lava crossing environments, create a reasonable pattern
                    spacing = width // (num_crossings + 1)
                    for crossing_idx in range(num_crossings):
                        x = spacing * (crossing_idx + 1)
                        for y in range(1, height-1):
                            # Add some gaps in the crossings
                            if y % 3 != 0 or crossing_idx % 2 == 0:
                                lava_positions.append((x, y))
        
        # For LavaGap environments
        elif "LavaGap" in env_id:
            # Extract size from environment ID
            match = re.search(r"LavaGapS(\d+)", env_id)
            if match:
                size = int(match.group(1))
                
                # Check if size matches our width/height
                if size != width or size != height:
                    width = size
                    height = size
                    grid = np.zeros((height, width))
                    # Recalculate wall positions for new size
                    wall_positions = []
                    for x in range(width):
                        for y in range(height):
                            if x == 0 or x == width-1 or y == 0 or y == height-1:
                                wall_positions.append((x, y))
                
                # Create a horizontal lava strip with a gap in the middle
                mid_y = height // 2
                for x in range(1, width-1):
                    if x != width // 2:  # Gap in the middle
                        lava_positions.append((x, mid_y))
        
        # For Empty environments, just use the border walls
        elif "Empty" in env_id:
            match = re.search(r"Empty-(\d+)x(\d+)", env_id)
            if match:
                width = int(match.group(1))
                height = int(match.group(2))
                grid = np.zeros((height, width))
                # Recalculate wall positions for new size
                wall_positions = []
                for x in range(width):
                    for y in range(height):
                        if x == 0 or x == width-1 or y == 0 or y == height-1:
                            wall_positions.append((x, y))
    
    # If no environment ID or failed to parse, use the simpler env_type logic
    else:
        # Configure environment based on type
        if env_type == "lava_gap":
            # Create a horizontal lava strip with a gap in the middle
            mid_y = height // 2
            for x in range(width):
                if x != width // 2:  # Gap in the middle
                    lava_positions.append((x, mid_y))
        
        elif env_type == "lava_crossing":
            # Create multiple diagonal lines of lava cells based on size
            # For larger environments, add more lava crossings
            if width >= 11:  # For larger environments like 11x11
                # We're dealing with a larger environment with multiple lava crossings
                # In MiniGrid-LavaCrossingS11N5-v0, there are 5 lava lines
                num_crossings = min(5, width // 2)
                spacing = width // (num_crossings + 1)
                
                for crossing_idx in range(num_crossings):
                    # Calculate starting position for this crossing
                    start_x = spacing * (crossing_idx + 1)
                    
                    # Create a diagonal line from top to bottom
                    for i in range(height):
                        if 0 <= start_x < width and 0 <= i < height:
                            lava_positions.append((start_x, i))
            else:
                # For smaller environments, create a single diagonal line
                for i in range(min(width-2, height-2)):
                    lava_positions.append((i+1, i+1))
    
    # Generate initial distribution
    if dist_type == "uniform":
        grid.fill(1.0 / (width * height))
    
    elif dist_type == "gaussian_2d":
        center = params.get("center", [1.0, 1.0])  # Normalized center coordinates
        std = params.get("std", [0.2, 0.2])        # Normalized standard deviations
        directional = params.get("directional", False)
        angle = params.get("angle", 0)
        
        # Convert normalized coordinates to grid coordinates
        center_x = center[0] * (width - 1)
        center_y = center[1] * (height - 1)
        
        # Convert normalized standard deviations to grid units
        scale_factor = max(width, height)
        sigma_x = std[0] * scale_factor
        sigma_y = std[1] * scale_factor
        
        # Convert angle to radians if using directional Gaussian
        if directional:
            theta = np.radians(angle)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
        
        # Compute the Gaussian distribution
        for y in range(height):
            for x in range(width):
                if directional:
                    # Rotate coordinates for directional Gaussian
                    x_shifted = x - center_x
                    y_shifted = y - center_y
                    
                    # Rotate coordinates
                    x_rot = x_shifted * cos_theta + y_shifted * sin_theta
                    y_rot = -x_shifted * sin_theta + y_shifted * cos_theta
                    
                    # Normalized distances in the rotated coordinate system
                    x_term = (x_rot / sigma_x) ** 2
                    y_term = (y_rot / sigma_y) ** 2
                else:
                    # Standard Gaussian - use regular x,y distances
                    x_term = ((x - center_x) / sigma_x) ** 2
                    y_term = ((y - center_y) / sigma_y) ** 2
                
                # Calculate probability using the 2D Gaussian formula
                exponent = -0.5 * (x_term + y_term)
                prob = np.exp(exponent)
                grid[y, x] = prob
    
    else:
        # Get goal position
        goal_x, goal_y = goal_pos
        
        # Fill grid with appropriate distribution values
        for y in range(height):
            for x in range(width):
                # Calculate Manhattan distance from goal
                distance = abs(x - goal_x) + abs(y - goal_y)
                max_distance = abs(0 - goal_x) + abs(0 - goal_y)
                
                # Normalized distance [0, 1]
                norm_distance = distance / max_distance
                
                # Calculate density based on distribution type and favor_near
                if dist_type == "poisson_goal":
                    lambda_param = params.get("lambda_param", 1.0)
                    prob = np.exp(-lambda_param * distance)
                    
                    if not favor_near:
                        # Invert to favor positions far from goal
                        prob = np.exp(-lambda_param * (max_distance - distance))
                        
                elif dist_type == "gaussian_goal":
                    sigma = params.get("sigma", 2.0)
                    dist_squared = (x - goal_x)**2 + (y - goal_y)**2
                    prob = np.exp(-dist_squared / (2 * sigma**2))
                    
                    if not favor_near:
                        # Invert to favor positions far from goal
                        max_dist_squared = (width-1)**2 + (height-1)**2
                        inverse_dist_squared = max_dist_squared - dist_squared
                        prob = np.exp(-inverse_dist_squared / (2 * sigma**2))
                        
                elif dist_type == "distance_goal":
                    power = params.get("power", 2)
                    
                    if favor_near:
                        # Higher probability closer to goal
                        prob = 1 - (norm_distance ** power)
                    else:
                        # Higher probability away from goal
                        prob = norm_distance ** power
                        
                else:
                    # Default uniform distribution
                    prob = 1.0
                
                grid[y, x] = prob
    
    # Apply masks to zero out invalid cells
    
    # Zero out the goal position
    if 0 <= goal_pos[0] < width and 0 <= goal_pos[1] < height:
        grid[goal_pos[1], goal_pos[0]] = 0.0
    
    # Zero out lava positions
    for lx, ly in lava_positions:
        if 0 <= lx < width and 0 <= ly < height:
            grid[ly, lx] = 0.0
    
    # Zero out wall positions
    for wx, wy in wall_positions:
        if 0 <= wx < width and 0 <= wy < height:
            grid[wy, wx] = 0.0
    
    # Normalize to make it a proper probability distribution
    total = np.sum(grid)
    if total > 0:
        grid = grid / total
    
    # Create environment info dictionary
    env_info = {
        "goal_pos": goal_pos,
        "lava_positions": lava_positions,
        "wall_positions": wall_positions,
        "width": width,
        "height": height,
        "distribution_type": dist_type,
        "favor_near": favor_near,
        "env_type": env_type,
        "env_id": env_id
    }
    
    return grid, env_info

# Explicitly add the function to the module's namespace
__all__ = ["print_spawn_distribution_info", "generate_numeric_distribution"]