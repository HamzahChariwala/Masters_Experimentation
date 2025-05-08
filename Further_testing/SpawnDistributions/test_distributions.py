import os
import numpy as np
import matplotlib.pyplot as plt
from spawn_distributions import DistributionMap, FlexibleSpawnWrapper
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


def create_output_dir():
    """Create output directory for plots."""
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_distribution(dist_map, title, output_dir, filename):
    """Plot and save a distribution map."""
    save_path = os.path.join(output_dir, filename)
    dist_map.plot(title=title, show=False, save_path=save_path)
    print(f"Saved distribution plot to {save_path}")


def test_standalone_distributions():
    """Test distribution generation without an environment."""
    print("\n=== Testing Standalone Distributions ===")
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Grid dimensions
    width, height = 10, 10
    
    # Sample point (goal) position
    goal_pos = (7, 3)
    
    # 1. Uniform distribution
    print("Creating uniform distribution...")
    uniform_dist = DistributionMap(width, height).uniform_distribution()
    plot_distribution(uniform_dist, "Uniform Distribution", output_dir, "uniform_dist.png")
    
    # 2. Poisson distribution centered on goal
    print("Creating Poisson distribution (near goal)...")
    poisson_near = DistributionMap(width, height).poisson_from_point(goal_pos[0], goal_pos[1], lambda_param=0.5)
    plot_distribution(poisson_near, "Poisson Distribution (Near Goal, λ=0.5)", output_dir, "poisson_near_goal.png")
    
    # 3. Inverted Poisson distribution
    print("Creating Poisson distribution (far from goal)...")
    poisson_far = DistributionMap(width, height).poisson_from_point(goal_pos[0], goal_pos[1], lambda_param=0.5).invert()
    plot_distribution(poisson_far, "Poisson Distribution (Far from Goal, λ=0.5)", output_dir, "poisson_far_goal.png")
    
    # 4. Gaussian distribution
    print("Creating Gaussian distribution (near goal)...")
    gaussian_near = DistributionMap(width, height).gaussian_from_point(goal_pos[0], goal_pos[1], sigma=1.5)
    plot_distribution(gaussian_near, "Gaussian Distribution (Near Goal, σ=1.5)", output_dir, "gaussian_near_goal.png")
    
    # 5. Inverted Gaussian distribution
    print("Creating Gaussian distribution (far from goal)...")
    gaussian_far = DistributionMap(width, height).gaussian_from_point(goal_pos[0], goal_pos[1], sigma=1.5).invert()
    plot_distribution(gaussian_far, "Gaussian Distribution (Far from Goal, σ=1.5)", output_dir, "gaussian_far_goal.png")
    
    # 6. Distance-based distribution
    print("Creating distance-based distribution (near goal)...")
    distance_near = DistributionMap(width, height).distance_based_from_point(goal_pos[0], goal_pos[1], favor_near=True, power=2)
    plot_distribution(distance_near, "Distance-Based Distribution (Near Goal, Power=2)", output_dir, "distance_near_goal.png")
    
    # 7. Inverted distance-based distribution
    print("Creating distance-based distribution (far from goal)...")
    distance_far = DistributionMap(width, height).distance_based_from_point(goal_pos[0], goal_pos[1], favor_near=False, power=2)
    plot_distribution(distance_far, "Distance-Based Distribution (Far from Goal, Power=2)", output_dir, "distance_far_goal.png")
    
    # 8. Multi-point distribution
    print("Creating multi-point Poisson distribution...")
    points = [(2, 2), (7, 7), (3, 8)]
    multi_point = DistributionMap(width, height).multi_point_distribution(points, "poisson", {"lambda_param": 0.5})
    plot_distribution(multi_point, "Multi-Point Poisson Distribution (λ=0.5)", output_dir, "multi_point_poisson.png")
    
    # 9. Masked distribution
    print("Creating masked distribution...")
    # Create a mask with a corridor pattern
    mask = np.ones((height, width))
    for i in range(height):
        for j in range(width):
            if 3 <= i <= 6 and not (2 <= j <= 7):
                mask[i, j] = 0
    
    masked_dist = DistributionMap(width, height).uniform_distribution().mask_cells(mask)
    plot_distribution(masked_dist, "Masked Uniform Distribution", output_dir, "masked_uniform.png")
    
    # 10. Temporal transition between distributions
    print("Creating temporal transition sequence...")
    # Start with near goal distribution
    start_dist = DistributionMap(width, height).poisson_from_point(goal_pos[0], goal_pos[1], lambda_param=0.5)
    # End with far from goal distribution
    end_dist = DistributionMap(width, height).poisson_from_point(goal_pos[0], goal_pos[1], lambda_param=0.5).invert()
    
    # Create transition sequence
    for step in [0.0, 0.25, 0.5, 0.75, 1.0]:
        transition_dist = DistributionMap(width, height)
        transition_dist.from_existing_distribution(start_dist.probabilities)
        transition_dist.temporal_interpolation(end_dist, step)
        plot_distribution(transition_dist, f"Temporal Transition (Progress={step:.2f})", 
                        output_dir, f"temporal_transition_{int(step*100)}.png")
    
    print("\nAll standalone distribution tests completed.\n")


def visualize_env_grid(env, output_dir, filename="env_grid.png"):
    """Visualize the MiniGrid environment grid."""
    grid = env.unwrapped.grid
    width, height = grid.width, grid.height
    
    # Create a grid representation
    rgb_grid = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Color mapping
    colors = {
        'empty': [220, 220, 220],    # Light gray
        'wall': [120, 120, 120],     # Dark gray
        'goal': [0, 255, 0],         # Green
        'lava': [255, 0, 0],         # Red
        'agent': [0, 0, 255]         # Blue
    }
    
    # Fill the grid with the appropriate colors
    for i in range(width):
        for j in range(height):
            cell = grid.get(i, j)
            if cell is None:
                rgb_grid[j, i] = colors['empty']
            else:
                rgb_grid[j, i] = colors.get(cell.type, colors['empty'])
    
    # Mark agent position
    agent_pos = env.unwrapped.agent_pos
    rgb_grid[agent_pos[1], agent_pos[0]] = colors['agent']
    
    # Visualize the grid
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_grid)
    plt.title("Environment Grid")
    plt.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add legend
    patches = [mpatches.Patch(color=np.array(c)/255, label=k) for k, c in colors.items()]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the visualization
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def test_with_environment():
    """Test the FlexibleSpawnWrapper with an actual MiniGrid environment."""
    print("\n=== Testing FlexibleSpawnWrapper with MiniGrid Environment ===")
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Create the environment
    base_env = gym.make("MiniGrid-Empty-8x8-v0")
    
    # Visualize the base environment
    visualize_env_grid(base_env, output_dir, "base_env.png")
    print("Saved base environment visualization.")
    
    # Test different distribution types
    distributions = [
        # (dist_type, dist_params, title, filename)
        ("uniform", None, "Uniform Distribution", "env_uniform.png"),
        ("poisson_goal", {"lambda_param": 0.5, "favor_near": True}, 
         "Poisson Near Goal (λ=0.5)", "env_poisson_near.png"),
        ("poisson_goal", {"lambda_param": 0.5, "favor_near": False}, 
         "Poisson Far from Goal (λ=0.5)", "env_poisson_far.png"),
        ("gaussian_goal", {"sigma": 1.5, "favor_near": True}, 
         "Gaussian Near Goal (σ=1.5)", "env_gaussian_near.png"),
        ("distance_goal", {"favor_near": False, "power": 2}, 
         "Distance-Based Far from Goal", "env_distance_far.png")
    ]
    
    # Sample and plot the distributions
    for dist_type, dist_params, title, filename in distributions:
        # Create a new environment with the specified distribution
        env = gym.make("MiniGrid-Empty-8x8-v0")
        
        # Apply the flexible spawn wrapper
        wrapper = FlexibleSpawnWrapper(
            env, 
            distribution_type=dist_type,
            distribution_params=dist_params,
            exclude_occupied=True,
            exclude_goal_adjacent=False
        )
        
        # Reset to initialize the distribution
        wrapper.reset()
        
        # Visualize the distribution
        wrapper.visualize_distribution(title=title, save_path=os.path.join(output_dir, filename))
        print(f"Saved distribution visualization: {filename}")
        
        # Close the environment
        env.close()
    
    # Test temporal transition
    print("\nTesting temporal transition...")
    env = gym.make("MiniGrid-Empty-8x8-v0")
    
    # Configure temporal transition from near goal to far from goal
    temporal_wrapper = FlexibleSpawnWrapper(
        env,
        distribution_type="poisson_goal",
        distribution_params={"lambda_param": 0.5, "favor_near": True},
        total_timesteps=1000,
        temporal_transition={
            "target_type": "poisson_goal",
            "target_params": {"lambda_param": 0.5, "favor_near": False},
            "rate": 1.0
        }
    )
    
    # Reset to initialize
    temporal_wrapper.reset()
    
    # Visualize initial distribution
    temporal_wrapper.visualize_distribution(
        title="Initial Distribution (Near Goal)",
        save_path=os.path.join(output_dir, "temporal_initial.png")
    )
    
    # Simulate steps and visualize at different progress points
    steps = [250, 500, 750, 1000]
    for i in range(max(steps)):
        # Take a dummy action (0 = turn left)
        temporal_wrapper.step(0)
        
        if i+1 in steps:
            progress = (i+1) / 1000
            temporal_wrapper.visualize_distribution(
                title=f"Temporal Progress: {progress:.2f}",
                save_path=os.path.join(output_dir, f"temporal_progress_{int(progress*100)}.png")
            )
            print(f"Saved temporal progress visualization: {progress:.2f}")
    
    # Close the environment
    env.close()
    
    print("\nAll environment wrapper tests completed.\n")


def test_spawn_positions(num_samples=1000):
    """Test sampling from different distributions and plot the resulting positions."""
    print("\n=== Testing Spawn Position Sampling ===")
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Create the environment
    env = gym.make("MiniGrid-Empty-8x8-v0")
    
    # Distributions to test
    distributions = [
        ("uniform", None, "Uniform Distribution Spawns"),
        ("poisson_goal", {"lambda_param": 0.5, "favor_near": True}, "Near Goal Spawns"),
        ("poisson_goal", {"lambda_param": 0.5, "favor_near": False}, "Far from Goal Spawns"),
        ("gaussian_goal", {"sigma": 1.5, "favor_near": True}, "Gaussian Near Goal Spawns")
    ]
    
    for dist_type, dist_params, title in distributions:
        # Create wrapper
        wrapper = FlexibleSpawnWrapper(
            env,
            distribution_type=dist_type,
            distribution_params=dist_params
        )
        
        # Collect spawn positions
        spawn_positions = []
        for _ in range(num_samples):
            obs, _ = wrapper.reset()
            spawn_positions.append(wrapper.unwrapped.agent_pos)
        
        # Create a heatmap of spawn frequencies
        width, height = wrapper.unwrapped.grid.width, wrapper.unwrapped.grid.height
        spawn_heatmap = np.zeros((height, width))
        
        for x, y in spawn_positions:
            spawn_heatmap[y, x] += 1
        
        # Normalize
        if np.sum(spawn_heatmap) > 0:
            spawn_heatmap = spawn_heatmap / np.sum(spawn_heatmap)
        
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(spawn_heatmap, cmap='viridis')
        plt.colorbar(label='Spawn Frequency')
        plt.title(f"{title} (n={num_samples})")
        plt.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add goal position marker
        goal_pos = None
        for i in range(width):
            for j in range(height):
                cell = wrapper.unwrapped.grid.get(i, j)
                if cell and cell.type == 'goal':
                    goal_pos = (i, j)
                    break
            if goal_pos:
                break
                
        if goal_pos:
            plt.plot(goal_pos[0], goal_pos[1], 'rx', markersize=15, markeredgewidth=3, label='Goal')
            plt.legend()
        
        # Save the plot
        filename = f"spawn_heatmap_{dist_type}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated spawn heatmap for {dist_type}: {filename}")
    
    env.close()
    print("\nSpawn position testing completed.\n")


if __name__ == "__main__":
    # Create the output directory to save plots
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")
    
    # Run the tests
    test_standalone_distributions()
    test_with_environment()
    test_spawn_positions(num_samples=1000)
    
    print("All tests completed successfully.")