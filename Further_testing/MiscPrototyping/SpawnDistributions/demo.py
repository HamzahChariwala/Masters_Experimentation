import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from EnvironmentEdits.BespokeEdits.SpawnDistribution import FlexibleSpawnWrapper
import time
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper


def demo_spawn_distributions():
    """
    Demonstrate how to use the FlexibleSpawnWrapper with a MiniGrid environment.
    This shows different spawn distribution configurations, including temporal transitions.
    """
    print("\n=== Flexible Spawn Distribution Demo ===\n")
    
    # Create output directory for visualizations
    output_dir = os.path.join(os.path.dirname(__file__), "demo_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Uniform distribution (equivalent to original RandomSpawnWrapper)
    print("1. Uniform Distribution Example")
    print("   (Equivalent to original RandomSpawnWrapper)")
    
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    
    # Wrap with uniform distribution
    wrapper = FlexibleSpawnWrapper(
        env,
        distribution_type="uniform",
        exclude_occupied=True
    )
    
    # Reset to initialize
    obs, _ = wrapper.reset()
    
    # Visualize
    wrapper.visualize_distribution(
        title="Uniform Distribution",
        save_path=os.path.join(output_dir, "demo_uniform.png")
    )
    
    print("   Uniform distribution will spawn the agent with equal probability across all empty cells")
    print("   Saved visualization to demo_uniform.png\n")
    
    # Close the environment
    env.close()
    
    # 2. Spawn near goal example
    print("2. Near-Goal Spawn Distribution Example")
    
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    
    # Wrap with near goal distribution
    wrapper = FlexibleSpawnWrapper(
        env,
        distribution_type="poisson_goal",
        distribution_params={"lambda_param": 0.5, "favor_near": True},
        exclude_occupied=True,
        exclude_goal_adjacent=False  # Allow spawning adjacent to goal
    )
    
    # Reset to initialize
    obs, _ = wrapper.reset()
    
    # Visualize
    wrapper.visualize_distribution(
        title="Near Goal Distribution (λ=0.5)",
        save_path=os.path.join(output_dir, "demo_near_goal.png")
    )
    
    print("   This distribution favors spawn positions close to the goal")
    print("   Higher probability cells are closer to the goal")
    print("   Lambda parameter controls how quickly probability falls off with distance")
    print("   Saved visualization to demo_near_goal.png\n")
    
    # Close the environment
    env.close()
    
    # 3. Spawn far from goal example
    print("3. Far-from-Goal Spawn Distribution Example")
    
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    
    # Wrap with far from goal distribution
    wrapper = FlexibleSpawnWrapper(
        env,
        distribution_type="poisson_goal",
        distribution_params={"lambda_param": 0.5, "favor_near": False},
        exclude_occupied=True
    )
    
    # Reset to initialize
    obs, _ = wrapper.reset()
    
    # Visualize
    wrapper.visualize_distribution(
        title="Far from Goal Distribution (λ=0.5)",
        save_path=os.path.join(output_dir, "demo_far_goal.png")
    )
    
    print("   This distribution favors spawn positions far from the goal")
    print("   Created by inverting the near-goal distribution")
    print("   Saved visualization to demo_far_goal.png\n")
    
    # Close the environment
    env.close()
    
    # 4. Gaussian distribution example
    print("4. Gaussian Distribution Example")
    
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    
    # Wrap with gaussian distribution
    wrapper = FlexibleSpawnWrapper(
        env,
        distribution_type="gaussian_goal",
        distribution_params={"sigma": 1.2, "favor_near": True},
        exclude_occupied=True
    )
    
    # Reset to initialize
    obs, _ = wrapper.reset()
    
    # Visualize
    wrapper.visualize_distribution(
        title="Gaussian Distribution (σ=1.2)",
        save_path=os.path.join(output_dir, "demo_gaussian.png")
    )
    
    print("   Gaussian distribution centered on goal")
    print("   Sigma parameter controls the spread of the distribution")
    print("   Saved visualization to demo_gaussian.png\n")
    
    # Close the environment
    env.close()
    
    # 5. Temporal Transition Example
    print("5. Temporal Transition Example")
    print("   (Starts near goal, gradually transitions to far from goal)")
    
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    
    # Define temporal transition
    wrapper = FlexibleSpawnWrapper(
        env,
        distribution_type="poisson_goal",
        distribution_params={"lambda_param": 0.5, "favor_near": True},
        total_timesteps=1000,  # 1000 timesteps for the full transition
        temporal_transition={
            "target_type": "poisson_goal",  # Target distribution
            "target_params": {"lambda_param": 0.5, "favor_near": False},  # Target params
            "rate": 1.0  # Linear transition rate
        }
    )
    
    # Reset to initialize
    obs, _ = wrapper.reset()
    
    # Visualize initial state
    wrapper.visualize_distribution(
        title="Initial Distribution (t=0)",
        save_path=os.path.join(output_dir, "demo_temporal_0.png")
    )
    
    print("   Initial distribution saved to demo_temporal_0.png")
    
    # Progress points to visualize
    progress_points = [250, 500, 750, 1000]
    
    # Run simulation to show transition
    for i in range(max(progress_points)):
        # Take an action (0 = turn left)
        wrapper.step(0)
        
        # Visualize at specified progress points
        if i+1 in progress_points:
            progress = (i+1) / 1000
            wrapper.visualize_distribution(
                title=f"Distribution at t={i+1} (progress={progress:.2f})",
                save_path=os.path.join(output_dir, f"demo_temporal_{i+1}.png")
            )
            print(f"   Distribution at progress {progress:.2f} saved to demo_temporal_{i+1}.png")
    
    print("\n   As training progresses, spawn points gradually shift from near goal to far from goal")
    print("   This creates a curriculum that starts with easier tasks and gets more difficult\n")
    
    # Close the environment
    env.close()
    
    # 6. Integration Example with MiniGrid
    print("6. Integration Example with Agent Training Loop")
    print("   (Simulating a simplified training loop)")
    
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    
    # Apply RGB observation wrapper for visualization
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    
    # Add flexible spawn wrapper with temporal transition
    wrapper = FlexibleSpawnWrapper(
        env,
        distribution_type="poisson_goal",
        distribution_params={"lambda_param": 0.5, "favor_near": True},
        total_timesteps=100,  # Short for the demo
        temporal_transition={
            "target_type": "poisson_goal",
            "target_params": {"lambda_param": 0.5, "favor_near": False},
            "rate": 1.0
        }
    )
    
    # Simplified training loop
    print("   Running simplified training loop...")
    max_episodes = 5
    max_steps = 25
    
    for episode in range(max_episodes):
        obs, _ = wrapper.reset()
        
        # Save initial observation for this episode
        img = env.render()
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"Episode {episode+1} Initial State")
        plt.savefig(os.path.join(output_dir, f"demo_episode_{episode+1}_start.png"))
        plt.close()
        
        # Show distribution
        if episode == 0 or episode == max_episodes - 1:
            wrapper.visualize_distribution(
                title=f"Spawn Distribution (Episode {episode+1})",
                save_path=os.path.join(output_dir, f"demo_episode_{episode+1}_dist.png")
            )
        
        # Run the episode
        for step in range(max_steps):
            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapper.step(action)
            
            if terminated or truncated:
                break
        
        # Show progress
        progress = (episode + 1) / max_episodes
        print(f"   Episode {episode+1}/{max_episodes} completed (progress: {progress:.2f})")
    
    print("\n   Completed simplified training loop demonstration")
    print("   Images saved to demo_outputs directory\n")
    
    # Close the environment
    env.close()
    
    print("=== Demonstration Completed ===")
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    demo_spawn_distributions()