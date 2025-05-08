#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from minigrid.envs.empty import EmptyEnv
from SpawnDistributions.spawn_distributions import FlexibleSpawnWrapper, DistributionMap

# Create output directory
output_dir = "training_demo"
os.makedirs(output_dir, exist_ok=True)

def make_env(stage_based=False, continuous_transition=False, seed=0):
    """
    Create a MiniGrid environment with the FlexibleSpawnWrapper.
    
    Parameters:
    ----------
    stage_based : bool
        Whether to use stage-based training
    continuous_transition : bool
        Whether to use continuous transition between distributions
    seed : int
        Random seed
    """
    # Create the base environment with seed
    env = EmptyEnv(size=8, render_mode=None)
    
    # Set the seed for reproducibility
    env.reset(seed=seed)
    
    # Total training timesteps
    total_timesteps = 20000  # Small for demo purposes
    
    # Stage-based configuration
    if stage_based:
        # Four stages of training
        stage_config = {
            "num_stages": 4,
            "distributions": [
                # Stage 1: Very close to goal
                {"type": "poisson_goal", "params": {"lambda_param": 1.0, "favor_near": True}},
                # Stage 2: Medium distance
                {"type": "distance_goal", "params": {"favor_near": True, "power": 1}},
                # Stage 3: Far from goal
                {"type": "gaussian_goal", "params": {"sigma": 2.0, "favor_near": False}},
                # Stage 4: Uniform
                {"type": "uniform"}
            ]
        }
        
        # Apply the wrapper with stage-based configuration
        env = FlexibleSpawnWrapper(
            env,
            total_timesteps=total_timesteps,
            stage_based_training=stage_config
        )
        
    # Continuous transition configuration
    elif continuous_transition:
        # Start with spawning close to goal, gradually transition to uniform
        temporal_config = {
            "target_type": "uniform",
            "rate": 1.0  # Linear transition rate
        }
        
        # Apply the wrapper with continuous transition
        env = FlexibleSpawnWrapper(
            env,
            distribution_type="poisson_goal",
            distribution_params={"lambda_param": 1.0, "favor_near": True},
            total_timesteps=total_timesteps,
            temporal_transition=temporal_config
        )
        
    # Default configuration (fixed distribution)
    else:
        # Just use a fixed distribution - spawning far from goal
        env = FlexibleSpawnWrapper(
            env,
            distribution_type="distance_goal",
            distribution_params={"favor_near": False, "power": 1}
        )
    
    # Ensure the environment is initialized
    env.reset()
    
    return env

def run_demo():
    """Run the training integration demo."""
    print("=== SpawnDistributions Training Integration Demo ===")
    
    # Create a directory for visualization output
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Create environments with different configurations
    print("\nCreating demo environments...")
    stage_env = make_env(stage_based=True)
    continuous_env = make_env(continuous_transition=True)
    fixed_env = make_env()
    
    # 2. Visualize initial distributions
    print("\nVisualizing initial spawn distributions...")
    stage_env.visualize_distribution(
        title="Stage-Based Training: Initial Distribution",
        save_path=os.path.join(vis_dir, "stage_initial.png")
    )
    
    continuous_env.visualize_distribution(
        title="Continuous Transition: Initial Distribution",
        save_path=os.path.join(vis_dir, "continuous_initial.png")
    )
    
    fixed_env.visualize_distribution(
        title="Fixed Distribution",
        save_path=os.path.join(vis_dir, "fixed.png")
    )
    
    print("Initial distributions visualized to:", vis_dir)
    
    # Simulate stage-based training with manual distribution updates
    print("\nSimulating stage-based training progress...")
    timesteps = [0, 5000, 10000, 15000, 19999]
    stage_imgs = []
    
    for ts in timesteps:
        # Calculate current stage based on timestep
        stage = min(ts // 5000, 3)
        stage_name = stage_env.stage_based_training["distributions"][stage]["type"]
        
        # Manual simulation of stage changes by directly updating distribution
        # We avoid calling reset_timestep() since it might clear existing state
        
        # Update timestep counter
        stage_env.timestep = ts
        
        # Update stage if it changed
        if stage_env.current_stage != stage:
            stage_env.current_stage = stage
            stage_dist = stage_env.stage_based_training["distributions"][stage]
            stage_env._apply_distribution(stage_dist["type"], stage_dist.get("params", {}))
            stage_env.current_distribution.mask_cells(stage_env.valid_cells_mask)
            stage_env.current_distribution.build_sampling_map()
        
        # Save the visualization
        progress = ts / 20000 * 100
        img_path = os.path.join(vis_dir, f"stage_{stage+1}_progress_{int(progress)}.png")
        stage_env.visualize_distribution(
            title=f"Stage {stage+1}: {stage_name} ({progress:.1f}% of training)",
            save_path=img_path
        )
        stage_imgs.append(img_path)
        print(f"Generated visualization for stage {stage+1} at {progress:.1f}% of training")
    
    # Simulate continuous transition training
    print("\nSimulating continuous transition training progress...")
    
    # For continuous transition, we manually update the distribution
    continuous_env.distribution_history = []  # Clear history if it exists
    cont_imgs = []
    
    for ts in timesteps:
        # Update timestep
        continuous_env.timestep = ts
        
        # Calculate progress
        progress = ts / 20000
        
        # Create target distribution (uniform)
        target_dist = DistributionMap(continuous_env.current_distribution.width, 
                                     continuous_env.current_distribution.height)
        target_dist.uniform_distribution()
        
        # Interpolate between initial and target
        continuous_env.current_distribution.temporal_interpolation(target_dist, progress)
        continuous_env.current_distribution.build_sampling_map()
        
        # Record this state for history
        continuous_env.distribution_history.append((ts, continuous_env.current_distribution.probabilities.copy()))
        
        # Save visualization
        img_path = os.path.join(vis_dir, f"continuous_progress_{int(progress*100)}.png")
        continuous_env.visualize_distribution(
            title=f"Continuous Transition ({progress:.2f})",
            save_path=img_path
        )
        cont_imgs.append(img_path)
        print(f"Generated visualization for continuous transition at {progress:.2f} progress")
    
    # Create animated GIFs if matplotlib.animation is available
    try:
        import matplotlib.animation as animation
        from PIL import Image
        
        print("\nCreating animated visualizations...")
        
        # Create GIF for stage-based training
        images = [Image.open(img) for img in stage_imgs]
        images[0].save(
            os.path.join(vis_dir, "stage_animation.gif"),
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=700,
            loop=0
        )
        
        # Create GIF for continuous transition
        images = [Image.open(img) for img in cont_imgs]
        images[0].save(
            os.path.join(vis_dir, "continuous_animation.gif"),
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=700,
            loop=0
        )
        
        print("Animation GIFs created successfully")
    except (ImportError, Exception) as e:
        print(f"Could not create animations: {e}")
    
    print("\nDemo complete! Visualizations saved to:", vis_dir)

if __name__ == "__main__":
    run_demo() 