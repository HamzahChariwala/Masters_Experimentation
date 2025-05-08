#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
from minigrid.envs.empty import EmptyEnv
from SpawnDistributions.spawn_distributions import DistributionMap, FlexibleSpawnWrapper

# Create output directory for stage-based training visualization
output_dir = "stage_distribution_samples"
os.makedirs(output_dir, exist_ok=True)

def demonstrate_stage_based_training():
    """
    Create a demonstration of stage-based training with different spawn distributions
    that evolve as training progresses.
    """
    print("Demonstrating stage-based spawn distributions for curriculum learning...")
    
    # Create a simple MiniGrid environment
    env = EmptyEnv(size=10)
    
    # Define a 4-stage training curriculum that gradually moves spawns farther from goal
    # In MiniGrid, the goal is typically at the bottom-right corner (8,8) for a 10x10 grid
    stage_config = {
        "num_stages": 4,
        "distributions": [
            # Stage 1: Very close to goal - agent learns the basic task
            {
                "type": "poisson_goal",
                "params": {
                    "lambda_param": 1.0,
                    "favor_near": True
                }
            },
            # Stage 2: Medium distance from goal - agent learns to navigate simple paths
            {
                "type": "distance_goal",
                "params": {
                    "favor_near": True,
                    "power": 1
                }
            },
            # Stage 3: Farther from goal - agent learns longer trajectories 
            {
                "type": "gaussian_goal",
                "params": {
                    "sigma": 3.0,
                    "favor_near": False
                }
            },
            # Stage 4: Anywhere in the grid - agent masters the environment
            {
                "type": "uniform"
            }
        ]
    }
    
    # Set up the flexible spawn wrapper with stage-based training
    # Simulate 100,000 training steps total (25,000 per stage)
    total_timesteps = 100000
    spawn_env = FlexibleSpawnWrapper(
        env,
        distribution_type="uniform",  # This will be overridden by stages
        total_timesteps=total_timesteps,
        stage_based_training=stage_config
    )
    
    # We need to reset once to initialize the environment and distributions
    spawn_env.reset()
    
    # Manually advance the timestep to simulate training progress
    # We'll generate visualizations at specific points
    checkpoints = [0, 10000, 25000, 35000, 50000, 75000, 99999]
    
    for checkpoint in checkpoints:
        # Set the timestep to the checkpoint
        spawn_env.timestep = checkpoint
        
        # Simulate stage transitions by manually checking stage
        current_stage = min(checkpoint // (total_timesteps // stage_config["num_stages"]), 
                           stage_config["num_stages"] - 1)
                           
        # Update to correct stage
        if spawn_env.current_stage != current_stage:
            spawn_env.current_stage = current_stage
            
            # Apply the distribution for this stage
            stage_dist = stage_config["distributions"][current_stage]
            spawn_env._apply_distribution(stage_dist["type"], stage_dist.get("params", {}))
            spawn_env.current_distribution.mask_cells(spawn_env.valid_cells_mask)
            spawn_env.current_distribution.build_sampling_map()
        
        # Visualize the current distribution
        progress_pct = checkpoint / total_timesteps * 100
        stage_name = stage_config["distributions"][current_stage]["type"]
        title = f"Stage {current_stage+1}: {stage_name} ({progress_pct:.1f}% of training)"
        
        save_path = os.path.join(output_dir, f"stage_{current_stage+1}_progress_{int(progress_pct)}.png")
        spawn_env.visualize_distribution(title=title, save_path=save_path)
        
        print(f"Generated visualization at timestep {checkpoint} (Stage {current_stage+1}, {progress_pct:.1f}%)")
    
    print(f"\nStage-based training visualizations saved to {output_dir}/")
    print("This demonstrates how spawn positions evolve during curriculum learning:")
    print(" - Stage 1: Agent spawns very close to goal (learns basic rewards)")
    print(" - Stage 2: Agent spawns at medium distances (learns navigation)")
    print(" - Stage 3: Agent spawns far from goal (learns longer trajectories)")
    print(" - Stage 4: Agent spawns uniformly across environment (masters task)")

if __name__ == "__main__":
    demonstrate_stage_based_training() 