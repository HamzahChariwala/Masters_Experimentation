#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
from minigrid.envs.empty import EmptyEnv
from minigrid.envs.crossing import CrossingEnv  # Has lava
from SpawnDistributions.spawn_distributions import DistributionMap, FlexibleSpawnWrapper

# Create output directory
output_dir = "validation_results"
os.makedirs(output_dir, exist_ok=True)

def validate_goal_exclusion():
    """Test that the goal position is always excluded from spawn distributions."""
    print("\n=== Testing Goal Position Exclusion ===")
    
    # Create a simple environment with a goal
    env = EmptyEnv(size=8)
    
    # Test different distribution types
    distribution_types = [
        "uniform",
        "poisson_goal",
        "gaussian_goal",
        "distance_goal"
    ]
    
    for dist_type in distribution_types:
        print(f"\nTesting {dist_type} distribution...")
        
        # Create wrapper with this distribution
        if dist_type == "uniform":
            wrapped_env = FlexibleSpawnWrapper(env, distribution_type=dist_type)
        else:
            # For goal-based distributions, test both near and far preferences
            for favor_near in [True, False]:
                dist_params = {"favor_near": favor_near}
                if dist_type == "poisson_goal":
                    dist_params["lambda_param"] = 1.0
                elif dist_type == "gaussian_goal":
                    dist_params["sigma"] = 2.0
                
                print(f"  With favor_near={favor_near}")
                wrapped_env = FlexibleSpawnWrapper(
                    env, 
                    distribution_type=dist_type, 
                    distribution_params=dist_params
                )
                
                # Reset to initialize
                wrapped_env.reset()
                
                # Validate distribution
                report = wrapped_env.validate_distributions()
                if report["is_valid"]:
                    print(f"  ✅ Validation passed: Goal has zero probability")
                else:
                    print(f"  ❌ Validation failed:")
                    if not report["goal_zero_probability"]:
                        print(f"     - Goal position does not have zero probability")
                    for pos in report["invalid_spawn_positions"]:
                        print(f"     - {pos['type']} at {pos['position']} has probability {pos['probability']:.6f}")
                
                # Visualize distribution
                dir_suffix = "near_goal" if favor_near else "far_from_goal"
                wrapped_env.visualize_distribution(
                    title=f"{dist_type.capitalize()} Distribution ({'Near' if favor_near else 'Far from'} Goal)",
                    save_path=os.path.join(output_dir, f"{dist_type}_{dir_suffix}.png")
                )
        
        # If uniform distribution, test it separately
        if dist_type == "uniform":
            wrapped_env.reset()
            report = wrapped_env.validate_distributions()
            if report["is_valid"]:
                print(f"  ✅ Validation passed: Goal has zero probability")
            else:
                print(f"  ❌ Validation failed:")
                if not report["goal_zero_probability"]:
                    print(f"     - Goal position does not have zero probability")
                for pos in report["invalid_spawn_positions"]:
                    print(f"     - {pos['type']} at {pos['position']} has probability {pos['probability']:.6f}")
            
            wrapped_env.visualize_distribution(
                title="Uniform Distribution",
                save_path=os.path.join(output_dir, "uniform.png")
            )
    
    print("\nGoal position exclusion tests completed.")

def validate_lava_exclusion():
    """Test that lava cells are always excluded from spawn distributions."""
    print("\n=== Testing Lava Cell Exclusion ===")
    
    # Create a crossing environment with lava
    env = CrossingEnv(size=9, num_crossings=1)
    
    # Test with uniform distribution (simplest case)
    wrapped_env = FlexibleSpawnWrapper(env, distribution_type="uniform")
    wrapped_env.reset()
    
    # Validate distribution
    report = wrapped_env.validate_distributions()
    if report["is_valid"]:
        print("✅ Validation passed: All lava cells have zero probability")
    else:
        print("❌ Validation failed:")
        if not report["lava_zero_probability"]:
            print("  - One or more lava cells do not have zero probability")
        for pos in report["invalid_spawn_positions"]:
            print(f"  - {pos['type']} at {pos['position']} has probability {pos['probability']:.6f}")
    
    # Visualize distribution
    wrapped_env.visualize_distribution(
        title="Distribution with Lava Exclusion",
        save_path=os.path.join(output_dir, "lava_exclusion.png")
    )
    
    print("\nLava exclusion tests completed.")

def validate_temporal_transition():
    """Test that temporal transitions maintain zero probability for goal and lava."""
    print("\n=== Testing Temporal Transitions ===")
    
    # Create a crossing environment with lava
    env = CrossingEnv(size=9, num_crossings=1)
    
    # Set up a temporal transition from goal-centered to uniform
    temporal_config = {
        "target_type": "uniform",
        "rate": 1.0
    }
    
    wrapped_env = FlexibleSpawnWrapper(
        env,
        distribution_type="poisson_goal",
        distribution_params={"lambda_param": 1.0, "favor_near": True},
        total_timesteps=10000,
        temporal_transition=temporal_config
    )
    
    # Reset to initialize
    wrapped_env.reset()
    
    # Test at different points in the transition
    progress_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for progress in progress_points:
        # Set timestep 
        timestep = int(10000 * progress)
        wrapped_env.timestep = timestep
        
        # If this is not the initial state, we need to manually update the distribution
        if progress > 0:
            target_dist = DistributionMap(wrapped_env.current_distribution.width, 
                                          wrapped_env.current_distribution.height)
            target_dist.uniform_distribution()
            wrapped_env.current_distribution.temporal_interpolation(target_dist, progress)
            wrapped_env.current_distribution.build_sampling_map()
        
        # Validate distribution
        report = wrapped_env.validate_distributions()
        if report["is_valid"]:
            print(f"Progress {progress:.2f}: ✅ Validation passed")
        else:
            print(f"Progress {progress:.2f}: ❌ Validation failed:")
            if not report["goal_zero_probability"]:
                print("  - Goal position does not have zero probability")
            if not report["lava_zero_probability"]:
                print("  - One or more lava cells do not have zero probability")
        
        # Visualize distribution
        wrapped_env.visualize_distribution(
            title=f"Temporal Distribution (Progress: {progress:.2f})",
            save_path=os.path.join(output_dir, f"temporal_progress_{int(progress*100)}.png")
        )
    
    print("\nTemporal transition tests completed.")

def validate_stage_based_training():
    """Test that stage-based training maintains zero probability for goal and lava."""
    print("\n=== Testing Stage-Based Training ===")
    
    # Create a crossing environment with lava
    env = CrossingEnv(size=9, num_crossings=1)
    
    # Set up a 4-stage training config
    stage_config = {
        "num_stages": 4,
        "distributions": [
            {"type": "poisson_goal", "params": {"lambda_param": 1.0, "favor_near": True}},
            {"type": "distance_goal", "params": {"favor_near": True, "power": 1}},
            {"type": "gaussian_goal", "params": {"sigma": 2.0, "favor_near": False}},
            {"type": "uniform"}
        ]
    }
    
    # Set up the environment
    total_timesteps = 10000
    wrapped_env = FlexibleSpawnWrapper(
        env,
        total_timesteps=total_timesteps,
        stage_based_training=stage_config
    )
    
    # Reset to initialize
    wrapped_env.reset()
    
    # Test each stage
    for stage in range(4):
        # Calculate timestep for this stage
        timestep = int((stage + 0.5) * total_timesteps / 4)  # Middle of each stage
        wrapped_env.timestep = timestep
        
        # Update stage if needed
        if wrapped_env.current_stage != stage:
            wrapped_env.current_stage = stage
            stage_dist = stage_config["distributions"][stage]
            wrapped_env._apply_distribution(stage_dist["type"], stage_dist.get("params", {}))
            wrapped_env.current_distribution.mask_cells(wrapped_env.valid_cells_mask)
            wrapped_env.current_distribution.build_sampling_map()
        
        # Validate distribution
        report = wrapped_env.validate_distributions()
        if report["is_valid"]:
            print(f"Stage {stage+1}: ✅ Validation passed")
        else:
            print(f"Stage {stage+1}: ❌ Validation failed:")
            if not report["goal_zero_probability"]:
                print("  - Goal position does not have zero probability")
            if not report["lava_zero_probability"]:
                print("  - One or more lava cells do not have zero probability")
        
        # Visualize distribution
        dist_type = stage_config["distributions"][stage]["type"]
        wrapped_env.visualize_distribution(
            title=f"Stage {stage+1} Distribution: {dist_type}",
            save_path=os.path.join(output_dir, f"stage_{stage+1}.png")
        )
    
    print("\nStage-based training tests completed.")

def run_validation():
    """Run all validation tests."""
    print("Starting spawn distribution validation...")
    
    # Test goal exclusion
    validate_goal_exclusion()
    
    # Test lava exclusion
    validate_lava_exclusion()
    
    # Test temporal transitions
    validate_temporal_transition()
    
    # Test stage-based training
    validate_stage_based_training()
    
    print("\nValidation completed. Results saved to:", output_dir)

if __name__ == "__main__":
    run_validation() 