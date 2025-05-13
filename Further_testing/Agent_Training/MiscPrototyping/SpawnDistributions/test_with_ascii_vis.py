"""
Quick test script to show spawn distributions with visualization of actual values.
Runs a short training with numeric visualizations to show lava cells.
"""
import os
import random
import numpy as np
import torch
import time
import gymnasium as gym
from collections import defaultdict
import logging

# Import just the pieces we need, removing the dependency on observation_params
from main import Env, DQN, CustomCombinedExtractor

# Import our new distribution inspector
from SpawnDistributions.distribution_inspector import (
    visualize_env_spawn_distribution, 
    print_numeric_distribution,
    extract_distribution_from_env,
    generate_ascii_visualization
)

# Import the distribution creation functions from test_spawn_distributions
from SpawnDistributions.test_distributions_standalone import (
    apply_uniform_distribution,
    apply_poisson_distribution,
    apply_gaussian_distribution,
    apply_distance_distribution,
    create_empty_grid,
    create_lava_grid,
    print_ascii_visualization,
    plot_distribution
)

# Import everything from SpawnDistribution
from EnvironmentEdits.BespokeEdits.SpawnDistribution import FlexibleSpawnWrapper, EnhancedSpawnDistributionCallback, generate_final_visualizations

def debug_env_hierarchy(env, prefix=""):
    """
    Print the hierarchy of environment wrappers to help debugging.
    
    Parameters:
    ----------
    env : gym.Env
        Environment to inspect
    prefix : str
        Prefix for indentation
    """
    print(f"{prefix}Type: {type(env).__name__}")
    
    # Try to find the FlexibleSpawnWrapper
    try:
        if isinstance(env, FlexibleSpawnWrapper):
            print(f"{prefix}*** FOUND FlexibleSpawnWrapper ***")
            return True
    except ImportError:
        pass
    
    # Check common attributes
    found = False
    if hasattr(env, 'env'):
        print(f"{prefix}└── env:")
        found = debug_env_hierarchy(env.env, prefix + "    ") or found
    
    if hasattr(env, 'venv') and not found:
        print(f"{prefix}└── venv:")
        found = debug_env_hierarchy(env.venv, prefix + "    ") or found
    
    if hasattr(env, 'envs') and len(getattr(env, 'envs', [])) > 0 and not found:
        print(f"{prefix}└── envs[0]:")
        found = debug_env_hierarchy(env.envs[0], prefix + "    ") or found
    
    return found

if __name__ == "__main__":
    # Define test parameters directly instead of copying from main.py
    TEST_SPAWN_PARAMS = {
        "use_flexible_spawn": True,
        "spawn_distribution_type": "poisson_goal",
        "spawn_distribution_params": {"lambda_param": 1.0, "favor_near": True},
        "exclude_goal_adjacent": True,
        "window_size": 7,  # Required for environment creation
        "cnn_keys": [],  # Required for environment creation
        "mlp_keys": ["four_way_goal_direction", "four_way_angle_alignment", "barrier_mask", "lava_mask"],
    }
    
    # Set up a simple 2-stage training for a quick demo
    TEST_SPAWN_PARAMS["use_stage_training"] = True
    TEST_SPAWN_PARAMS["stage_training_config"] = {
        "num_stages": 2,
        "distributions": [
            # Stage 1: Near the goal (easy)
            {"type": "poisson_goal", "params": {"lambda_param": 1.0, "favor_near": True}},
            # Stage 2: Far from the goal (harder)
            {"type": "poisson_goal", "params": {"lambda_param": 1.0, "favor_near": False}}
        ]
    }
    
    # Set up visualization settings
    VIS_DIR = "./test_spawn_vis_output"
    os.makedirs(VIS_DIR, exist_ok=True)
    TEST_SPAWN_PARAMS["spawn_vis_dir"] = VIS_DIR
    TEST_SPAWN_PARAMS["spawn_vis_frequency"] = 1000  # Visualize more frequently for testing
    
    # Create demo environment with lava
    ENV_ID = 'MiniGrid-LavaGapS7-v0'  # Lava environment to show obstacle handling
    NUM_ENVS = 2  # Use fewer environments for a quicker test
    
    # Define standard seeds for reproducibility
    MODEL_SEED = 42
    ENV_SEED = 123
    
    # Set seeds
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    
    # Display the ASCII visualization for our distribution setup
    TOTAL_TIMESTEPS = 5_000  # Short test run
    print("\n==== SPAWN DISTRIBUTION TEST VISUALIZATION ====")
    print("Running a test with numeric visualization of spawn distributions")
    print(f"Training for {TOTAL_TIMESTEPS} steps on {ENV_ID} with {NUM_ENVS} environments")
    
    print("\n====== ENVIRONMENT CREATION ======")
    print(f"Base seed: {ENV_SEED}, Different envs: True")
    print(f"Flexible spawn: {TEST_SPAWN_PARAMS['use_flexible_spawn']}")
    print(f"Distribution type: {TEST_SPAWN_PARAMS['spawn_distribution_type']}")
    print(f"Favor near: {TEST_SPAWN_PARAMS['spawn_distribution_params'].get('favor_near', True)}")
    print("===============================\n")
    
    # Create the environment 
    env = Env.make_parallel_env(
        env_id=ENV_ID,
        num_envs=NUM_ENVS,
        env_seed=ENV_SEED,       # Only pass the environment seed
        use_different_envs=True,  # Enable diverse environments
        **TEST_SPAWN_PARAMS
    )
    
    # Debug the environment hierarchy just for information
    print("\n==== ENVIRONMENT HIERARCHY ====")
    single_env = env.envs[0] if hasattr(env, 'envs') and len(env.envs) > 0 else env
    wrapper_found = debug_env_hierarchy(single_env)
    print("===============================\n")
    
    print("\n==== DIRECT VISUALIZATION APPROACH ====")
    print("Generating distributions directly instead of extracting from environment...")
    
    # Set up grid dimensions
    width, height = 7, 7
    
    # Define goal position (typically bottom-right corner)
    goal_pos = (width-2, height-2)  # (5, 5) for a 7x7 grid
    
    # Create grid with lava gap
    print("\nCreating lava environment grid...")
    grid, lava_positions = create_lava_grid(width, height, "gap")
    
    # ===========================================
    # Generate and show various distribution types
    # ===========================================
    
    # Stage 1 - Near Goal
    print("\n==== STAGE 1: NEAR GOAL DISTRIBUTION ====")
    print("Distribution type: Poisson (near goal)")
    
    stage1_grid = apply_poisson_distribution(
        grid.copy(), goal_pos, lava_positions, 
        lambda_param=1.0, favor_near=True
    )
    
    # Show ASCII visualization
    print_ascii_visualization(
        stage1_grid, goal_pos, lava_positions, 
        title="Stage 1: Poisson Distribution (Near Goal)"
    )
    
    # Show numeric visualization
    print_numeric_distribution(
        stage1_grid, goal_pos, lava_positions,
        title="Stage 1: Poisson Distribution Values (Near Goal)"
    )
    
    # Save visualization
    stage1_filename = os.path.join(VIS_DIR, "stage1_poisson_near.png")
    plot_distribution(
        stage1_grid, goal_pos, lava_positions,
        title="Stage 1: Poisson Distribution (Near Goal)",
        filename=stage1_filename
    )
    
    # Stage 2 - Far from Goal
    print("\n==== STAGE 2: FAR FROM GOAL DISTRIBUTION ====")
    print("Distribution type: Poisson (far from goal)")
    
    stage2_grid = apply_poisson_distribution(
        grid.copy(), goal_pos, lava_positions, 
        lambda_param=1.0, favor_near=False
    )
    
    # Show ASCII visualization
    print_ascii_visualization(
        stage2_grid, goal_pos, lava_positions, 
        title="Stage 2: Poisson Distribution (Far from Goal)"
    )
    
    # Show numeric visualization
    print_numeric_distribution(
        stage2_grid, goal_pos, lava_positions,
        title="Stage 2: Poisson Distribution Values (Far from Goal)"
    )
    
    # Save visualization
    stage2_filename = os.path.join(VIS_DIR, "stage2_poisson_far.png")
    plot_distribution(
        stage2_grid, goal_pos, lava_positions,
        title="Stage 2: Poisson Distribution (Far from Goal)",
        filename=stage2_filename
    )
    
    # Additional distribution examples
    print("\n==== ADDITIONAL DISTRIBUTION EXAMPLES ====")
    
    # Uniform distribution
    print("\n=== UNIFORM DISTRIBUTION ===")
    uniform_grid = apply_uniform_distribution(grid.copy(), lava_positions, goal_pos)
    print_numeric_distribution(
        uniform_grid, goal_pos, lava_positions,
        title="Uniform Distribution"
    )
    
    uniform_filename = os.path.join(VIS_DIR, "uniform_distribution.png")
    plot_distribution(
        uniform_grid, goal_pos, lava_positions,
        title="Uniform Distribution",
        filename=uniform_filename
    )
    
    # Gaussian distribution (near goal)
    print("\n=== GAUSSIAN DISTRIBUTION (NEAR GOAL) ===")
    gaussian_grid = apply_gaussian_distribution(
        grid.copy(), goal_pos, lava_positions,
        sigma=2.0, favor_near=True
    )
    print_numeric_distribution(
        gaussian_grid, goal_pos, lava_positions,
        title="Gaussian Distribution (Near Goal)"
    )
    
    gaussian_near_filename = os.path.join(VIS_DIR, "gaussian_near_goal.png")
    plot_distribution(
        gaussian_grid, goal_pos, lava_positions,
        title="Gaussian Distribution (Near Goal)",
        filename=gaussian_near_filename
    )
    
    # Gaussian distribution (far from goal)
    print("\n=== GAUSSIAN DISTRIBUTION (FAR FROM GOAL) ===")
    gaussian_far_grid = apply_gaussian_distribution(
        grid.copy(), goal_pos, lava_positions,
        sigma=2.0, favor_near=False
    )
    print_numeric_distribution(
        gaussian_far_grid, goal_pos, lava_positions,
        title="Gaussian Distribution (Far from Goal)"
    )
    
    gaussian_far_filename = os.path.join(VIS_DIR, "gaussian_far_goal.png")
    plot_distribution(
        gaussian_far_grid, goal_pos, lava_positions,
        title="Gaussian Distribution (Far from Goal)",
        filename=gaussian_far_filename
    )
    
    # Distance-based distribution (near goal)
    print("\n=== DISTANCE-BASED DISTRIBUTION (NEAR GOAL) ===")
    distance_near_grid = apply_distance_distribution(
        grid.copy(), goal_pos, lava_positions,
        power=1.0, favor_near=True
    )
    print_numeric_distribution(
        distance_near_grid, goal_pos, lava_positions,
        title="Distance-based Distribution (Near Goal)"
    )
    
    distance_near_filename = os.path.join(VIS_DIR, "distance_near_goal.png")
    plot_distribution(
        distance_near_grid, goal_pos, lava_positions,
        title="Distance-based Distribution (Near Goal)",
        filename=distance_near_filename
    )
    
    # Distance-based distribution (far from goal)
    print("\n=== DISTANCE-BASED DISTRIBUTION (FAR FROM GOAL) ===")
    distance_far_grid = apply_distance_distribution(
        grid.copy(), goal_pos, lava_positions,
        power=1.0, favor_near=False
    )
    print_numeric_distribution(
        distance_far_grid, goal_pos, lava_positions,
        title="Distance-based Distribution (Far from Goal)"
    )
    
    distance_far_filename = os.path.join(VIS_DIR, "distance_far_goal.png")
    plot_distribution(
        distance_far_grid, goal_pos, lava_positions,
        title="Distance-based Distribution (Far from Goal)",
        filename=distance_far_filename
    )
    
    # ===========================================
    # Run a short training session
    # ===========================================
    
    # Create a minimal model for testing
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            cnn_num_layers=1,
            cnn_channels=[16],
            cnn_kernels=[3],
            cnn_strides=[1],
            cnn_paddings=[1],
            mlp_num_layers=1,
            mlp_hidden_sizes=[32],
        )
    )
    
    # Use CPU for testing
    device = torch.device("cpu")
    
    # Create a simpler model for quicker testing
    model = DQN(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        buffer_size=10_000,
        learning_starts=1_000,
        batch_size=32,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        gamma=0.6,
        learning_rate=1e-3,
        train_freq=4,
        target_update_interval=100,
        verbose=1,
        device=device
    )
    
    # Now run the actual training (without visualization callbacks)
    print("\n==== STARTING TRAINING (NO VISUALIZATION DURING TRAINING) ====")
    print("Training will run with no additional visualizations.")
    print("The visualizations above show how the distributions work at start:")
    print("- Lava cells have zero probability")
    print("- Goal position has zero probability")
    print("- Distribution changes according to stages or transition")
    print("==============================\n")
    
    # Train the model without visualization callbacks
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    print("\n==== TEST COMPLETE ====")
    print(f"All visualizations saved to {VIS_DIR}")
    print("The visualizations demonstrate how lava cells and the goal")
    print("have zero probability in all spawn distributions.")
    print("===========================\n") 