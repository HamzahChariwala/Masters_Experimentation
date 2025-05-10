#!/usr/bin/env python
import os
import random
import numpy as np
import torch
from typing import Tuple
import time

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import MultiDiscrete

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgObsWrapper, OneHotPartialObsWrapper, NoDeath, DirectionObsWrapper

from stable_baselines3 import PPO, DQN, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from AgentTraining.TerminalCondition import CustomTerminationCallback
from AgentTraining.Tooling import visualize_agent_behavior, evaluate_with_timeout

# ---------------------------------------------------------------

from EnvironmentEdits.BespokeEdits.CustomWrappers import (GoalAngleDistanceWrapper, 
                                             PartialObsWrapper, 
                                             ExtractAbstractGrid, 
                                             PartialRGBObsWrapper, 
                                             PartialGrayObsWrapper, 
                                             ForceFloat32)
from EnvironmentEdits.BespokeEdits.FeatureExtractor import CustomCombinedExtractor, SelectiveObservationWrapper
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.BespokeEdits.GymCompatibility import OldGymCompatibility

# Import environment generation functions
import EnvironmentEdits.EnvironmentGeneration as Env

# Import our flexible spawn distribution wrapper
from EnvironmentEdits.BespokeEdits.SpawnDistribution import FlexibleSpawnWrapper

# Import spawn configuration
from spawn_config import *

# Import visualization tools for spawn distributions
from SpawnDistributions.visualization import SpawnDistributionCallback, generate_final_visualizations

# ---------------------------------------------------------------

torch.set_default_dtype(torch.float32)


if __name__ == "__main__":

    log_dir = "./logs/dqn_spawn_config"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create directory for performance tracking
    performance_log_dir = os.path.join(log_dir, "performance")
    os.makedirs(performance_log_dir, exist_ok=True)

    ENV_ID = 'MiniGrid-Empty-8x8-v0'
    # ENV_ID = 'MiniGrid-LavaCrossingS9N1-v0'
    NUM_ENVS = 15  # Increased number of environments for better diversity
    
    # Clearly separated seeds for different sources of randomness
    MODEL_SEED = 811     # For network initialization
    ENV_SEED = 12345    # For environment generation
    EVAL_SEED = 67890   # For evaluation environments
    
    # Evaluation parameters
    EVAL_TIMEOUT = 15   # Timeout in seconds for each evaluation episode
    
    # NoDeath wrapper parameters
    USE_NO_DEATH = True              # Whether to use the NoDeath wrapper
    NO_DEATH_TYPES = ("lava",)       # Which elements should not cause death
    DEATH_COST = 0                   # Penalty for hitting death elements

    # Environment step limit
    MAX_EPISODE_STEPS = 150         # Maximum steps per episode (None = use env default)
    
    # Diagonal move parameters
    DIAGONAL_SUCCESS_REWARD = 0.01   # Reward for successful diagonal movesx
    DIAGONAL_FAILURE_PENALTY = 0  # Penalty for failed diagonal moves
    
    # Diagonal movement monitoring
    MONITOR_DIAGONAL_MOVES = True   # Track diagonal move usage

    # Process spawn configuration from spawn_config.py
    # Determine which type of distribution to use based on config
    if USE_FLEXIBLE_SPAWN:
        SPAWN_DISTRIBUTION_VIS_DIR = os.path.join(log_dir, "spawn_distributions")
        os.makedirs(SPAWN_DISTRIBUTION_VIS_DIR, exist_ok=True)
        
        # Setup based on the selected approach
        if USE_STAGE_BASED_TRAINING:
            # Use stage-based training
            USE_STAGE_TRAINING = True
            USE_CONTINUOUS_TRANSITION = False
            SPAWN_DISTRIBUTION_TYPE = "uniform"  # Will be overridden by stages
            SPAWN_DISTRIBUTION_PARAMS = {}
        elif USE_CONTINUOUS_TRANSITION:
            # Use continuous transition
            USE_STAGE_TRAINING = False
            USE_CONTINUOUS_TRANSITION = True
            SPAWN_DISTRIBUTION_TYPE = CONTINUOUS_TRANSITION_CONFIG.get("initial_type", "poisson_goal") 
            SPAWN_DISTRIBUTION_PARAMS = CONTINUOUS_TRANSITION_CONFIG.get("initial_params", {"lambda_param": 1.0, "favor_near": True})
            # Reformat the continuous transition config 
            CONTINUOUS_TRANSITION_CONFIG = {
                "target_type": CONTINUOUS_TRANSITION_CONFIG.get("target_type", "uniform"),
                "target_params": CONTINUOUS_TRANSITION_CONFIG.get("target_params", {}),
                "rate": CONTINUOUS_TRANSITION_CONFIG.get("rate", 1.0)
            }
        else:
            # Use fixed distribution
            USE_STAGE_TRAINING = False
            USE_CONTINUOUS_TRANSITION = False
            SPAWN_DISTRIBUTION_TYPE = FIXED_DISTRIBUTION.get("type", "uniform")
            SPAWN_DISTRIBUTION_PARAMS = FIXED_DISTRIBUTION.get("params", {})
    else:
        # Fall back to random spawn if flexible spawn is disabled
        USE_RANDOM_SPAWN = True
        USE_STAGE_TRAINING = False
        USE_CONTINUOUS_TRANSITION = False
        SPAWN_DISTRIBUTION_TYPE = "uniform"
        SPAWN_DISTRIBUTION_PARAMS = {}

    # Define standard observation parameters
    observation_params = {
        "window_size": 7,
        "cnn_keys": [],
        "mlp_keys": ["four_way_goal_direction",
                    "four_way_angle_alignment",
                    "barrier_mask",
                    "lava_mask"],
        "use_random_spawn": not USE_FLEXIBLE_SPAWN,  # Use random spawn only if flexible spawn is disabled
        "exclude_goal_adjacent": EXCLUDE_GOAL_ADJACENT,  # Don't spawn next to goal
        "use_no_death": USE_NO_DEATH,       # Whether to use the NoDeath wrapper
        "no_death_types": NO_DEATH_TYPES,   # Types that don't cause death
        "death_cost": DEATH_COST,           # Penalty for hitting death elements
        "max_episode_steps": MAX_EPISODE_STEPS,  # Maximum steps per episode
        "monitor_diagonal_moves": MONITOR_DIAGONAL_MOVES,  # Track diagonal move usage
        "diagonal_success_reward": DIAGONAL_SUCCESS_REWARD,  # Reward for successful diagonal moves
        "diagonal_failure_penalty": DIAGONAL_FAILURE_PENALTY,  # Penalty for failed diagonal moves
        "use_flexible_spawn": USE_FLEXIBLE_SPAWN,  # Whether to use our flexible spawn wrapper
        "spawn_distribution_type": SPAWN_DISTRIBUTION_TYPE,  # Type of distribution to use for flexible spawn
        "spawn_distribution_params": SPAWN_DISTRIBUTION_PARAMS,  # Parameters for the distribution
        "use_stage_training": USE_STAGE_TRAINING,  # Whether to use stage-based training
        "stage_training_config": STAGE_TRAINING_CONFIG if USE_STAGE_TRAINING else None,  # Configuration for stage-based training
        "use_continuous_transition": USE_CONTINUOUS_TRANSITION,  # Whether to use continuous transition
        "continuous_transition_config": CONTINUOUS_TRANSITION_CONFIG if USE_CONTINUOUS_TRANSITION else None,  # Configuration for continuous transition
        "spawn_vis_dir": SPAWN_DISTRIBUTION_VIS_DIR if USE_FLEXIBLE_SPAWN else None,  # Directory for spawn distribution visualizations
        "spawn_vis_frequency": VISUALIZE_FREQUENCY if VISUALIZE_SPAWN_DISTRIBUTIONS else 0  # Frequency of spawn distribution visualizations
    }

    # Set seeds for reproducible model training
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)

    # Print training setup information
    print("\n====== TRAINING SETUP ======")
    print(f"Environment: {ENV_ID}")
    print(f"Number of parallel environments: {NUM_ENVS}")
    print(f"Model seed: {MODEL_SEED}")
    print(f"Environment seed: {ENV_SEED}")
    print(f"Evaluation seed: {EVAL_SEED}")
    print(f"Evaluation timeout: {EVAL_TIMEOUT} seconds")
    print(f"Max episode steps: {MAX_EPISODE_STEPS}")
    print(f"Diagonal success reward: {DIAGONAL_SUCCESS_REWARD}")
    print(f"Diagonal failure penalty: {DIAGONAL_FAILURE_PENALTY}")
    
    if USE_NO_DEATH:
        print(f"NoDeath wrapper: {USE_NO_DEATH}")
        print(f"  - Death types: {NO_DEATH_TYPES}")
        print(f"  - Death cost: {DEATH_COST}")
    
    # Print spawn distribution information
    if USE_FLEXIBLE_SPAWN:
        print("\n=== Spawn Distribution Settings (from spawn_config.py) ===")
        if USE_STAGE_BASED_TRAINING:
            print("Using stage-based curriculum learning")
            print(f"Number of stages: {STAGE_TRAINING_CONFIG['num_stages']}")
            for i, dist in enumerate(STAGE_TRAINING_CONFIG['distributions']):
                near_far = ""
                if "params" in dist and "favor_near" in dist["params"]:
                    near_far = "near goal" if dist["params"]["favor_near"] else "far from goal"
                
                print(f"  Stage {i+1}: {dist['type']} {near_far}")
        elif USE_CONTINUOUS_TRANSITION:
            print("Using continuous curriculum learning")
            print(f"Initial distribution: {SPAWN_DISTRIBUTION_TYPE}")
            if "favor_near" in SPAWN_DISTRIBUTION_PARAMS:
                near_far = "near goal" if SPAWN_DISTRIBUTION_PARAMS["favor_near"] else "far from goal"
                print(f"  - {near_far}")
            
            print(f"Target distribution: {CONTINUOUS_TRANSITION_CONFIG['target_type']}")
            print(f"Transition rate: {CONTINUOUS_TRANSITION_CONFIG['rate']}")
        else:
            print(f"Using fixed spawn distribution: {SPAWN_DISTRIBUTION_TYPE}")
            if "favor_near" in SPAWN_DISTRIBUTION_PARAMS:
                near_far = "near goal" if SPAWN_DISTRIBUTION_PARAMS["favor_near"] else "far from goal"
                print(f"  - {near_far}")
        
        print(f"Exclude goal-adjacent spawns: {EXCLUDE_GOAL_ADJACENT}")
        print(f"Visualization frequency: Every {VISUALIZE_FREQUENCY} steps")
    else:
        print("\nUsing default random spawn (flexible spawn disabled in spawn_config.py)")
    
    print("===========================\n")

    # Option 1: Use the enhanced make_parallel_env with different environments
    env = Env.make_parallel_env(
        env_id=ENV_ID,
        num_envs=NUM_ENVS,
        env_seed=ENV_SEED,       # Only pass the environment seed
        use_different_envs=True,  # Enable diverse environments
        total_timesteps=100_000,  # For calculating stage transitions
        **observation_params
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            cnn_num_layers=1,
            cnn_channels=[32],
            cnn_kernels=[3],
            cnn_strides=[1],
            cnn_paddings=[1],
            mlp_num_layers=1,
            mlp_hidden_sizes=[64],
        )
    )

    use_mps = False  # Set to False to disable MPS

    device = torch.device("mps" if use_mps and torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model with improved exploration parameters
    model = DQN(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        buffer_size=100_000,             # Increased buffer size for more diverse experiences
        learning_starts=10_000,          # Allow more random exploration before learning starts
        batch_size=64,                 # Larger batch size for more stable updates
        exploration_fraction=0.05,      # Higher exploration rate to discover diagonal moves
        exploration_final_eps=0.05,    # Higher final exploration to keep trying diagonal moves
        gamma=0.6,                    # Slightly reduced discount factor to focus more on immediate rewards
        learning_rate=2.5e-4,            # Adjusted learning rate
        train_freq=8,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=log_dir,
        device=device  # Specify the device here
    )

    # Create a modified version of observation_params with flexible spawn disabled for evaluation
    eval_params = observation_params.copy()
    eval_params["use_flexible_spawn"] = False  # Disable flexible spawn for evaluation
    eval_params["use_random_spawn"] = False  # Also disable random spawn for evaluation (fixed start pos)
    # Keep diagonal move monitoring during evaluation
    eval_params["monitor_diagonal_moves"] = True

    # Create multiple evaluation environments with different seeds
    NUM_EVAL_ENVS = 5  # Number of different evaluation environments
    eval_envs = []
    
    print("\n====== CREATING EVALUATION ENVIRONMENTS ======")
    print(f"Creating {NUM_EVAL_ENVS} evaluation environments with different seeds")
    
    for i in range(NUM_EVAL_ENVS):
        # Use different seeds for each evaluation environment
        eval_seed = EVAL_SEED + (i * 1000)  # Well-spaced seeds
        eval_env = Env.make_eval_env(
            env_id=ENV_ID, 
            seed=eval_seed,
            **eval_params
        )
        eval_envs.append(eval_env)
    
    print(f"Created {len(eval_envs)} evaluation environments")
    print("=============================================\n")

    # Create termination callback with improved evaluation settings and multiple environments
    termination_callback = CustomTerminationCallback(
        eval_envs=eval_envs,  # Pass list of environments instead of single environment
        check_freq=10_000,                # Check every 10k steps
        target_reward_threshold=0.99,    # Target reward to achieve
        max_runtime=30000,               # Maximum runtime in seconds (about 8 hours)
        n_eval_episodes=1,               # Only evaluate on a single episode per environment
        eval_timeout=EVAL_TIMEOUT,       # Use the configurable timeout parameter
        log_dir=performance_log_dir,     # Directory for performance tracking
        verbose=1
    )

    # Create the spawn distribution visualization callback
    spawn_vis_callback = SpawnDistributionCallback(
        vis_dir=SPAWN_DISTRIBUTION_VIS_DIR if USE_FLEXIBLE_SPAWN else None,
        vis_frequency=VISUALIZE_FREQUENCY if USE_FLEXIBLE_SPAWN and VISUALIZE_SPAWN_DISTRIBUTIONS else 0
    )

    # Combine callbacks
    callbacks = [termination_callback]
    if USE_FLEXIBLE_SPAWN and VISUALIZE_SPAWN_DISTRIBUTIONS:
        callbacks.append(spawn_vis_callback)

    # Print training start message
    print("\n====== STARTING TRAINING ======")
    print(f"Target timesteps: 100,000")
    print(f"Evaluation frequency: Every {termination_callback.check_freq} steps")
    print(f"Evaluation environments: {NUM_EVAL_ENVS}")
    print(f"Evaluation episodes per environment: {termination_callback.n_eval_episodes}")
    print(f"Total evaluation episodes: {NUM_EVAL_ENVS * termination_callback.n_eval_episodes}")
    print(f"Evaluation timeout: {EVAL_TIMEOUT} seconds")
    print("==============================\n")

    model.learn(
        total_timesteps=100_000, 
        tb_log_name="DQN_MiniGrid",
        callback=callbacks
    )
    model.save(os.path.join(log_dir, "trained_model"))

    # Print final evaluation message
    print("\n====== TRAINING COMPLETE ======")
    print("Running final evaluation...")
    print("==============================\n")

    # Final evaluation across multiple environments
    NUM_FINAL_EVAL_ENVS = 5  # More environments for final evaluation
    EPISODES_PER_ENV = 3      # Multiple episodes per environment
    FINAL_EVAL_TIMEOUT = 3   # Timeout in seconds per environment evaluation
    
    print(f"Evaluating on {NUM_FINAL_EVAL_ENVS} environments, {EPISODES_PER_ENV} episodes each")
    print(f"Timeout per environment: {FINAL_EVAL_TIMEOUT} seconds")
    
    all_rewards = []
    all_lengths = []
    
    for i in range(NUM_FINAL_EVAL_ENVS):
        # Create a fresh evaluation environment with a different seed
        final_eval_seed = EVAL_SEED + 2000 + (i * 1000)  # Different seeds than those used during training
        final_eval_env = Env.make_eval_env(
            env_id=ENV_ID, 
            seed=final_eval_seed,
            **eval_params
        )
        
        # Use our timeout evaluation function
        env_rewards, env_lengths, error = evaluate_with_timeout(
            model=model,
            env=final_eval_env,
            n_eval_episodes=EPISODES_PER_ENV,
            timeout=FINAL_EVAL_TIMEOUT,
            deterministic=True
        )
        
        # Handle evaluation result
        if error:
            print(f"Environment {i+1}/{NUM_FINAL_EVAL_ENVS} (Seed: {final_eval_seed}): {error}")
            final_eval_env.close()
            continue
            
        # Store results
        all_rewards.extend(env_rewards)
        all_lengths.extend(env_lengths)
        
        # Print per-environment results
        print(f"Environment {i+1}/{NUM_FINAL_EVAL_ENVS} (Seed: {final_eval_seed}):")
        print("  Rewards   : mean %.3f  std %.3f  min %.3f  max %.3f"
            % (np.mean(env_rewards),
                np.std(env_rewards),
                np.min(env_rewards),
                np.max(env_rewards)))
        print("  Lengths   : mean %.1f  std %.1f  min %d    max %d"
            % (np.mean(env_lengths),
                np.std(env_lengths),
                np.min(env_lengths),
                np.max(env_lengths)))
        
        # Close the environment
        final_eval_env.close()
        
    # Only print overall results if we have any
    if len(all_rewards) > 0:
        print("\n====== FINAL RESULTS (All Environments) ======")
        print(f"Total episodes evaluated: {len(all_rewards)}")
        print("Rewards   : mean %.3f  std %.3f  min %.3f  max %.3f"
            % (np.mean(all_rewards),
                np.std(all_rewards),
                np.min(all_rewards),
                np.max(all_rewards)))
        print("Lengths   : mean %.1f  std %.1f  min %d    max %d"
            % (np.mean(all_lengths),
                np.std(all_lengths),
                np.min(all_lengths),
                np.max(all_lengths)))
    else:
        print("\n====== FINAL RESULTS ======")
        print("No valid evaluation results obtained")
    print("==========================\n")
    
    # Display final performance comparison if available
    if hasattr(termination_callback, "performance_tracker"):
        print("\n====== TRAINING VS EVALUATION PERFORMANCE ======")
        print("Generating final performance comparison plots...")
        # Generate final plots with the show option set to False (we're in headless mode)
        termination_callback.performance_tracker.plot_performance(save=True, show=False)
        print(f"Performance plots saved to: {performance_log_dir}")
        print("===============================================\n")

    # Generate training progression visualizations
    if USE_FLEXIBLE_SPAWN:
        generate_final_visualizations(
            model=model,
            spawn_vis_dir=SPAWN_DISTRIBUTION_VIS_DIR,
            use_stage_based_training=USE_STAGE_BASED_TRAINING
        )

    # Visualize the agent's behavior
    print("\n===== DETAILED AGENT BEHAVIOR ANALYSIS =====")
    visualize_agent_behavior(
        model=model,
        env_id=ENV_ID,
        observation_params=eval_params,  # Use evaluation parameters (no flexible spawn)
        seed=EVAL_SEED + 5000,  # Use a different seed than training/evaluation
        num_episodes=3,
        render=False  # Set to True if you have a display available
    ) 