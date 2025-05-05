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
from AgentTraining.Visualization import visualize_agent_behavior

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

# ---------------------------------------------------------------

torch.set_default_dtype(torch.float32)


if __name__ == "__main__":

    log_dir = "./logs/dqn_run_2"
    os.makedirs(log_dir, exist_ok=True)

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
    DIAGONAL_SUCCESS_REWARD = 0.25   # Reward for successful diagonal moves
    DIAGONAL_FAILURE_PENALTY = 0.1  # Penalty for failed diagonal moves
    
    # Diagonal movement monitoring
    MONITOR_DIAGONAL_MOVES = True   # Track diagonal move usage

    # Define standard observation parameters
    observation_params = {
        "window_size": 7,
        "cnn_keys": [],
        "mlp_keys": ["four_way_goal_direction",
                    "four_way_angle_alignment",
                    "barrier_mask",
                    "lava_mask"],
        "use_random_spawn": True,            # Enable random agent spawn positions
        "exclude_goal_adjacent": False,      # Don't spawn next to goal
        "use_no_death": USE_NO_DEATH,        # Whether to use the NoDeath wrapper
        "no_death_types": NO_DEATH_TYPES,    # Types that don't cause death
        "death_cost": DEATH_COST,            # Penalty for hitting death elements
        "max_episode_steps": MAX_EPISODE_STEPS,  # Maximum steps per episode
        "monitor_diagonal_moves": MONITOR_DIAGONAL_MOVES,  # Track diagonal move usage
        "diagonal_success_reward": DIAGONAL_SUCCESS_REWARD,  # Reward for successful diagonal moves
        "diagonal_failure_penalty": DIAGONAL_FAILURE_PENALTY  # Penalty for failed diagonal moves
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
    print(f"Random agent spawning: {observation_params.get('use_random_spawn', False)}")
    print(f"Evaluation timeout: {EVAL_TIMEOUT} seconds")
    print(f"Max episode steps: {MAX_EPISODE_STEPS}")
    print(f"Diagonal success reward: {DIAGONAL_SUCCESS_REWARD}")
    print(f"Diagonal failure penalty: {DIAGONAL_FAILURE_PENALTY}")
    print(f"NoDeath wrapper: {USE_NO_DEATH}")
    if USE_NO_DEATH:
        print(f"  - Death types: {NO_DEATH_TYPES}")
        print(f"  - Death cost: {DEATH_COST}")
    print("===========================\n")

    # Option 1: Use the enhanced make_parallel_env with different environments
    env = Env.make_parallel_env(
        env_id=ENV_ID,
        num_envs=NUM_ENVS,
        env_seed=ENV_SEED,       # Only pass the environment seed
        use_different_envs=True,  # Enable diverse environments
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
        buffer_size=10000,             # Increased buffer size for more diverse experiences
        learning_starts=5000,          # Allow more random exploration before learning starts
        batch_size=32,                 # Larger batch size for more stable updates
        exploration_fraction=0.4,      # Higher exploration rate to discover diagonal moves
        exploration_final_eps=0.05,    # Higher final exploration to keep trying diagonal moves
        gamma=0.99,                    # Slightly reduced discount factor to focus more on immediate rewards
        learning_rate=5e-4,            # Adjusted learning rate
        train_freq=64,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=log_dir,
        device=device  # Specify the device here
    )

    # Create a modified version of observation_params with random spawn disabled for evaluation
    eval_params = observation_params.copy()
    eval_params["use_random_spawn"] = False  # Disable random spawn for all evaluations
    # Keep diagonal move monitoring during evaluation
    eval_params["monitor_diagonal_moves"] = True

    # Create multiple evaluation environments with different seeds
    NUM_EVAL_ENVS = 15  # Number of different evaluation environments
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
        check_freq=50000,                # Check every 50k steps
        min_reward_threshold=0.9,        # Minimum reward to meet before applying termination conditions
        target_reward_threshold=0.95,    # Target reward to achieve
        max_runtime=30000,               # Maximum runtime in seconds (about 8 hours)
        n_eval_episodes=1,               # Only evaluate on a single episode per environment
        eval_timeout=EVAL_TIMEOUT,       # Use the configurable timeout parameter
        verbose=1
    )

    # Print training start message
    print("\n====== STARTING TRAINING ======")
    print(f"Target timesteps: 1,000,000")
    print(f"Evaluation frequency: Every {termination_callback.check_freq} steps")
    print(f"Evaluation environments: {NUM_EVAL_ENVS}")
    print(f"Evaluation episodes per environment: {termination_callback.n_eval_episodes}")
    print(f"Total evaluation episodes: {NUM_EVAL_ENVS * termination_callback.n_eval_episodes}")
    print(f"Evaluation timeout: {EVAL_TIMEOUT} seconds")
    print(f"Random spawn during training: {observation_params['use_random_spawn']}")
    print(f"Random spawn during evaluation: {eval_params['use_random_spawn']}")
    print("==============================\n")

    model.learn(
        total_timesteps=100_000, 
        tb_log_name="DQN_MiniGrid",
        callback=termination_callback
    )
    model.save("dqn_minigrid_agent_empty_test")

    # Print final evaluation message
    print("\n====== TRAINING COMPLETE ======")
    print("Running final evaluation...")
    print("==============================\n")

    # Final evaluation across multiple environments
    NUM_FINAL_EVAL_ENVS = 10  # More environments for final evaluation
    EPISODES_PER_ENV = 3      # Multiple episodes per environment
    
    print(f"Evaluating on {NUM_FINAL_EVAL_ENVS} environments, {EPISODES_PER_ENV} episodes each")
    
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
        
        # Evaluate on this environment
        env_rewards, env_lengths = evaluate_policy(
            model,
            final_eval_env,
            n_eval_episodes=EPISODES_PER_ENV,
            deterministic=True,
            return_episode_rewards=True,
        )
        
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
    print("==========================\n")
    
    # Visualize the agent's behavior
    print("\n===== DETAILED AGENT BEHAVIOR ANALYSIS =====")
    visualize_agent_behavior(
        model=model,
        env_id=ENV_ID,
        observation_params=observation_params,
        seed=EVAL_SEED + 5000,  # Use a different seed than training/evaluation
        num_episodes=3,
        render=False  # Set to True if you have a display available
    )