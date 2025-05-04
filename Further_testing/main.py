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

    ENV_ID = 'MiniGrid-LavaCrossingS9N2-v0'
    NUM_ENVS = 10  # Increased number of environments for better diversity
    
    # Clearly separated seeds for different sources of randomness
    MODEL_SEED = 42     # For network initialization
    ENV_SEED = 12345    # For environment generation
    EVAL_SEED = 67890   # For evaluation environments
    
    # Evaluation parameters
    EVAL_TIMEOUT = 10   # Timeout in seconds for each evaluation episode
    
    # NoDeath wrapper parameters
    USE_NO_DEATH = True              # Whether to use the NoDeath wrapper
    NO_DEATH_TYPES = ("lava",)       # Which elements should not cause death
    DEATH_COST = -0.5               # Penalty for hitting death elements

    # Define standard observation parameters
    observation_params = {
        "window_size": 5,
        "cnn_keys": [],
        "mlp_keys": ["goal_direction_vector",
                    "goal_angle",
                    "goal_rotation",
                    "barrier_mask",
                    "lava_mask",
                    "goal_mask"],
        "use_random_spawn": True,          # Enable random agent spawn positions
        "exclude_goal_adjacent": True,     # Don't spawn next to goal
        "use_no_death": USE_NO_DEATH,      # Whether to use the NoDeath wrapper
        "no_death_types": NO_DEATH_TYPES,  # Types that don't cause death
        "death_cost": DEATH_COST           # Penalty for hitting death elements
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

    # Option 2: Use the dedicated make_diverse_parallel_env function
    # env = Env.make_diverse_parallel_env(
    #     env_id=ENV_ID,
    #     num_envs=NUM_ENVS,
    #     env_seed=ENV_SEED,     # Only pass the environment seed
    #     **observation_params
    # )

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
        buffer_size=100000,             # Increased buffer size for more diverse experiences
        learning_starts=10000,          # Allow more random exploration before learning starts
        batch_size=256,                 # Larger batch size for more stable updates
        exploration_fraction=0.3,       # Higher exploration rate 
        exploration_final_eps=0.075,    # Higher final exploration
        gamma=0.97,                     # Slightly reduced discount factor to focus more on immediate rewards
        learning_rate=5e-4,             # Adjusted learning rate
        train_freq=256,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=log_dir,
        device=device  # Specify the device here
    )

    # Use the dedicated function to create a properly configured eval environment
    eval_env = Env.make_eval_env(
        env_id=ENV_ID, 
        seed=EVAL_SEED,  # Separate seed for evaluation
        **observation_params
    )

    # Create termination callback with improved evaluation settings
    termination_callback = CustomTerminationCallback(
        eval_env=eval_env,
        check_freq=50000,                # Check every 50k steps
        min_reward_threshold=0.9,        # Minimum reward to meet before applying termination conditions
        target_reward_threshold=0.95,    # Target reward to achieve
        max_runtime=30000,               # Maximum runtime in seconds (about 8 hours)
        n_eval_episodes=1,               # Only evaluate on a single episode
        eval_timeout=EVAL_TIMEOUT,       # Use the configurable timeout parameter
        verbose=1
    )

    # Print training start message
    print("\n====== STARTING TRAINING ======")
    print(f"Target timesteps: 1,000,000")
    print(f"Evaluation frequency: Every {termination_callback.check_freq} steps")
    print(f"Evaluation episodes: {termination_callback.n_eval_episodes}")
    print(f"Evaluation timeout: {EVAL_TIMEOUT} seconds")
    print("==============================\n")

    model.learn(
        total_timesteps=1_000_000, 
        tb_log_name="DQN_MiniGrid",
        callback=termination_callback
    )
    model.save("dqn_minigrid_agent_lava_test")

    # Print final evaluation message
    print("\n====== TRAINING COMPLETE ======")
    print("Running final evaluation...")
    print("==============================\n")

    # Create a fresh evaluation environment for final assessment
    final_eval_env = Env.make_eval_env(
        env_id=ENV_ID, 
        seed=EVAL_SEED + 1000,  # Different seed for final evaluation
        **observation_params
    )

    print("Running final evaluation...")
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        final_eval_env,
        n_eval_episodes=10,  # More episodes for final evaluation
        deterministic=True,
        return_episode_rewards=True,
    )

    print("\n====== FINAL RESULTS ======")
    print("Rewards   : mean %.3f  std %.3f  min %.3f  max %.3f"
        % (np.mean(episode_rewards),
            np.std(episode_rewards),
            np.min(episode_rewards),
            np.max(episode_rewards)))
    print("Lengths   : mean %.1f  std %.1f  min %d    max %d"
        % (np.mean(episode_lengths),
            np.std(episode_lengths),
            np.min(episode_lengths),
            np.max(episode_lengths)))
    print("==========================\n")

    # obs, _ = env.reset()

    # print("Observation:", obs)
    # print("Observation type:", type(obs))
    # if isinstance(obs, dict):
    #     for key, value in obs.items():
    #         print(f"Key: {key}, Shape: {value.shape}, Dtype: {value.dtype}")

    # done = False
    # while not done:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    #     print(f"Reward: {reward}, Info: {info}")
