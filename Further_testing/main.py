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

from EnvironmentEdits.CustomWrappers import (GoalAngleDistanceWrapper, 
                                             PartialObsWrapper, 
                                             ExtractAbstractGrid, 
                                             PartialRGBObsWrapper, 
                                             PartialGrayObsWrapper, 
                                             ForceFloat32)
from EnvironmentEdits.FeatureExtractor import CustomCombinedExtractor, SelectiveObservationWrapper
from EnvironmentEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.GymCompatibility import OldGymCompatibility

# ---------------------------------------------------------------

torch.set_default_dtype(torch.float32)


def make_env(env_id: str, rank: int, env_seed: int, render_mode: str = None,
    window_size: int = 5, cnn_keys: list = None, mlp_keys: list = None) -> callable:

    # Always use an explicit render_mode (default to None if not provided)
    env = gym.make(env_id, render_mode=render_mode)

    # Wrap environment with our custom action wrapper
    env = CustomActionWrapper(env)

    env = RecordEpisodeStatistics(env)
    env = FullyObsWrapper(env)
    env = GoalAngleDistanceWrapper(env)
    env = PartialObsWrapper(env, window_size)
    env = ExtractAbstractGrid(env)
    env = PartialRGBObsWrapper(env, window_size)
    env = PartialGrayObsWrapper(env, window_size)
    env = SelectiveObservationWrapper(
        env,
        cnn_keys=cnn_keys or [],
        mlp_keys=mlp_keys or []
    )
    env = ForceFloat32(env)
    env = Monitor(env)

    observation, info = env.reset(seed=env_seed + rank)

    print("Custom action space:", env.action_space)
    print(info)
    print(f"Available observation keys: {list(observation.keys())}")
    print(f"Observation space type: {type(env.observation_space)}")
    print(env.observation_space)

    # Return only the environment for compatibility with DQN
    return env


def make_parallel_env(env_id: str, num_envs: int, env_seed: int, window_size: int = 5, cnn_keys: list = None, mlp_keys: list = None):
    """
    Create a parallelized environment using SubprocVecEnv.
    """
    def _make_env(rank):
        def _init():
            # Always pass explicit render_mode=None for vector environments
            env = make_env(
                env_id, 
                rank, 
                env_seed, 
                render_mode=None,
                window_size=window_size, 
                cnn_keys=cnn_keys, 
                mlp_keys=mlp_keys
            )
            return env
        return _init

    return SubprocVecEnv([_make_env(i) for i in range(num_envs)])


def make_eval_env(env_id: str, seed: int, window_size: int = 5, cnn_keys: list = None, mlp_keys: list = None):
    """
    Create a properly configured evaluation environment.
    """
    # Create the environment with explicit render_mode=None to avoid warning
    env = gym.make(env_id, render_mode=None)
    

    
    # Apply necessary wrappers - keeping all original observation features
    env = CustomActionWrapper(env)
    env = RecordEpisodeStatistics(env)
    env = FullyObsWrapper(env)
    env = GoalAngleDistanceWrapper(env)
    env = PartialObsWrapper(env, window_size)
    env = ExtractAbstractGrid(env)
    env = PartialRGBObsWrapper(env, window_size)
    env = PartialGrayObsWrapper(env, window_size)
    
    env = SelectiveObservationWrapper(
        env,
        cnn_keys=cnn_keys or [],
        mlp_keys=mlp_keys or []
    )
    env = ForceFloat32(env)
    
    # Apply OldGymCompatibility wrapper
    env = OldGymCompatibility(env)
    
    # IMPORTANT: Apply Monitor wrapper AFTER OldGymCompatibility
    env = Monitor(env)
    
    # Reset with seed
    env.reset(seed=seed)
    
    return env


if __name__ == "__main__":

    log_dir = "./logs/dqn_run_2"
    os.makedirs(log_dir, exist_ok=True)

    ENV_ID = 'MiniGrid-LavaCrossingS9N2-v0'
    NUM_ENVS = 8  # Number of parallel environments

    env = make_parallel_env(
        env_id=ENV_ID,
        # rank=0,
        num_envs=NUM_ENVS,
        env_seed=42,
        window_size=5,
        cnn_keys=[],
        mlp_keys=["goal_direction_vector",
                  "goal_angle",
                  "goal_rotation",
                  "barrier_mask",
                  "lava_mask",
                  "goal_mask"]
    )

    # env = make_env(
    #     env_id=ENV_ID,
    #     rank=0,
    #     # num_envs=NUM_ENVS,
    #     env_seed=42,
    #     window_size=5,
    #     cnn_keys=[],
    #     mlp_keys=["goal_angle", "goal_rotation", "goal_distance", "goal_direction_vector", "barrier_mask", "goal_mask", "lava_mask"]
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

    # Ensure all tensors are in float32 for MPS compatibility
    # torch.set_default_dtype(torch.float32)

    device = torch.device("mps" if use_mps and torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DQN(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        buffer_size=50000,
        learning_starts=5000,
        batch_size=128,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=512,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=log_dir,
        device=device  # Specify the device here
    )

    # Use the dedicated function to create a properly configured eval environment
    eval_env = make_eval_env(
        env_id=ENV_ID, 
        seed=811,
        window_size=5,
        cnn_keys=[],
        mlp_keys=["goal_direction_vector",
                  "goal_angle",
                  "goal_rotation",
                  "barrier_mask",
                  "lava_mask",
                  "goal_mask"]
    )

    termination_callback = CustomTerminationCallback(
        eval_env=eval_env,
        check_freq=50000,             
        min_reward_threshold=0.9,      
        target_reward_threshold=0.95,  
        max_runtime=30000,              
        n_eval_episodes=5,             
        verbose=1
    )
    
    # Create the custom termination callback with new parameters
    # termination_callback = CustomTerminationCallback(
    #     eval_env=eval_env,             # Required parameter
    #     check_freq=50000,              # How often to check conditions
        
    #     # Minimum thresholds that must be met before other conditions apply
    #     min_reward_threshold=0.9,      # Minimum reward threshold
    #     # min_episode_length_threshold=100, # Minimum episode length threshold (lower is better)
        
    #     # Target conditions
    #     target_reward_threshold=0.95,   # Target reward to achieve
    #     # max_episodes=2000,             # Maximum number of episodes
    #     max_runtime=7200,              # 1 hour time limit
        
    #     # No improvement conditions
    #     # max_no_improvement_reward_steps=50000,  # Stop if no reward improvement
    #     # max_no_improvement_length_steps=30000,  # Stop if no length improvement
        
    #     n_eval_episodes=5,            # Number of episodes to evaluate
    #     verbose=1
    # )

    model.learn(
        total_timesteps=100_000_000, 
        tb_log_name="DQN_MiniGrid",
        callback=termination_callback
    )
    model.save("dqn_minigrid_agent_lava_test")


    # Create a fresh evaluation environment for final assessment
    print("Creating final evaluation environment...")
    final_eval_env = make_eval_env(
        env_id=ENV_ID, 
        seed=42,
        window_size=5,
        cnn_keys=[],
        mlp_keys=["goal_direction_vector",
                  "goal_angle",
                  "goal_rotation",
                  "barrier_mask",
                  "lava_mask",
                  "goal_mask"]
    )

    print("Running final evaluation...")
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        final_eval_env,
        n_eval_episodes=100,
        deterministic=True,
        return_episode_rewards=True,
    )

    # now you can compute whatever summary you like:
    import numpy as np
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
