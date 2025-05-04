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
    NUM_ENVS = 5  # Number of parallel environments

    env = Env.make_parallel_env(
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

    # env = Env.make_env(
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
    eval_env = Env.make_eval_env(
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

    model.learn(
        total_timesteps=100_000_000, 
        tb_log_name="DQN_MiniGrid",
        callback=termination_callback
    )
    model.save("dqn_minigrid_agent_lava_test")


    # Create a fresh evaluation environment for final assessment
    print("Creating final evaluation environment...")
    final_eval_env = Env.make_eval_env(
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
