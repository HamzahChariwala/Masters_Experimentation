import os
import random
import numpy as np
import torch
from typing import Tuple

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgObsWrapper, OneHotPartialObsWrapper, NoDeath, DirectionObsWrapper

from stable_baselines3 import PPO, DQN, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------

from EnvironmentEdits.CustomWrappers import GoalAngleDistanceWrapper, PartialObsWrapper, ExtractAbstractGrid, PartialRGBObsWrapper, PartialGrayObsWrapper
from EnvironmentEdits.FeatureExtractor import CustomCombinedExtractor, SelectiveObservationWrapper
from EnvironmentEdits.ActionSpace import CustomActionWrapper

# ---------------------------------------------------------------


def make_env(env_id: str, rank: int, env_seed: int, render_mode: str = None,
    window_size: int = 5, cnn_keys: list = None, mlp_keys: list = None) -> callable:

    if render_mode is not None:
        env = gym.make(env_id, render_mode=render_mode)
    else:
        env = gym.make(env_id)

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
    env = Monitor(env)

    observation, info = env.reset(seed=env_seed + rank)

    print("Custom action space:", env.action_space)
    print(info)
    print(f"Available observation keys: {list(observation.keys())}")
    print(f"Observation space type: {type(env.observation_space)}")
    print(env.observation_space)

    return env, observation


if __name__ == "__main__":

    log_dir = "./logs/dqn_run_1"
    os.makedirs(log_dir, exist_ok=True)

    # ENV_ID = "MiniGrid-LavaGapS7-v0"
    ENV_ID = 'MiniGrid-Empty-5x5-v0'
    env, obs = make_env(
        env_id=ENV_ID,
        rank=0,
        env_seed=42,
        window_size=5,
        cnn_keys=['grey_partial'],
        mlp_keys=["goal_angle", "goal_rotation", "goal_distance", "goal_direction_vector"]
    )

    #  "lava_mask", "barrier_mask", "goal_mask"

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

    model = DQN(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=log_dir
    )

    # model.learn(total_timesteps=1_000_000, tb_log_name="DQN_MiniGrid")
    model.learn(total_timesteps=250_000, tb_log_name="DQN_MiniGrid")
    model.save("dqn_minigrid_agent_cnn_grey")

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Reward: {reward}, Info: {info}")


    # policy_kwargs = dict(
    #     features_extractor_class=CustomCombinedExtractor,
    #     features_extractor_kwargs=dict(
    #         features_dim=256,
    #         cnn_num_layers=2,
    #         cnn_channels=[32, 64],
    #         cnn_kernels=[3, 3],
    #         cnn_strides=[2, 2],
    #         mlp_num_layers=2,
    #         mlp_hidden_sizes=[128, 64],
    #     ),
    #     net_arch=dict(
    #         pi=[128, 64],   # <- Add this
    #         vf=[256, 128]   # <- And this
    #     )
    # )
    # model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
