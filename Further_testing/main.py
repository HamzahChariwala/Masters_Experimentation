import os
import random
import numpy as np
import torch
from typing import Tuple

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import MultiDiscrete

import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgObsWrapper, OneHotPartialObsWrapper, NoDeath, DirectionObsWrapper

from stable_baselines3 import PPO, DQN, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from GymCompatibility import OldGymCompatibility

# ---------------------------------------------------------------

from EnvironmentEdits.CustomWrappers import GoalAngleDistanceWrapper, PartialObsWrapper, ExtractAbstractGrid, PartialRGBObsWrapper, PartialGrayObsWrapper
from EnvironmentEdits.FeatureExtractor import CustomCombinedExtractor, SelectiveObservationWrapper
from EnvironmentEdits.ActionSpace import CustomActionWrapper

# ---------------------------------------------------------------

torch.set_default_dtype(torch.float32)

# Preserve the original torch.Tensor type
original_tensor_type = torch.Tensor

# Define a utility function to create tensors with debugging
original_tensor = torch.Tensor

def create_tensor(*args, **kwargs):
    tensor = original_tensor(*args, **kwargs)
    if isinstance(tensor, original_tensor_type) and tensor.dtype == torch.float64:
        print(f"Warning: Found tensor with dtype {tensor.dtype}. Converting to float32.")
        return tensor.float()
    return tensor

# Extend the DebugWrapper to handle numpy arrays
class DebugWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated  # Combine terminated and truncated flags
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # Debug and convert observations
        if isinstance(obs, dict):
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    print(f"Reset Observation Key: {key}, Shape: {value.shape}, Dtype: {value.dtype}")
                    if value.dtype == np.float64:
                        print(f"Converting observation key {key} to float32")
                        obs[key] = value.astype(np.float32)
        elif isinstance(obs, np.ndarray):
            print(f"Reset Observation Shape: {obs.shape}, Dtype: {obs.dtype}")
            if obs.dtype == np.float64:
                print("Converting observation to float32")
                obs = obs.astype(np.float32)

        return obs


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

    # Wrap the environment with DebugWrapper
    env = DebugWrapper(env)

    # Return only the environment for compatibility with DQN
    return env


def make_parallel_env(env_id: str, num_envs: int, env_seed: int, window_size: int = 5, cnn_keys: list = None, mlp_keys: list = None):
    """
    Create a parallelized environment using SubprocVecEnv.
    """
    def _make_env(rank):
        def _init():
            env = make_env(env_id, rank, env_seed, window_size=window_size, cnn_keys=cnn_keys, mlp_keys=mlp_keys)
            return env
        return _init

    return SubprocVecEnv([_make_env(i) for i in range(num_envs)])


if __name__ == "__main__":

    log_dir = "./logs/dqn_run_1"
    os.makedirs(log_dir, exist_ok=True)

    ENV_ID = 'MiniGrid-Empty-5x5-v0'
    NUM_ENVS = 15  # Number of parallel environments

    env = make_parallel_env(
        env_id=ENV_ID,
        # rank=0,
        num_envs=NUM_ENVS,
        env_seed=42,
        window_size=5,
        cnn_keys=[],
        mlp_keys=["goal_angle", "goal_rotation", "goal_distance", "goal_direction_vector", "barrier_mask", "goal_mask", "lava_mask"]
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

    use_mps = True  # Set to False to disable MPS

    # Ensure all tensors are in float32 for MPS compatibility
    # torch.set_default_dtype(torch.float32)

    device = torch.device("mps" if use_mps and torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DQN(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=log_dir,
        device=device  # Specify the device here
    )

    model.learn(total_timesteps=50_000, tb_log_name="DQN_MiniGrid")
    model.save("dqn_minigrid_agent_cnn_grey")

    # obs = env.reset()

    # print("Observation:", obs)
    # print("Observation type:", type(obs))
    # if isinstance(obs, dict):
    #     for key, value in obs.items():
    #         print(f"Key: {key}, Shape: {value.shape}, Dtype: {value.dtype}")

    # done = False
    # while not done:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     print(f"Reward: {reward}, Info: {info}")


    # 1) wrap your single env in a DummyVecEnv
    raw_eval_env = make_env(
        ENV_ID, rank=0, env_seed=42,
        window_size=5,
        cnn_keys=[], 
        mlp_keys=[
        "goal_angle","goal_rotation","goal_distance",
        "goal_direction_vector","barrier_mask",
        "goal_mask","lava_mask",
        ]
    )

    eval_env = OldGymCompatibility(raw_eval_env)

    # returns two lists: all episode‐rewards and all episode‐lengths
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        eval_env,
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
