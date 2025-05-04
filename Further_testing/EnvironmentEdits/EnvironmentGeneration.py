import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgObsWrapper, OneHotPartialObsWrapper, NoDeath, DirectionObsWrapper

from EnvironmentEdits.CustomWrappers import (GoalAngleDistanceWrapper, 
                                             PartialObsWrapper, 
                                             ExtractAbstractGrid, 
                                             PartialRGBObsWrapper, 
                                             PartialGrayObsWrapper, 
                                             ForceFloat32)
from EnvironmentEdits.FeatureExtractor import CustomCombinedExtractor, SelectiveObservationWrapper
from EnvironmentEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.GymCompatibility import OldGymCompatibility


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
