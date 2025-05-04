import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgObsWrapper, OneHotPartialObsWrapper, NoDeath, DirectionObsWrapper

from EnvironmentEdits.BespokeEdits.CustomWrappers import (GoalAngleDistanceWrapper, 
                                             PartialObsWrapper, 
                                             ExtractAbstractGrid, 
                                             PartialRGBObsWrapper, 
                                             PartialGrayObsWrapper, 
                                             ForceFloat32)
from EnvironmentEdits.BespokeEdits.FeatureExtractor import CustomCombinedExtractor, SelectiveObservationWrapper
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.BespokeEdits.GymCompatibility import OldGymCompatibility


def make_env(env_id: str, rank: int, env_seed: int, render_mode: str = None,
    window_size: int = 5, cnn_keys: list = None, mlp_keys: list = None) -> callable:

    # Always use an explicit render_mode (default to None if not provided)
    env = gym.make(env_id, render_mode=render_mode)

    # Apply NoDeath wrapper to prevent episode termination on death
    env = NoDeath(env, no_death_types=("lava",), death_cost=-0.1)
    
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


def make_parallel_env(env_id: str, num_envs: int, env_seed: int, window_size: int = 5, cnn_keys: list = None, mlp_keys: list = None, use_different_envs: bool = True, seed_offset: int = 0):
    """
    Create a parallelized environment using SubprocVecEnv.
    
    Parameters:
    ----------
    env_id : str
        The environment ID to create
    num_envs : int
        Number of parallel environments to create
    env_seed : int
        Base seed for environments
    window_size : int
        Size of observation window
    cnn_keys : list
        List of CNN observation keys to include
    mlp_keys : list
        List of MLP observation keys to include
    use_different_envs : bool
        If True, each environment will have a different seed to create diverse environments
        If False, all environments will share the same seed
    seed_offset : int
        An offset to add to all seeds (for creating different environment sets with same patterns)
        
    Returns:
    -------
    SubprocVecEnv with the specified number of environments
    """
    # Store seeds for printing
    env_seeds = []
    
    def _make_env(rank):
        def _init():
            # Calculate the seed for this environment
            if use_different_envs:
                # Each environment gets a unique seed derived from env_seed + rank*1000
                # Using rank*1000 ensures seeds are well separated to create diverse environments
                env_instance_seed = env_seed + seed_offset + (rank * 1000)
            else:
                # All environments share the same env_seed
                env_instance_seed = env_seed + seed_offset
            
            # Store the seed for printing
            if rank >= len(env_seeds):
                env_seeds.append(env_instance_seed)
            
            # Always pass explicit render_mode=None for vector environments
            env = make_env(
                env_id, 
                rank, 
                env_instance_seed, 
                render_mode=None,
                window_size=window_size, 
                cnn_keys=cnn_keys, 
                mlp_keys=mlp_keys
            )
            return env
        return _init

    # Create environments
    env = SubprocVecEnv([_make_env(i) for i in range(num_envs)])
    
    # Print environment seeds
    print("\n====== ENVIRONMENT SEEDS ======")
    print(f"Base seed: {env_seed}, Offset: {seed_offset}, Different envs: {use_different_envs}")
    for i, seed in enumerate(env_seeds):
        print(f"Environment {i}: Seed = {seed}")
    print("===============================\n")
    
    return env


def generate_env_seeds(base_seed: int, num_envs: int, separation: int = 1000):
    """
    Generate an array of well-separated seeds for multiple environments.
    
    Parameters:
    ----------
    base_seed : int
        The base seed value
    num_envs : int
        Number of environments to generate seeds for
    separation : int
        Minimum separation between seeds
        
    Returns:
    -------
    numpy array of seeds
    """
    # Use numpy's random number generator with the base seed for reproducibility
    rng = np.random.RandomState(base_seed)
    
    # Option 1: Deterministically spaced seeds
    # seeds = np.array([base_seed + (i * separation) for i in range(num_envs)])
    
    # Option 2: Pseudo-random but reproducible seeds
    seeds = rng.randint(0, 2**31 - 1, size=num_envs)
    
    return seeds


def make_diverse_parallel_env(env_id: str, num_envs: int, env_seed: int, 
                             window_size: int = 5, cnn_keys: list = None, mlp_keys: list = None):
    """
    Create a parallelized environment with diverse, reproducible seeds.
    
    Parameters:
    ----------
    env_id : str
        The environment ID to create
    num_envs : int
        Number of parallel environments to create
    env_seed : int
        Seed for generating environment seeds
    window_size : int
        Size of observation window
    cnn_keys : list
        List of CNN observation keys to include
    mlp_keys : list
        List of MLP observation keys to include
        
    Returns:
    -------
    SubprocVecEnv with diverse environments
    """
    # Generate diverse seeds for each environment
    seeds = generate_env_seeds(env_seed, num_envs)
    
    # Print generated seeds
    print("\n====== DIVERSE ENVIRONMENT SEEDS ======")
    print(f"Generator seed: {env_seed}, Num environments: {num_envs}")
    for i, seed in enumerate(seeds):
        print(f"Environment {i}: Seed = {seed}")
    print("======================================\n")
    
    def _make_env(rank):
        def _init():
            # Use the generated seed for this environment
            env_seed = int(seeds[rank])
            
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
    print(f"\n====== EVALUATION ENVIRONMENT ======")
    print(f"Creating evaluation environment with seed: {seed}")
    print("====================================\n")
    
    # Create the environment with explicit render_mode=None to avoid warning
    env = gym.make(env_id, render_mode=None)
    
    # Apply NoDeath wrapper to prevent episode termination on death
    env = NoDeath(env, no_death_types=("lava",), death_cost=-0.1)
    
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
