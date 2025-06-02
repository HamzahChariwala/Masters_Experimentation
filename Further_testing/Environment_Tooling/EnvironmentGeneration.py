import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordEpisodeStatistics
import numpy as np
from typing import Dict, Any, List, Callable, Union, Tuple, Optional
import random
import torch
import os

# Custom monitor wrapper that properly accesses attributes through unwrapped
# class SafeMonitor(Monitor):
#     """A wrapper that properly accesses environment attributes through unwrapped to avoid deprecation warnings."""
#     def __init__(self, env, filename=None, allow_early_resets=True, 
#                  reset_keywords=(), info_keywords=()):
#         super().__init__(env, filename, allow_early_resets, reset_keywords, info_keywords)
        
#     def get_agent_pos(self):
#         # Always access agent_pos through the unwrapped environment
#         if hasattr(self.env.unwrapped, 'agent_pos'):
#             return self.env.unwrapped.agent_pos
#         return None  # Return None if agent_pos doesn't exist
    
#     def reset(self, *, seed=None, options=None):
#         #   — forward both arguments so MiniGrid.reset(options=…) actually runs!
#         obs, info = self.env.reset(seed=seed, options=options)
#         #  now let Monitor do its logging book-keeping
#         return obs, info
    

# class SafeMonitor(RecordEpisodeStatistics):
#     def __init__(self, env):
#         # no filename / keywords here—just wrap the env
#         super().__init__(env)

#     def get_agent_pos(self):
#         if hasattr(self.env.unwrapped, 'agent_pos'):
#             return self.env.unwrapped.agent_pos
#         return None

#     def reset(self, *, seed=None, options=None):
#         # forward both seed and options so MiniGrid.reset(options=…) works
#         obs, info = self.env.reset(seed=seed, options=options)
#         return obs, info


class SafeMonitor(Monitor):
    """Propagate reset(options=…) and let Monitor do its own logging."""
    def __init__(self, env):
        super().__init__(env)
    def get_agent_pos(self):
        # Still works the same way
        if hasattr(self.env.unwrapped, 'agent_pos'):
            return self.env.unwrapped.agent_pos
        return None
    # no override of reset needed—Monitor already forwards `reset(seed=…, options=…)`



# Safe version of evaluate_policy that ensures all attribute access goes through unwrapped
def safe_evaluate_policy(
    model, 
    env, 
    n_eval_episodes=10, 
    deterministic=True, 
    render=False, 
    callback=None, 
    reward_threshold=None, 
    return_episode_rewards=False,
    warn=True
):
    """
    A wrapper around stable_baselines3's evaluate_policy that ensures environment attributes
    are accessed through unwrapped to avoid deprecation warnings.
    
    This function has the same interface as stable_baselines3.common.evaluation.evaluate_policy
    """
    # Import here to avoid circular imports
    from stable_baselines3.common.evaluation import evaluate_policy
    
    # Make sure the environment is wrapped with SafeMonitor if it's not already
    if not hasattr(env, "get_agent_pos"):
        # For VecEnv, we need to access the base envs
        if hasattr(env, "envs") and len(env.envs) > 0:
            for i, sub_env in enumerate(env.envs):
                if not hasattr(sub_env, "get_agent_pos"):
                    env.envs[i] = SafeMonitor(sub_env)
        else:
            # Try to wrap directly
            env = SafeMonitor(env)
    
    # Create a wrapper for the callback to ensure agent_pos is accessed safely
    original_callback = callback
    
    def safe_callback(locals_dict, globals_dict):
        # If access to agent_pos or similar attributes is needed in the callback,
        # make sure to use unwrapped
        if original_callback is not None:
            return original_callback(locals_dict, globals_dict)
        return None
    
    # Call the original evaluate_policy with our safeguards
    return evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
        callback=safe_callback if original_callback else None,
        reward_threshold=reward_threshold,
        return_episode_rewards=return_episode_rewards,
        warn=warn
    )

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgObsWrapper, OneHotPartialObsWrapper, NoDeath, DirectionObsWrapper

from Environment_Tooling.BespokeEdits.CustomWrappers import (GoalAngleDistanceWrapper, 
                                             PartialObsWrapper, 
                                             ExtractAbstractGrid, 
                                             PartialRGBObsWrapper, 
                                             PartialGrayObsWrapper, 
                                             ForceFloat32,
                                             RandomSpawnWrapper,
                                             DiagonalMoveMonitor)
from Environment_Tooling.BespokeEdits.RewardModifications import EpisodeCompletionRewardWrapper, LavaStepCounterWrapper
from Environment_Tooling.BespokeEdits.FeatureExtractor import CustomCombinedExtractor, SelectiveObservationWrapper
from Environment_Tooling.BespokeEdits.ActionSpace import CustomActionWrapper
from Environment_Tooling.BespokeEdits.GymCompatibility import OldGymCompatibility
from Environment_Tooling.BespokeEdits.SpawnDistribution import FlexibleSpawnWrapper


def _make_env(env_id, 
             seed=None,
             render_mode=None,
             window_size=None, 
             cnn_keys=None,
             mlp_keys=None,
             use_random_spawn=False,
             exclude_goal_adjacent=False,
             use_no_death=False,
             no_death_types=None,
             death_cost=0,
             max_episode_steps=None,
             monitor_diagonal_moves=False,
             diagonal_success_reward=0.01,
             diagonal_failure_penalty=0.01,
             use_flexible_spawn=False,
             spawn_distribution_type="uniform",
             spawn_distribution_params=None,
             use_stage_training=False,
             stage_training_config=None,
             use_continuous_transition=False,
             continuous_transition_config=None,
             spawn_vis_dir=None,
             spawn_vis_frequency=10000,
             use_reward_function=False,
             reward_type="linear",
             reward_x_intercept=100,
             reward_y_intercept=1.0,
             reward_transition_width=10,
             reward_verbose=True,
             debug_logging=False,
             count_lava_steps=False,
             lava_step_multiplier=2.0,
             **kwargs):
    """
    Create and configure a MiniGrid environment with specified wrappers.
    
    Parameters:
    -----------
    env_id : str
        The environment ID to create
    seed : int, optional
        Random seed for environment
    render_mode : str, optional
        Rendering mode ('human', 'rgb_array', etc.)
    window_size : int, optional
        Size of observation window
    cnn_keys : list, optional
        List of observation keys to use for CNN
    mlp_keys : list, optional
        List of observation keys to use for MLP
    use_random_spawn : bool, optional
        Whether to use random spawn positions
    exclude_goal_adjacent : bool, optional
        Whether to exclude positions adjacent to goal
    use_no_death : bool, optional
        Whether to prevent death
    no_death_types : list, optional
        List of death types to prevent
    death_cost : float, optional
        Cost for hitting death elements
    max_episode_steps : int, optional
        Maximum steps per episode
    monitor_diagonal_moves : bool, optional
        Whether to monitor diagonal moves
    diagonal_success_reward : float, optional
        Reward for successful diagonal moves
    diagonal_failure_penalty : float, optional
        Penalty for failed diagonal moves
    use_flexible_spawn : bool, optional
        Whether to use flexible spawn distribution
    spawn_distribution_type : str, optional
        Type of spawn distribution
    spawn_distribution_params : dict, optional
        Parameters for spawn distribution
    use_stage_training : bool, optional
        Whether to use stage-based training
    stage_training_config : dict, optional
        Configuration for stage-based training
    use_continuous_transition : bool, optional
        Whether to use continuous transition
    continuous_transition_config : dict, optional
        Configuration for continuous transition
    spawn_vis_dir : str, optional
        Directory for spawn visualization
    spawn_vis_frequency : int, optional
        Frequency of spawn visualization
    use_reward_function : bool, optional
        Whether to use custom reward function
    reward_type : str, optional
        Type of reward function ('linear', 'exponential', 'sigmoid')
    reward_x_intercept : float, optional
        Number of steps at which reward becomes approximately 0
    reward_y_intercept : float, optional
        Initial reward value at step 0
    reward_transition_width : float, optional
        For sigmoid function, controls transition speed
    reward_verbose : bool, optional
        Whether to include verbose reward information
    debug_logging : bool, optional
        Whether to print detailed step-by-step debugging logs
    count_lava_steps : bool, optional
        Whether to count lava steps
    lava_step_multiplier : float, optional
        Multiplier for lava step count
    **kwargs : dict
        Additional arguments passed to environment creation
    """
    # Create base environment
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    
    # Apply max episode steps wrapper if specified
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
    
    # Apply no death wrapper if enabled
    if use_no_death:
        env = NoDeath(env, no_death_types=no_death_types, death_cost=death_cost)
    
    # Apply random spawn wrapper if enabled
    if use_random_spawn:
        env = RandomSpawnWrapper(env, exclude_goal_adjacent=exclude_goal_adjacent)
    
    # Apply flexible spawn wrapper if enabled
    if use_flexible_spawn:
        env = FlexibleSpawnWrapper(
            env,
            distribution_type=spawn_distribution_type,
            distribution_params=spawn_distribution_params,
            stage_based_training=stage_training_config if use_stage_training else None,
            temporal_transition=continuous_transition_config if use_continuous_transition else None,
            exclude_goal_adjacent=exclude_goal_adjacent
        )
    
    # Apply custom action wrapper if using diagonal moves
    if monitor_diagonal_moves:
        env = CustomActionWrapper(
            env,
            diagonal_success_reward=diagonal_success_reward,
            diagonal_failure_penalty=diagonal_failure_penalty
        )
    
    # Apply reward function wrapper if enabled
    if use_reward_function:
        # First apply lava step counter if count_lava_steps is enabled
        if count_lava_steps:
            env = LavaStepCounterWrapper(
                env,
                lava_step_multiplier=lava_step_multiplier,
                verbose=reward_verbose,
                debug_logging=debug_logging
            )
        
        # Then apply the reward wrapper
        env = EpisodeCompletionRewardWrapper(
            env,
            reward_type=reward_type,
            x_intercept=reward_x_intercept,
            y_intercept=reward_y_intercept,
            transition_width=reward_transition_width,
            count_lava_steps=count_lava_steps,
            verbose=reward_verbose
        )
    
    # Apply necessary observation wrappers
    env = FullyObsWrapper(env)
    env = GoalAngleDistanceWrapper(env)
    
    # Apply partial observation for windowed view
    if window_size is not None:
        env = PartialObsWrapper(env, window_size)
        env = ExtractAbstractGrid(env)
        env = PartialRGBObsWrapper(env, window_size)
        env = PartialGrayObsWrapper(env, window_size)
    
    # Apply wrapper to select specific observation keys
    if cnn_keys is not None or mlp_keys is not None:
        env = SelectiveObservationWrapper(
            env,
            cnn_keys=cnn_keys or [],
            mlp_keys=mlp_keys or []
        )
    
    # Apply wrapper to ensure float32 type for observations
    env = ForceFloat32(env)
    
    # Apply action wrapper to ensure compatibility with old gym action space
    env = OldGymCompatibility(env)
    
    # If seed is provided, seed the environment
    if seed is not None:
        env.reset(seed=seed)
    
    return env


def make_env(env_id: str, rank: int, env_seed: int, render_mode: str = None,
           window_size: int = 7, cnn_keys: list = None, mlp_keys: list = None,
           use_random_spawn: bool = False, exclude_goal_adjacent: bool = False,
           use_no_death: bool = True, no_death_types: tuple = ("lava",), death_cost: float = -0.25,
           max_episode_steps: int = None, monitor_diagonal_moves: bool = False,
           diagonal_success_reward: float = 1.5, diagonal_failure_penalty: float = 0.1,
           use_flexible_spawn: bool = False, spawn_distribution_type: str = "uniform",
           spawn_distribution_params: dict = None, use_stage_training: bool = False,
           stage_training_config: dict = None, use_continuous_transition: bool = False,
           continuous_transition_config: dict = None, spawn_vis_dir: str = None,
           spawn_vis_frequency: int = 10000, **kwargs) -> callable:

    # Always use an explicit render_mode (default to None if not provided)
    render_mode = render_mode or None

    def _init():
        # Create the environment with a unique seed based on env_seed + rank
        env_instance_seed = env_seed + rank
        
        # Use the _make_env function
        env = _make_env(
            env_id=env_id,
            seed=env_instance_seed,
            render_mode=render_mode,
            window_size=window_size,
            cnn_keys=cnn_keys,
            mlp_keys=mlp_keys,
            use_random_spawn=use_random_spawn,
            exclude_goal_adjacent=exclude_goal_adjacent,
            use_no_death=use_no_death,
            no_death_types=no_death_types,
            death_cost=death_cost,
            max_episode_steps=max_episode_steps,
            monitor_diagonal_moves=monitor_diagonal_moves,
            diagonal_success_reward=diagonal_success_reward,
            diagonal_failure_penalty=diagonal_failure_penalty,
            use_flexible_spawn=use_flexible_spawn,
            spawn_distribution_type=spawn_distribution_type,
            spawn_distribution_params=spawn_distribution_params,
            use_stage_training=use_stage_training,
            stage_training_config=stage_training_config,
            use_continuous_transition=use_continuous_transition,
            continuous_transition_config=continuous_transition_config,
            spawn_vis_dir=spawn_vis_dir,
            spawn_vis_frequency=spawn_vis_frequency,
            **kwargs
        )
        
        # Apply Monitor wrapper to record episode information
        env = SafeMonitor(env)

        observation, info = env.reset(seed=env_seed + rank)
        
        if render_mode is not None:
            print(f"Created environment {env_id} with render_mode={render_mode}, seed={env_instance_seed}")
        
        return env

    return _init


def make_parallel_env(env_id: str, num_envs: int, env_seed: int, window_size: int = 5, 
                      cnn_keys: list = None, mlp_keys: list = None, use_different_envs: bool = True, 
                      seed_offset: int = 0, use_random_spawn: bool = False, exclude_goal_adjacent: bool = True,
                      use_no_death: bool = True, no_death_types: tuple = ("lava",), death_cost: float = -0.25,
                      max_episode_steps: int = None, monitor_diagonal_moves: bool = False,
                      diagonal_success_reward: float = 1.5, diagonal_failure_penalty: float = 0.1,
                      use_flexible_spawn: bool = False, spawn_distribution_type: str = "uniform",
                      spawn_distribution_params: dict = None, use_stage_training: bool = False,
                      stage_training_config: dict = None, use_continuous_transition: bool = False,
                      continuous_transition_config: dict = None, spawn_vis_dir: str = None,
                      spawn_vis_frequency: int = 10000, **kwargs):
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
    use_random_spawn : bool
        If True, agent will spawn at random empty locations
    exclude_goal_adjacent : bool 
        If True and use_random_spawn is True, agent won't spawn adjacent to goal
    use_no_death : bool
        If True, apply the NoDeath wrapper to prevent episode termination on death
    no_death_types : tuple
        Types of elements that shouldn't cause death (e.g., "lava")
    death_cost : float
        Penalty applied when agent would normally die but is prevented by the wrapper
    max_episode_steps : int
        Maximum number of steps per episode before truncation. If None, uses environment default.
    monitor_diagonal_moves : bool
        If True, apply the DiagonalMoveMonitor wrapper to track diagonal move usage
    diagonal_success_reward : float
        Reward for successful diagonal moves
    diagonal_failure_penalty : float
        Penalty for failed diagonal moves
    use_flexible_spawn : bool
        If True, use the FlexibleSpawnWrapper for spawn distribution
    spawn_distribution_type : str
        Type of spawn distribution to use
    spawn_distribution_params : dict
        Parameters for the spawn distribution
    use_stage_training : bool
        If True, use stage-based training for spawn distribution
    stage_training_config : dict
        Configuration for stage-based training
    use_continuous_transition : bool
        If True, use continuous transition for spawn distribution
    continuous_transition_config : dict
        Configuration for continuous transition
    spawn_vis_dir : str
        Directory for spawn visualization
    spawn_vis_frequency : int
        Frequency for spawn visualization
        
    Returns:
    -------
    SubprocVecEnv with the specified number of environments
    """
    # Store seeds for printing
    env_seeds = []
    
    def env_creator(rank):
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
            
            # Use the global _make_env function
            env = _make_env(
                env_id=env_id, 
                seed=env_instance_seed,
                render_mode=None,  # Always use None for vector environments
                window_size=window_size, 
                cnn_keys=cnn_keys, 
                mlp_keys=mlp_keys,
                use_random_spawn=use_random_spawn,
                exclude_goal_adjacent=exclude_goal_adjacent,
                use_no_death=use_no_death,
                no_death_types=no_death_types,
                death_cost=death_cost,
                max_episode_steps=max_episode_steps,
                monitor_diagonal_moves=monitor_diagonal_moves,
                diagonal_success_reward=diagonal_success_reward,
                diagonal_failure_penalty=diagonal_failure_penalty,
                use_flexible_spawn=use_flexible_spawn,
                spawn_distribution_type=spawn_distribution_type,
                spawn_distribution_params=spawn_distribution_params,
                use_stage_training=use_stage_training,
                stage_training_config=stage_training_config,
                use_continuous_transition=use_continuous_transition,
                continuous_transition_config=continuous_transition_config,
                spawn_vis_dir=spawn_vis_dir,
                spawn_vis_frequency=spawn_vis_frequency,
                **kwargs
            )
            
            # Wrap with monitor
            env = SafeMonitor(env)
            return env
            
        return _init

    # Create environments
    env = SubprocVecEnv([env_creator(i) for i in range(num_envs)])
    
    # Print environment seeds
    print("\n====== ENVIRONMENT SEEDS ======")
    print(f"Base seed: {env_seed}, Offset: {seed_offset}, Different envs: {use_different_envs}")
    print(f"Random agent spawning: {use_random_spawn}")
    if use_no_death:
        print(f"NoDeath wrapper: active, types={no_death_types}, cost={death_cost}")
    else:
        print(f"NoDeath wrapper: disabled")
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
                             window_size: int = 5, cnn_keys: list = None, mlp_keys: list = None,
                             use_random_spawn: bool = False, exclude_goal_adjacent: bool = True,
                             use_no_death: bool = True, no_death_types: tuple = ("lava",), death_cost: float = -0.25,
                             max_episode_steps: int = None, monitor_diagonal_moves: bool = False,
                             diagonal_success_reward: float = 1.5, diagonal_failure_penalty: float = 0.1,
                             use_flexible_spawn: bool = False, spawn_distribution_type: str = "uniform",
                             spawn_distribution_params: dict = None, use_stage_training: bool = False,
                             stage_training_config: dict = None, use_continuous_transition: bool = False,
                             continuous_transition_config: dict = None, spawn_vis_dir: str = None,
                             spawn_vis_frequency: int = 10000, **kwargs):
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
    use_random_spawn : bool
        If True, agent will spawn at random empty locations
    exclude_goal_adjacent : bool 
        If True and use_random_spawn is True, agent won't spawn adjacent to goal
    use_no_death : bool
        If True, apply the NoDeath wrapper to prevent episode termination on death
    no_death_types : tuple
        Types of elements that shouldn't cause death (e.g., "lava")
    death_cost : float
        Penalty applied when agent would normally die but is prevented by the wrapper
    max_episode_steps : int
        Maximum number of steps per episode before truncation. If None, uses environment default.
    monitor_diagonal_moves : bool
        If True, apply the DiagonalMoveMonitor wrapper to track diagonal move usage
    diagonal_success_reward : float
        Reward for successful diagonal moves
    diagonal_failure_penalty : float
        Penalty for failed diagonal moves
    use_flexible_spawn : bool
        If True, use the FlexibleSpawnWrapper for spawn distribution
    spawn_distribution_type : str
        Type of spawn distribution to use
    spawn_distribution_params : dict
        Parameters for the spawn distribution
    use_stage_training : bool
        If True, use stage-based training for spawn distribution
    stage_training_config : dict
        Configuration for stage-based training
    use_continuous_transition : bool
        If True, use continuous transition for spawn distribution
    continuous_transition_config : dict
        Configuration for continuous transition
    spawn_vis_dir : str
        Directory for spawn visualization
    spawn_vis_frequency : int
        Frequency for spawn visualization
        
    Returns:
    -------
    SubprocVecEnv with diverse environments
    """
    # Generate diverse seeds for each environment
    seeds = generate_env_seeds(env_seed, num_envs)
    
    # Print generated seeds
    print("\n====== DIVERSE ENVIRONMENT SEEDS ======")
    print(f"Generator seed: {env_seed}, Num environments: {num_envs}")
    print(f"Random agent spawning: {use_random_spawn}")
    if use_no_death:
        print(f"NoDeath wrapper: active, types={no_death_types}, cost={death_cost}")
    else:
        print(f"NoDeath wrapper: disabled")
    for i, seed in enumerate(seeds):
        print(f"Environment {i}: Seed = {seed}")
    print("======================================\n")
    
    def env_creator(rank):
        def _init():
            # Use the generated seed for this environment
            env_instance_seed = int(seeds[rank])
            
            # Use the global _make_env function
            env = _make_env(
                env_id=env_id, 
                seed=env_instance_seed,
                render_mode=None,  # Always use None for vector environments
                window_size=window_size, 
                cnn_keys=cnn_keys, 
                mlp_keys=mlp_keys,
                use_random_spawn=use_random_spawn,
                exclude_goal_adjacent=exclude_goal_adjacent,
                use_no_death=use_no_death,
                no_death_types=no_death_types,
                death_cost=death_cost,
                max_episode_steps=max_episode_steps,
                monitor_diagonal_moves=monitor_diagonal_moves,
                diagonal_success_reward=diagonal_success_reward,
                diagonal_failure_penalty=diagonal_failure_penalty,
                use_flexible_spawn=use_flexible_spawn,
                spawn_distribution_type=spawn_distribution_type,
                spawn_distribution_params=spawn_distribution_params,
                use_stage_training=use_stage_training,
                stage_training_config=stage_training_config,
                use_continuous_transition=use_continuous_transition,
                continuous_transition_config=continuous_transition_config,
                spawn_vis_dir=spawn_vis_dir,
                spawn_vis_frequency=spawn_vis_frequency,
                **kwargs
            )
            
            # Wrap with monitor
            env = SafeMonitor(env)
            return env
        return _init

    return SubprocVecEnv([env_creator(i) for i in range(num_envs)])


def make_eval_env(env_id: str, seed: int, window_size: int = 5, cnn_keys: list = None, 
                 mlp_keys: list = None, use_random_spawn: bool = False, exclude_goal_adjacent: bool = True,
                 use_no_death: bool = True, no_death_types: tuple = ("lava",), death_cost: float = -0.25,
                 max_episode_steps: int = None, monitor_diagonal_moves: bool = False,
                 diagonal_success_reward: float = 1.5, diagonal_failure_penalty: float = 0.1,
                 use_flexible_spawn: bool = False, spawn_distribution_type: str = "uniform",
                 spawn_distribution_params: dict = None, use_stage_training: bool = False,
                 stage_training_config: dict = None, use_continuous_transition: bool = False,
                 continuous_transition_config: dict = None, spawn_vis_dir: str = None,
                 spawn_vis_frequency: int = 10000, **kwargs):
    """
    Create a properly configured evaluation environment.
    
    Parameters:
    ----------
    env_id : str
        The environment ID to create
    seed : int
        Seed for the environment
    window_size : int
        Size of observation window
    cnn_keys : list
        List of CNN observation keys to include
    mlp_keys : list
        List of MLP observation keys to include
    use_random_spawn : bool
        If True, agent will spawn at random empty locations
    exclude_goal_adjacent : bool 
        If True and use_random_spawn is True, agent won't spawn adjacent to goal
    use_no_death : bool
        If True, apply the NoDeath wrapper to prevent episode termination on death
    no_death_types : tuple
        Types of elements that shouldn't cause death (e.g., "lava")
    death_cost : float
        Penalty applied when agent would normally die but is prevented by the wrapper
    max_episode_steps : int
        Maximum number of steps per episode before truncation. If None, uses environment default.
    monitor_diagonal_moves : bool
        If True, apply the DiagonalMoveMonitor wrapper to track diagonal move usage
    diagonal_success_reward : float
        Reward for successful diagonal moves
    diagonal_failure_penalty : float
        Penalty for failed diagonal moves
    use_flexible_spawn : bool
        If True, use the FlexibleSpawnWrapper for spawn distribution
    spawn_distribution_type : str
        Type of spawn distribution to use
    spawn_distribution_params : dict
        Parameters for the spawn distribution
    use_stage_training : bool
        If True, use stage-based training for spawn distribution
    stage_training_config : dict
        Configuration for stage-based training
    use_continuous_transition : bool
        If True, use continuous transition for spawn distribution
    continuous_transition_config : dict
        Configuration for continuous transition
    spawn_vis_dir : str
        Directory for spawn visualization
    spawn_vis_frequency : int
        Frequency for spawn visualization
    """
    print(f"\n====== EVALUATION ENVIRONMENT ======")
    print(f"Creating evaluation environment with seed: {seed}")
    print(f"Random agent spawning: {use_random_spawn}")
    if use_no_death:
        print(f"NoDeath wrapper: active, types={no_death_types}, cost={death_cost}")
    else:
        print(f"NoDeath wrapper: disabled")
    if max_episode_steps is not None:
        print(f"Max episode steps: {max_episode_steps}")
    print(f"Diagonal success reward: {diagonal_success_reward}")
    print(f"Diagonal failure penalty: {diagonal_failure_penalty}")
    print("====================================\n")
    
    # Create the environment with explicit render_mode=None to avoid warning
    env = gym.make(env_id, render_mode=None)
    
    # Apply TimeLimit wrapper if max_episode_steps is specified
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    # Apply NoDeath wrapper to prevent episode termination on death if requested
    if use_no_death:
        env = NoDeath(env, no_death_types=no_death_types, death_cost=death_cost)
    
    # Apply necessary wrappers - keeping all original observation features
    env = CustomActionWrapper(env, 
                             diagonal_success_reward=diagonal_success_reward, 
                             diagonal_failure_penalty=diagonal_failure_penalty)
    env = RecordEpisodeStatistics(env)
    env = FullyObsWrapper(env)
    
    # Apply RandomSpawnWrapper if requested
    if use_random_spawn:
        env = RandomSpawnWrapper(env, exclude_goal_adjacent=exclude_goal_adjacent, env_id=999)  # Special ID for eval env
        
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
    env = SafeMonitor(env)
    
    # Reset with seed
    env.reset(seed=seed)
    
    # Apply flexible spawn distribution if specified
    if use_flexible_spawn:
        # Determine total timesteps (if not provided, use a reasonable default)
        total_timesteps = kwargs.get("total_timesteps", 100000)
        
        # Prepare the distribution parameters
        if spawn_distribution_params is None:
            spawn_distribution_params = {}
        
        # Handle different curriculum learning approaches
        if use_stage_training and stage_training_config:
            # Use stage-based training
            env = FlexibleSpawnWrapper(
                env,
                distribution_type=spawn_distribution_type,  # This will be overridden by stages
                distribution_params=spawn_distribution_params,
                total_timesteps=total_timesteps,
                exclude_occupied=True,
                exclude_goal_adjacent=exclude_goal_adjacent,
                stage_based_training=stage_training_config
            )
        elif use_continuous_transition and continuous_transition_config:
            # Use continuous transition
            env = FlexibleSpawnWrapper(
                env,
                distribution_type=spawn_distribution_type,
                distribution_params=spawn_distribution_params,
                total_timesteps=total_timesteps,
                exclude_occupied=True,
                exclude_goal_adjacent=exclude_goal_adjacent,
                temporal_transition=continuous_transition_config
            )
        else:
            # Use fixed distribution
            env = FlexibleSpawnWrapper(
                env,
                distribution_type=spawn_distribution_type,
                distribution_params=spawn_distribution_params,
                exclude_occupied=True,
                exclude_goal_adjacent=exclude_goal_adjacent
            )

    return env
