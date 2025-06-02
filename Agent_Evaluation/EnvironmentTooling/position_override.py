import numpy as np
import gymnasium as gym
import os
import sys

# Add the project root to Python path so it can find Environment_Tooling
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")

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


class ForceStartState(gym.Wrapper):
    """
    Wrapper that lets you force the agent to spawn at a given position and orientation
    on the very next reset(). Call `.force(pos, dir)` before `.reset()` to queue it up.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._forced_state = None

    def force(self, agent_pos: tuple[int, int], agent_dir: int):
        """
        Queue up a forced start state.
        
        Args:
            agent_pos: (x, y) grid coordinates
            agent_dir: orientation integer (0=right,1=down,2=left,3=up)
        """
        # store as tuple/list to ensure JSON‐serializable if you ever log it
        self._forced_state = (tuple(agent_pos), int(agent_dir))

    def reset(self, *, seed=None, options=None):
        # 1) do the normal reset
        obs, info = super().reset(seed=seed, options=options)

        # 2) if we have a forced state pending, apply it
        if self._forced_state is not None:
            pos, direction = self._forced_state
            base = self.unwrapped  # drill down to the raw MiniGrid env

            # set the env's internal state
            base.agent_pos = np.array(pos, dtype=int)
            base.agent_dir = direction

            # recompute the very first observation
            if hasattr(base, "gen_obs"):
                obs = base.gen_obs()
            elif hasattr(base, "_get_obs"):
                obs = base._get_obs()
            else:
                # fallback: sample from obs space (unlikely needed)
                obs = self.observation_space.sample()

            # clear for next episode
            self._forced_state = None

        return obs, info
 


def make_custom_env(env_id, 
             seed=None,
             render_mode=None,
             window_size=None, 
             cnn_keys=[],
             mlp_keys=[],
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

    # Create base environment
    env = gym.make(env_id, render_mode=render_mode, **kwargs)

    env = ForceStartState(env)
    
    # Apply max episode steps wrapper if specified
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
    
    # Apply no death wrapper if enabled
    if use_no_death:
        env = NoDeath(env, no_death_types=no_death_types, death_cost=death_cost)
    
    env = CustomActionWrapper(
        env,
        diagonal_success_reward=diagonal_success_reward,
        diagonal_failure_penalty=diagonal_failure_penalty
    )
    
    # Apply necessary observation wrappers
    env = FullyObsWrapper(env)
    env = GoalAngleDistanceWrapper(env)
    
    # Apply partial observation for windowed view
    if window_size is not None:
        env = PartialObsWrapper(env, window_size)
        env = ExtractAbstractGrid(env)
    
    # Apply wrapper to select specific observation keys
    env = SelectiveObservationWrapper(
        env,
        cnn_keys=cnn_keys or [],
        mlp_keys=mlp_keys or []
    )
    
    env = ForceFloat32(env)
    

    if seed is not None:
        env.reset(seed=seed)
    
    return env


def dump_wrappers(env):
    w = env
    stack = []
    while True:
        stack.append(type(w).__name__)
        # Drill down one level if possible
        if hasattr(w, 'env'):
            w = w.env
        elif hasattr(w, 'unwrapped') and w is not w.unwrapped:
            w = w.unwrapped
        else:
            break
    return stack