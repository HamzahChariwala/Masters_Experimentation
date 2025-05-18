import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import Wrapper
import math

# class PositionAwareWrapper(gym.Wrapper):
#     """
#     A wrapper that ensures agent position and orientation are properly set during evaluations
#     and that the feature extractors receive the correct position data.
    
#     This wrapper is specifically designed for evaluation purposes and should not affect
#     the training pipeline.
#     """
    
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env
#         # Keep track of current agent position and orientation for feature computation
#         self._agent_pos = None
#         self._agent_dir = None
        
#         # Find the actual MiniGrid environment in the wrapper stack
#         self._find_minigrid_env()
        
#     def _find_minigrid_env(self):
#         """Find the underlying MiniGrid environment in the wrapper stack."""
#         env = self.env
#         self.minigrid_env = None
        
#         # Search for environment with agent_pos attribute
#         while hasattr(env, 'env'):
#             if hasattr(env, 'agent_pos') and hasattr(env, 'agent_dir'):
#                 self.minigrid_env = env
#                 break
#             env = env.env
            
#         if self.minigrid_env is None:
#             print("WARNING: Could not find MiniGrid environment in wrapper stack")
    
#     def reset(self, **kwargs):
#         """
#         Reset the environment and ensure agent position and orientation are properly set.
#         """
#         obs, info = self.env.reset(**kwargs)
        
#         # Check if options contains agent position and direction
#         if 'options' in kwargs and kwargs['options'] is not None:
#             options = kwargs['options']
#             if 'agent_pos' in options and 'agent_dir' in options:
#                 self._agent_pos = options['agent_pos']
#                 self._agent_dir = options['agent_dir']
                
#                 # Force set the agent position and direction in the MiniGrid environment
#                 if self.minigrid_env is not None:
#                     print(f"PositionAwareWrapper: Setting agent position to {self._agent_pos} and direction to {self._agent_dir}")
#                     self.minigrid_env.agent_pos = np.array(self._agent_pos)
#                     self.minigrid_env.agent_dir = self._agent_dir
                    
#                     # Update observation data based on new position
#                     self._update_obs_with_position(obs, info)
        
#         return obs, info
    
#     def step(self, action):
#         """
#         Execute environment step and track agent position for feature computation.
#         """
#         obs, reward, terminated, truncated, info = self.env.step(action)
        
#         # Update agent position and direction after step
#         if self.minigrid_env is not None:
#             self._agent_pos = self.minigrid_env.agent_pos
#             self._agent_dir = self.minigrid_env.agent_dir
            
#             # Update observation data
#             self._update_obs_with_position(obs, info)
        
#         return obs, reward, terminated, truncated, info

    
# class PositionAwareWrapper(gym.Wrapper):
#     """
#     A wrapper that fixes agent_pos/agent_dir handling by always using env.unwrapped,
#     and updates obs/info accordingly.
#     """
#     def __init__(self, env):
#         super().__init__(env)
#         # The true MiniGrid env is always env.unwrapped
#         self.minigrid_env = env.unwrapped

#     def reset(self, *, seed=None, options=None):
#         # forward seed/options to underlying
#         obs, info = self.env.reset(seed=seed, options=options)
#         # if user provided a starting state, force-set it
#         if options and 'agent_pos' in options and 'agent_dir' in options:
#             pos = options['agent_pos']
#             d = options['agent_dir']
#             self.minigrid_env.agent_pos = np.array(pos)
#             self.minigrid_env.agent_dir = d
#             # regenerate raw observation so other wrappers see the change
#             if hasattr(self.minigrid_env, 'gen_obs'):
#                 raw = self.minigrid_env.gen_obs()
#                 # copy into obs dict where appropriate
#                 if isinstance(obs, dict) and 'image' in raw:
#                     obs.update(raw)
#                 else:
#                     obs = raw
#         # record base pos/dir
#         self._agent_pos = tuple(self.minigrid_env.agent_pos)
#         self._agent_dir = self.minigrid_env.agent_dir
#         # inject into info
#         info.setdefault('log_data', {})['agent_pos'] = self._agent_pos
#         info['log_data']['agent_dir'] = self._agent_dir
#         return obs, info

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         # always pull real pos/dir from base
#         self._agent_pos = tuple(self.minigrid_env.agent_pos)
#         self._agent_dir = self.minigrid_env.agent_dir
#         info.setdefault('log_data', {})['agent_pos'] = self._agent_pos
#         info['log_data']['agent_dir'] = self._agent_dir
#         return obs, reward, terminated, truncated, info

class PositionAwareWrapper(gym.Wrapper):
    """
    Ensure all position/orientation resets and steps use the true MiniGrid env,
    by locating it in the wrapper stack and syncing agent_pos/agent_dir into info['log_data'].
    """
    def __init__(self, env):
        super().__init__(env)
        # Locate the underlying MiniGrid environment (where agent_pos and agent_dir live)
        self.minigrid_env = self._find_minigrid_env(env)

    def _find_minigrid_env(self, env):
        current = env
        # Crawl through both .env and .unwrapped until we find the base
        while True:
            if hasattr(current, 'agent_pos') and hasattr(current, 'agent_dir'):
                return current
            if hasattr(current, 'env'):
                current = current.env
            elif hasattr(current, 'unwrapped'):
                current = current.unwrapped
            else:
                break
        raise RuntimeError("PositionAwareWrapper: cannot locate underlying MiniGrid env")

    def reset(self, *, seed=None, options=None):
        # Forward reset to wrapped env, including seed/options
        obs, info = self.env.reset(seed=seed, options=options)

        # If the user passed specific start-state, enforce it on the base env
        if options and 'agent_pos' in options and 'agent_dir' in options:
            pos, d = options['agent_pos'], options['agent_dir']
            self.minigrid_env.agent_pos = np.array(pos)
            self.minigrid_env.agent_dir = d
            # Regenerate the raw observation so subsequent wrappers see the new state
            if hasattr(self.minigrid_env, 'gen_obs'):
                raw = self.minigrid_env.gen_obs()
                if isinstance(obs, dict):
                    obs.update(raw)
                else:
                    obs = raw

        # Sync position/dir into info for logging
        self._sync_agent_info(info)
        return obs, info

    def step(self, action):
        # Step through entire wrapper stack
        obs, reward, terminated, truncated, info = self.env.step(action)
        # After step, sync the real base env's pos/dir
        self._sync_agent_info(info)
        return obs, reward, terminated, truncated, info

    def _sync_agent_info(self, info):
        """
        Write current agent_pos and agent_dir from the base MiniGrid env into info['log_data']
        """
        pos = tuple(self.minigrid_env.agent_pos)
        d = self.minigrid_env.agent_dir
        log_data = info.setdefault('log_data', {})
        log_data['agent_pos'] = pos
        log_data['agent_dir'] = d

############################################################################
    
    def _update_obs_with_position(self, obs, info):
        """
        Update observation data based on current agent position.
        Recalculates key features like goal direction, angle alignment, etc.
        """
        if 'log_data' not in info or not isinstance(info['log_data'], dict):
            return
        
        log_data = info['log_data']
        
        # Record agent position in log data for debugging
        log_data['agent_pos'] = self._agent_pos
        log_data['agent_dir'] = self._agent_dir
        
        # Only proceed if we have the MiniGrid environment and necessary grid information
        if self.minigrid_env is None:
            return
        
        # Get the goal position from the grid
        goal_pos = self._find_goal_position()
        if goal_pos is None:
            return
        
        # Recalculate goal-related features
        self._recalculate_goal_features(log_data, goal_pos)
        
        # Recalculate masks based on agent's view
        self._recalculate_masks(log_data)
        
        # Update the MLP input in the observation if present
        if isinstance(obs, dict) and 'MLP_input' in obs:
            self._update_mlp_input(obs, log_data)
    
    def _find_goal_position(self):
        """Find the goal position in the grid."""
        if not hasattr(self.minigrid_env, 'grid'):
            return None
            
        grid = self.minigrid_env.grid
        for i in range(grid.width):
            for j in range(grid.height):
                cell = grid.get(i, j)
                if cell is not None and cell.type == 'goal':
                    return (i, j)
        
        return None
    
    def _recalculate_goal_features(self, log_data, goal_pos):
        """Recalculate goal direction and angle alignment features."""
        # Calculate goal vector (from agent to goal)
        agent_pos = self._agent_pos
        agent_dir = self._agent_dir
        
        # Vector from agent to goal
        goal_vector = np.array([goal_pos[0] - agent_pos[0], goal_pos[1] - agent_pos[1]])
        goal_distance = np.linalg.norm(goal_vector)
        
        # Normalize goal vector if not zero
        if goal_distance > 0:
            goal_direction_vector = goal_vector / goal_distance
        else:
            goal_direction_vector = np.zeros(2)
        
        # Calculate goal angle
        goal_angle = math.atan2(goal_vector[1], goal_vector[0])
        if goal_angle < 0:
            goal_angle += 2 * math.pi
        
        # Calculate goal rotation (relative to agent's direction)
        # Agent directions: 0=right, 1=down, 2=left, 3=up
        agent_angle = agent_dir * (math.pi / 2)
        goal_rotation = (goal_angle - agent_angle) % (2 * math.pi)
        
        # Calculate four-way goal direction
        # This gives direction components in order: right, down, left, up
        four_way_goal_direction = np.zeros(4)
        
        # Simple version: Use normalized x,y components projected to each direction
        if goal_direction_vector[0] > 0:  # Right component
            four_way_goal_direction[0] = goal_direction_vector[0]
        if goal_direction_vector[1] > 0:  # Down component
            four_way_goal_direction[1] = goal_direction_vector[1]
        if goal_direction_vector[0] < 0:  # Left component
            four_way_goal_direction[2] = -goal_direction_vector[0]
        if goal_direction_vector[1] < 0:  # Up component
            four_way_goal_direction[3] = -goal_direction_vector[1]
        
        # Calculate four-way angle alignment
        # How aligned the agent is with each of the four directions
        four_way_angle_alignment = np.zeros(4)
        
        # Calculate alignment with each direction
        # The closer the goal is to the agent's current direction, the higher the value
        for i in range(4):
            dir_angle = i * (math.pi / 2)
            angle_diff = min((agent_angle - dir_angle) % (2 * math.pi), (dir_angle - agent_angle) % (2 * math.pi))
            alignment = max(0, 1 - (angle_diff / (math.pi / 2)))
            four_way_angle_alignment[i] = alignment
        
        # Add calculated features to log data
        log_data['goal_angle'] = goal_angle
        log_data['goal_rotation'] = goal_rotation
        log_data['goal_distance'] = goal_distance
        log_data['goal_direction_vector'] = goal_direction_vector
        log_data['four_way_goal_direction'] = four_way_goal_direction
        log_data['four_way_angle_alignment'] = four_way_angle_alignment
    
    def _recalculate_masks(self, log_data):
        """Recalculate vision masks based on agent position and orientation."""
        if not hasattr(self.minigrid_env, 'grid'):
            return
            
        grid = self.minigrid_env.grid
        window_size = 7  # Typical window size for MiniGrid
        
        # Create empty masks
        barrier_mask = np.zeros((window_size, window_size), dtype=np.int8)
        lava_mask = np.zeros((window_size, window_size), dtype=np.int8)
        goal_mask = np.zeros((window_size, window_size), dtype=np.int8)
        wall_mask = np.zeros((window_size, window_size), dtype=np.int8)
        empty_mask = np.zeros((window_size, window_size), dtype=np.int8)
        
        # Center coordinates in the window
        center_x, center_y = window_size // 2, window_size // 2
        
        # Fill masks based on surrounding cells
        half_size = window_size // 2
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                # Grid coordinates
                grid_x = self._agent_pos[0] + dx
                grid_y = self._agent_pos[1] + dy
                
                # Window coordinates (relative to center)
                win_x = center_x + dx
                win_y = center_y + dy
                
                # Check if within grid bounds
                if 0 <= grid_x < grid.width and 0 <= grid_y < grid.height:
                    cell = grid.get(grid_x, grid_y)
                    
                    if cell is None:
                        empty_mask[win_y, win_x] = 1
                    elif cell.type == 'wall':
                        wall_mask[win_y, win_x] = 1
                        barrier_mask[win_y, win_x] = 1
                    elif cell.type == 'lava':
                        lava_mask[win_y, win_x] = 1
                    elif cell.type == 'goal':
                        goal_mask[win_y, win_x] = 1
                else:
                    # Out of bounds is treated as barrier
                    barrier_mask[win_y, win_x] = 1
        
        # Update log data with new masks
        log_data['barrier_mask'] = barrier_mask
        log_data['lava_mask'] = lava_mask
        log_data['goal_mask'] = goal_mask
        log_data['wall_mask'] = wall_mask
        log_data['empty_mask'] = empty_mask
        
        # Set goal_bool based on goal presence in the view
        log_data['goal_bool'] = 1 if np.any(goal_mask) else 0
    
    def _update_mlp_input(self, obs, log_data):
        """Update the MLP_input in the observation based on recalculated features."""
        # Get the keys from the log_data that should be included in MLP_input
        mlp_keys = self._get_mlp_keys()
        
        # If we can't determine the keys, don't try to update
        if not mlp_keys:
            return
            
        # Flatten and concatenate the values for each key
        mlp_inputs = []
        for key in mlp_keys:
            if key in log_data:
                mlp_inputs.append(np.ravel(log_data[key]))
        
        # If we have inputs, update the MLP_input
        if mlp_inputs:
            obs['MLP_input'] = np.concatenate(mlp_inputs, axis=0)
            print(f"PositionAwareWrapper: Updated MLP_input with shape {obs['MLP_input'].shape}")
    
    def _get_mlp_keys(self):
        """Determine which keys should be included in the MLP_input."""
        # Try to find the mlp_keys from the environment
        env = self.env
        while hasattr(env, 'env'):
            if hasattr(env, 'mlp_keys'):
                return env.mlp_keys
            env = env.env
        
        # Default set of keys commonly used
        return ['four_way_goal_direction', 'four_way_angle_alignment', 'barrier_mask', 'lava_mask'] 