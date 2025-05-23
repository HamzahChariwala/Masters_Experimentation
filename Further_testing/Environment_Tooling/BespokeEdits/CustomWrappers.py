import numpy as np
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import random
import gymnasium as gym


class GoalAngleDistanceWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Extend the observation space
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            'goal_angle': spaces.Box(low=0.0, high=np.pi, shape=(), dtype=np.float32),
            'goal_rotation': spaces.MultiBinary(2),  # [left, right]
            'goal_distance': spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
            'goal_direction_vector': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'four_way_goal_direction': spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),  # [right, left, down, up]
            'four_way_angle_alignment': spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),  # [aligned, left_turn, right_turn, reverse]
            'orientation_distance': spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)  # [forward, backward, leftward, rightward]
        })

    def observation(self, obs):
        # Agent's position and direction
        agent_pos = np.array(self.unwrapped.agent_pos)
        agent_dir = self.unwrapped.agent_dir

        # Find goal position
        grid = self.unwrapped.grid
        goal_pos = None
        for i in range(grid.width):
            for j in range(grid.height):
                cell = grid.get(i, j)
                if cell and cell.type == 'goal':
                    goal_pos = np.array([i, j])
                    break
            if goal_pos is not None:
                break

        if goal_pos is None:
            raise ValueError("Goal not found in the grid.")

        # Vector from agent to goal
        vec_to_goal = goal_pos - agent_pos
        vec_to_goal = vec_to_goal.astype(np.float32)

        # Agent's facing direction vector
        dir_vectors = {
            0: np.array([1, 0]),   # Right
            1: np.array([0, 1]),   # Down
            2: np.array([-1, 0]),  # Left
            3: np.array([0, -1])   # Up
        }
        facing_vec = dir_vectors[agent_dir].astype(np.float32)

        # Normalize vectors
        norm_vec_to_goal = np.linalg.norm(vec_to_goal)
        norm_facing_vec = np.linalg.norm(facing_vec)

        # Initialize default values
        angle = 0.0
        rotation = np.array([0, 0], dtype=np.uint8)
        direction_vector = np.array([0.0, 0.0], dtype=np.float32)
        four_way_goal_direction = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [right, left, down, up]
        four_way_angle_alignment = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Default to perfectly aligned
        orientation_distance = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [forward, backward, leftward, rightward]

        if norm_vec_to_goal != 0 and norm_facing_vec != 0:
            # Compute angle between facing vector and vector to goal
            dot_product = np.dot(facing_vec, vec_to_goal)
            angle = np.arccos(np.clip(dot_product / (norm_facing_vec * norm_vec_to_goal), -1.0, 1.0))

            # Determine rotation direction using cross product
            cross = facing_vec[0] * vec_to_goal[1] - facing_vec[1] * vec_to_goal[0]
            if cross > 0:
                rotation = np.array([1, 0], dtype=np.uint8)  # Left
            elif cross < 0:
                rotation = np.array([0, 1], dtype=np.uint8)  # Right
            else:
                # When cross product is zero, angle is either 0 or π
                if dot_product > 0:
                    angle = 0.0
                    rotation = np.array([0, 0], dtype=np.uint8)  # Aligned
                else:
                    angle = np.pi
                    rotation = np.array([1, 1], dtype=np.uint8)  # Default to Left

            # Normalize direction vector
            direction_vector = vec_to_goal / norm_vec_to_goal
            
            # Create four-way direction representation (non-negative values only)
            # Convert 2D direction vector to 4D non-negative representation
            # Order: [right, left, down, up]
            right = max(0, direction_vector[0])  # Positive x = right
            left = max(0, -direction_vector[0])  # Negative x = left
            down = max(0, direction_vector[1])   # Positive y = down
            up = max(0, -direction_vector[1])    # Negative y = up
            
            four_way_goal_direction = np.array([right, left, down, up], dtype=np.float32)
            
            # Create four-way angle alignment representation
            # Order: [aligned, left_turn, right_turn, reverse]
            # Define angle thresholds
            aligned_threshold = np.pi / 6     # Within ±30 degrees
            slight_turn_threshold = np.pi / 3  # Within ±60 degrees
            large_turn_threshold = 2 * np.pi / 3  # Within ±120 degrees
            
            # Calculate the component values based on the angle and rotation direction
            if angle <= aligned_threshold:
                # Mostly aligned with the goal
                aligned_value = 1.0 - (angle / aligned_threshold)
                
                # Slight turn needed (decreases as alignment increases)
                turn_value = angle / aligned_threshold
                
                if cross > 0:  # Left turn needed
                    four_way_angle_alignment = np.array([aligned_value, turn_value, 0.0, 0.0], dtype=np.float32)
                elif cross < 0:  # Right turn needed
                    four_way_angle_alignment = np.array([aligned_value, 0.0, turn_value, 0.0], dtype=np.float32)
                else:  # Perfect alignment
                    four_way_angle_alignment = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    
            elif angle <= slight_turn_threshold:
                # Moderate turn needed
                turn_intensity = (angle - aligned_threshold) / (slight_turn_threshold - aligned_threshold)
                
                if cross > 0:  # Left turn needed
                    four_way_angle_alignment = np.array([0.0, turn_intensity, 0.0, 0.0], dtype=np.float32)
                elif cross < 0:  # Right turn needed
                    four_way_angle_alignment = np.array([0.0, 0.0, turn_intensity, 0.0], dtype=np.float32)
                    
            elif angle <= large_turn_threshold:
                # Large turn needed
                turn_intensity = (angle - slight_turn_threshold) / (large_turn_threshold - slight_turn_threshold)
                
                if cross > 0:  # Left turn needed
                    four_way_angle_alignment = np.array([0.0, 1.0, 0.0, turn_intensity], dtype=np.float32)
                elif cross < 0:  # Right turn needed
                    four_way_angle_alignment = np.array([0.0, 0.0, 1.0, turn_intensity], dtype=np.float32)
                    
            else:
                # Almost completely reversed - prioritize the "reverse" component
                reverse_intensity = (angle - large_turn_threshold) / (np.pi - large_turn_threshold)
                
                if cross > 0:  # Left turn suggested for reversing
                    four_way_angle_alignment = np.array([0.0, 1.0 - reverse_intensity, 0.0, reverse_intensity], dtype=np.float32)
                elif cross < 0:  # Right turn suggested for reversing
                    four_way_angle_alignment = np.array([0.0, 0.0, 1.0 - reverse_intensity, reverse_intensity], dtype=np.float32)
                else:  # Exactly reversed
                    four_way_angle_alignment = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            
            # Create orientation-aware distance
            # Order: [forward, backward, leftward, rightward]
            
            # Define the agent's orientation vectors
            forward_vec = facing_vec  # Agent's forward direction
            backward_vec = -facing_vec  # Agent's backward direction
            
            # The left and right vectors are perpendicular to the forward direction
            # For 2D, this is a 90-degree rotation
            leftward_vec = np.array([-facing_vec[1], facing_vec[0]])  # 90° CCW
            rightward_vec = np.array([facing_vec[1], -facing_vec[0]])  # 90° CW
            
            # Project the goal vector onto each orientation vector
            # This gives us how far the goal is in each direction
            forward_proj = max(0.0, np.dot(vec_to_goal, forward_vec))
            backward_proj = max(0.0, np.dot(vec_to_goal, backward_vec))
            leftward_proj = max(0.0, np.dot(vec_to_goal, leftward_vec))
            rightward_proj = max(0.0, np.dot(vec_to_goal, rightward_vec))
            
            # Normalize these distances based on the grid dimensions
            # to ensure values are in [0,1] range
            w, h = self.unwrapped.width, self.unwrapped.height
            max_possible_distance = np.sqrt(w**2 + h**2)
            
            forward_norm = min(1.0, forward_proj / max_possible_distance)
            backward_norm = min(1.0, backward_proj / max_possible_distance)
            leftward_norm = min(1.0, leftward_proj / max_possible_distance)
            rightward_norm = min(1.0, rightward_proj / max_possible_distance)
            
            orientation_distance = np.array([
                forward_norm,
                backward_norm,
                leftward_norm,
                rightward_norm
            ], dtype=np.float32)
            
        # Euclidean distance
        distance = norm_vec_to_goal

        # Add new information to observation
        obs['goal_angle'] = angle
        obs['goal_rotation'] = rotation
        obs['goal_distance'] = distance
        obs['goal_direction_vector'] = direction_vector
        obs['four_way_goal_direction'] = four_way_goal_direction
        obs['four_way_angle_alignment'] = four_way_angle_alignment
        obs['orientation_distance'] = orientation_distance

        return obs


class PartialObsWrapper(ObservationWrapper):
    def __init__(self, env, n):
        super().__init__(env)
        self.n = n
        self.agent_view_pos = (n // 2, 0)  # Middle-left
        base_spaces = dict(env.observation_space.spaces)
        base_spaces.update({
            'wall_mask': spaces.Box(low=0, high=1, shape=(n, n), dtype=np.uint8),
            'empty_mask': spaces.Box(low=0, high=1, shape=(n, n), dtype=np.uint8),
            'barrier_mask': spaces.Box(low=0, high=1, shape=(n, n), dtype=np.uint8),
            'lava_mask': spaces.Box(low=0, high=1, shape=(n, n), dtype=np.uint8),
            'goal_mask': spaces.Box(low=0, high=1, shape=(n, n), dtype=np.uint8),
            'goal_bool': spaces.Discrete(2)
        })
        self.observation_space = spaces.Dict(base_spaces)

    def observation(self, obs):
        agent_pos = np.array(self.unwrapped.agent_pos)
        agent_dir = self.unwrapped.agent_dir
        grid = self.unwrapped.grid
        width, height = grid.width, grid.height

        wall_mask = np.zeros((self.n, self.n), dtype=np.uint8)
        empty_mask = np.zeros((self.n, self.n), dtype=np.uint8)
        lava_mask = np.zeros((self.n, self.n), dtype=np.uint8)
        goal_mask = np.zeros((self.n, self.n), dtype=np.uint8)
        barrier_mask = np.zeros((self.n, self.n), dtype=np.uint8)

        # Rotation matrix for each direction (standard CCW)
        rotations = {
            0: np.array([[1, 0], [0, 1]]),    # Right → no rotation
            1: np.array([[0, -1], [1, 0]]),   # Down → rotate right
            2: np.array([[-1, 0], [0, -1]]),  # Left → rotate 180°
            3: np.array([[0, 1], [-1, 0]])    # Up → rotate left
        }

        rotation_matrix = rotations[agent_dir]

        for i in range(self.n):
            for j in range(self.n):
                # Coordinates in view space
                rel_pos = np.array([j - self.agent_view_pos[1], i - self.agent_view_pos[0]])

                # Rotate to match agent orientation
                env_offset = rotation_matrix @ rel_pos
                env_coords = agent_pos + env_offset
                x, y = env_coords

                if 0 <= x < width and 0 <= y < height:
                    cell = grid.get(x, y)
                    if cell:
                        if cell.type == 'wall':
                            wall_mask[i, j] = 1
                        elif cell.type == 'empty':
                            empty_mask[i, j] = 1
                        elif cell.type == 'lava':
                            lava_mask[i, j] = 1
                        elif cell.type == 'goal':
                            goal_mask[i, j] = 1
                else:
                    # Treat out-of-bounds as walls
                    empty_mask[i, j] = 1

        barrier_mask = wall_mask + empty_mask
        contains_goal = int(bool(np.any(goal_mask == 1)))

        obs['wall_mask'] = wall_mask
        obs['empty_mask'] = empty_mask
        obs['lava_mask'] = lava_mask
        obs['goal_mask'] = goal_mask
        obs['barrier_mask'] = barrier_mask
        obs['goal_bool'] = contains_goal

        return obs

    
class ExtractAbstractGrid(ObservationWrapper):
    def __init__(self, env):
        env = FullyObsWrapper(env)
        super().__init__(env)

        # Ensure the observation space remains a Dict with the new 'new_image' key
        original_spaces = env.observation_space.spaces
        w, h = env.unwrapped.width, env.unwrapped.height
        original_spaces['new_image'] = Box(low=0, high=3, shape=(2, w, h), dtype=np.uint8)
        self.observation_space = spaces.Dict(original_spaces)

    def observation(self, obs):
        grid = obs['image']  # shape: (w, h, 3)
        w, h = grid.shape[:2]

        agent_layer = np.zeros((w, h), dtype=np.uint8)
        object_layer = np.zeros((w, h), dtype=np.uint8)

        for x in range(w):
            for y in range(h):
                obj_id = grid[x, y, 0]

                # Object layer
                if obj_id == 1:
                    object_layer[y, x] = 1
                elif obj_id == 8:
                    object_layer[y, x] = 3
                elif obj_id == 9:
                    object_layer[y, x] = 2

                # Agent layer
                if obj_id == 10:
                    direction = self.env.unwrapped.agent_dir
                    agent_layer[y, x] = direction + 1
                    object_layer[y, x] = 1

        out = np.stack([agent_layer, object_layer], axis=0)
        obs['new_image'] = out

        return obs



class PartialRGBObsWrapper(ObservationWrapper):
    def __init__(self, env, n):
        super().__init__(env)
        self.n = n
        self.agent_view_pos = (n // 2, 0)  # Agent is centered at middle-left
        
        self.observation_space = spaces.Dict(dict(env.observation_space.spaces))
        self.observation_space.spaces['rgb_partial'] = spaces.Box(
            low=0, high=255, shape=(n, n, 3), dtype=np.uint8
        )

        # Define RGB colors for different object types
        self.color_map = {
            'wall': np.array([100, 100, 100], dtype=np.uint8),
            'floor': np.array([200, 200, 200], dtype=np.uint8),
            'lava': np.array([255, 0, 0], dtype=np.uint8),
            'goal': np.array([0, 255, 0], dtype=np.uint8),
            'agent': np.array([0, 0, 255], dtype=np.uint8)
        }

        # Rotation matrices for each direction
        self.rotations = {
            0: np.array([[1, 0], [0, 1]]),    # Right
            1: np.array([[0, -1], [1, 0]]),   # Down
            2: np.array([[-1, 0], [0, -1]]),  # Left
            3: np.array([[0, 1], [-1, 0]])    # Up
        }

    def observation(self, obs):
        agent_pos = np.array(self.unwrapped.agent_pos)
        agent_dir = self.unwrapped.agent_dir
        grid = self.unwrapped.grid
        width, height = grid.width, grid.height

        rotation_matrix = self.rotations[agent_dir]
        rgb_obs = np.zeros((self.n, self.n, 3), dtype=np.uint8)

        for i in range(self.n):
            for j in range(self.n):
                # Relative position in the agent's view
                rel_pos = np.array([j - self.agent_view_pos[1], i - self.agent_view_pos[0]])
                # Rotate to align with agent's orientation
                env_offset = rotation_matrix @ rel_pos
                env_coords = agent_pos + env_offset
                x, y = env_coords

                if 0 <= x < width and 0 <= y < height:
                    cell = grid.get(x, y)
                    if cell is None:
                        color = self.color_map['floor']
                    elif cell.type == 'wall':
                        color = self.color_map['wall']
                    elif cell.type == 'lava':
                        color = self.color_map['lava']
                    elif cell.type == 'goal':
                        color = self.color_map['goal']
                    else:
                        color = self.color_map['floor']
                else:
                    # Out-of-bounds treated as walls
                    color = self.color_map['wall']

                rgb_obs[j, i] = color

        # Mark the agent's position
        rgb_obs[self.agent_view_pos] = self.color_map['agent']
        rgb_obs = rgb_obs.transpose(2, 0, 1)  # Converts from (H, W, C) to (C, H, W)
        obs['rgb_partial'] = rgb_obs

        return obs
    


class PartialGrayObsWrapper(ObservationWrapper):
    def __init__(self, env, n):
        super().__init__(env)
        self.n = n
        self.agent_view_pos = (n // 2, 0)  # Agent is centered at middle-left
        
        self.observation_space = spaces.Dict(dict(env.observation_space.spaces))
        self.observation_space.spaces['grey_partial'] = spaces.Box(
            low=0, high=255, shape=(n, n, 1), dtype=np.uint8
        )

        # Define grayscale intensities for different object types
        self.gray_map = {
            'floor': 50,
            'agent': 200,
            'wall': 100,
            'lava': 150,
            'goal': 255
        }

        # Rotation matrices for each direction
        self.rotations = {
            0: np.array([[1, 0], [0, 1]]),    # Right
            1: np.array([[0, -1], [1, 0]]),   # Down
            2: np.array([[-1, 0], [0, -1]]),  # Left
            3: np.array([[0, 1], [-1, 0]])    # Up
        }

    def observation(self, obs):
        agent_pos = np.array(self.unwrapped.agent_pos)
        agent_dir = self.unwrapped.agent_dir
        grid = self.unwrapped.grid
        width, height = grid.width, grid.height

        rotation_matrix = self.rotations[agent_dir]
        gray_obs = np.zeros((self.n, self.n), dtype=np.uint8)

        for i in range(self.n):
            for j in range(self.n):
                # Relative position in the agent's view
                rel_pos = np.array([j - self.agent_view_pos[1], i - self.agent_view_pos[0]])
                # Rotate to align with agent's orientation
                env_offset = rotation_matrix @ rel_pos
                env_coords = agent_pos + env_offset
                x, y = env_coords

                if 0 <= x < width and 0 <= y < height:
                    cell = grid.get(x, y)
                    if cell is None:
                        gray = self.gray_map['floor']
                    elif cell.type == 'wall':
                        gray = self.gray_map['wall']
                    elif cell.type == 'lava':
                        gray = self.gray_map['lava']
                    elif cell.type == 'goal':
                        gray = self.gray_map['goal']
                    else:
                        gray = self.gray_map['floor']
                else:
                    # Out-of-bounds treated as walls
                    gray = self.gray_map['wall']

                gray_obs[j, i] = gray

        # Mark the agent's position
        gray_obs[self.agent_view_pos] = self.gray_map['agent']

        # Add channel dimension to match (1, H, W) format
        gray_obs = np.expand_dims(gray_obs, axis=0)
        obs['grey_partial'] = gray_obs

        return obs
    

class ForceFloat32(ObservationWrapper):
    def observation(self, obs):
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and v.dtype != np.float32:
                obs[k] = v.astype(np.float32)
        return obs


class RandomSpawnWrapper(gym.Wrapper):
    """
    Wrapper that randomizes the agent's starting position in the environment.
    
    The agent can spawn at any empty cell in the grid.
    
    Parameters:
    ----------
    env : gym.Env
        The environment to wrap
    exclude_goal_adjacent : bool
        If True, prevents spawning directly adjacent to the goal
    env_id : int
        Identifier for this environment instance (for logging)
    """
    
    def __init__(self, env, exclude_goal_adjacent=True, env_id=0):
        super().__init__(env)
        self.exclude_goal_adjacent = exclude_goal_adjacent
        self.env_id = env_id
        
    def reset(self, **kwargs):
        # First reset the environment
        obs, info = self.env.reset(**kwargs)
        
        # Get the grid
        grid = self.unwrapped.grid
        width, height = grid.width, grid.height
        
        # Find goal position if we're excluding goal-adjacent cells
        goal_pos = None
        if self.exclude_goal_adjacent:
            for i in range(width):
                for j in range(height):
                    cell = grid.get(i, j)
                    if cell and cell.type == 'goal':
                        goal_pos = (i, j)
                        break
                if goal_pos:
                    break
        
        # Find all empty cells
        empty_cells = []
        for i in range(width):
            for j in range(height):
                cell = grid.get(i, j)
                # Check if cell is empty (None or empty type)
                if cell is None or (hasattr(cell, 'type') and cell.type == 'empty'):
                    # Skip cells adjacent to goal if requested
                    if self.exclude_goal_adjacent and goal_pos:
                        # Check if this cell is adjacent to the goal
                        dx = abs(i - goal_pos[0])
                        dy = abs(j - goal_pos[1])
                        if dx <= 1 and dy <= 1:
                            continue
                    empty_cells.append((i, j))
        
        if empty_cells:
            # Randomly select an empty cell
            pos = random.choice(empty_cells)
            
            # Set agent position
            self.unwrapped.agent_pos = pos
            
            # Update agent's position in the grid
            self.unwrapped.grid.set(*self.unwrapped.agent_pos, None)
            
            # Recalculate the observation
            if hasattr(self.unwrapped, 'gen_obs'):
                # MiniGrid environments use gen_obs()
                obs = self.unwrapped.gen_obs()
            elif hasattr(self.unwrapped, '_get_obs'):
                # Some environments use _get_obs()
                obs = self.unwrapped._get_obs()
            else:
                # If neither method exists, re-reset the environment
                self.unwrapped.reset()
                obs, _ = self.env.reset()
            
        return obs, info


class DiagonalMoveMonitor(gym.Wrapper):
    """
    A wrapper that monitors the usage of diagonal moves.
    """
    def __init__(self, env):
        super().__init__(env)
        self.reset_stats()
        self.episode_count = 0
        self.episode_history = []
        
    def reset_stats(self):
        self.total_steps = 0
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # Count of each action type
        self.diagonal_steps = 0
        self.successful_diagonal_steps = 0
        self.left_diagonal_steps = 0
        self.right_diagonal_steps = 0
        
    def step(self, action):
        # Count the raw action before it's processed
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.total_steps += 1
        
        # Check if this was a diagonal move
        if "action" in info and info["action"] == "diagonal":
            self.diagonal_steps += 1
            
            # Check if it was successful
            if "failed" not in info or not info["failed"]:
                self.successful_diagonal_steps += 1
                
                # Track direction
                if info["diag_direction"] == "left":
                    self.left_diagonal_steps += 1
                elif info["diag_direction"] == "right":
                    self.right_diagonal_steps += 1
                    
        if terminated or truncated:
            # Print statistics at the end of each episode
            if self.total_steps > 0:
                success_rate = 0
                if self.diagonal_steps > 0:
                    success_rate = (self.successful_diagonal_steps / self.diagonal_steps) * 100
                
                # Store episode statistics
                self.episode_count += 1
                episode_stats = {
                    "episode": self.episode_count,
                    "total_steps": self.total_steps,
                    "action_counts": dict(self.action_counts),
                    "diagonal_attempts": self.diagonal_steps,
                    "diagonal_success": self.successful_diagonal_steps,
                    "success_rate": success_rate,
                    "left_diagonal": self.left_diagonal_steps,
                    "right_diagonal": self.right_diagonal_steps
                }
                self.episode_history.append(episode_stats)
                
                # Only print every 10 episodes to avoid too much output
                if self.episode_count % 10 == 0:
                    print("\n===== Diagonal Move Statistics (Episode {}) =====".format(self.episode_count))
                    print(f"Total steps: {self.total_steps}")
                    print(f"Action counts: {self.action_counts}")
                    print(f"Diagonal attempts: {self.diagonal_steps} ({self.diagonal_steps/self.total_steps*100:.1f}%)")
                    if self.diagonal_steps > 0:
                        print(f"Successful diagonal moves: {self.successful_diagonal_steps} ({success_rate:.1f}%)")
                        print(f"Left diagonal: {self.left_diagonal_steps}, Right diagonal: {self.right_diagonal_steps}")
                    print("==================================\n")
                
                # Reset stats for next episode
                self.reset_stats()
                
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        
    def get_episode_history(self):
        """Return the full history of episode statistics"""
        return self.episode_history


