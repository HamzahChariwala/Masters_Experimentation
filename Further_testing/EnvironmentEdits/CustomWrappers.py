import numpy as np
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box


class GoalAngleDistanceWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Extend the observation space
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            'goal_angle': spaces.Box(low=0.0, high=np.pi, shape=(), dtype=np.float32),
            'goal_rotation': spaces.MultiBinary(2),  # [left, right]
            'goal_distance': spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
            'goal_direction_vector': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
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

        # Euclidean distance
        distance = norm_vec_to_goal

        # Add new information to observation
        obs['goal_angle'] = angle
        obs['goal_rotation'] = rotation
        obs['goal_distance'] = distance
        obs['goal_direction_vector'] = direction_vector

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
            1: np.array([[0, 1], [-1, 0]]),   # Down → rotate right
            2: np.array([[-1, 0], [0, -1]]),  # Left → rotate 180°
            3: np.array([[0, -1], [1, 0]])    # Up → rotate left
        }

        rotation_matrix = rotations[agent_dir]

        for i in range(self.n):
            for j in range(self.n):
                # Coordinates in view space
                rel_pos = np.array([i - self.agent_view_pos[0], j - self.agent_view_pos[1]])

                # Rotate to match agent orientation
                env_offset = rotation_matrix @ rel_pos
                env_coords = agent_pos + env_offset
                x, y = env_coords

                if 0 <= x < width and 0 <= y < height:
                    cell = grid.get(y, x)
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
        w, h = env.width, env.height
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
                    direction = self.env.agent_dir
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
            1: np.array([[0, 1], [-1, 0]]),   # Down
            2: np.array([[-1, 0], [0, -1]]),  # Left
            3: np.array([[0, -1], [1, 0]])    # Up
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
                rel_pos = np.array([i - self.agent_view_pos[0], j - self.agent_view_pos[1]])
                # Rotate to align with agent's orientation
                env_offset = rotation_matrix @ rel_pos
                env_coords = agent_pos + env_offset
                x, y = env_coords

                if 0 <= x < width and 0 <= y < height:
                    cell = grid.get(y, x)
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

                rgb_obs[i, j] = color

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
            1: np.array([[0, 1], [-1, 0]]),   # Down
            2: np.array([[-1, 0], [0, -1]]),  # Left
            3: np.array([[0, -1], [1, 0]])    # Up
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
                rel_pos = np.array([i - self.agent_view_pos[0], j - self.agent_view_pos[1]])
                # Rotate to align with agent's orientation
                env_offset = rotation_matrix @ rel_pos
                env_coords = agent_pos + env_offset
                x, y = env_coords

                if 0 <= x < width and 0 <= y < height:
                    cell = grid.get(y, x)
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

                gray_obs[i, j] = gray

        # Mark the agent's position
        gray_obs[self.agent_view_pos] = self.gray_map['agent']

        # Add channel dimension to match (1, H, W) format
        gray_obs = np.expand_dims(gray_obs, axis=0)
        obs['grey_partial'] = gray_obs

        return obs


