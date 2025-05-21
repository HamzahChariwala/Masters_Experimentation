import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from gymnasium.core import ObservationWrapper


class RobustMaskGenerator(ObservationWrapper):
    """
    A more robust implementation of partial observation mask generation.
    This wrapper addresses coordinate system issues by:
    
    1. Extracting a window of the grid centered on the agent
    2. Rotating the grid based on the agent's orientation
    3. Generating masks with a consistent coordinate system
    
    All masks are generated in agent-centric coordinates, where:
    - Forward is always at the top of the mask
    - The agent is always at the center of the mask
    """
    
    def __init__(self, env, view_size=5):
        super().__init__(env)
        
        # View size must be odd to have the agent at the center
        assert view_size % 2 == 1, "View size must be odd"
        self.view_size = view_size
        self.agent_view_center = view_size // 2
        
        # Update observation space to include our masks
        base_spaces = dict(env.observation_space.spaces)
        base_spaces.update({
            'wall_mask': spaces.Box(low=0, high=1, shape=(view_size, view_size), dtype=np.uint8),
            'empty_mask': spaces.Box(low=0, high=1, shape=(view_size, view_size), dtype=np.uint8),
            'lava_mask': spaces.Box(low=0, high=1, shape=(view_size, view_size), dtype=np.uint8),
            'goal_mask': spaces.Box(low=0, high=1, shape=(view_size, view_size), dtype=np.uint8),
            'barrier_mask': spaces.Box(low=0, high=1, shape=(view_size, view_size), dtype=np.uint8),
            'agent_dir': spaces.Discrete(4),
            'goal_visible': spaces.Discrete(2)
        })
        self.observation_space = spaces.Dict(base_spaces)
        
        # Rotation matrices for each direction (to transform from agent-centric to world)
        # Agent's forward direction is always at the top of the view
        self.rotations = {
            0: np.array([[0, 1], [-1, 0]]),  # Right → rotate 90° CCW
            1: np.array([[-1, 0], [0, -1]]), # Down → rotate 180°
            2: np.array([[0, -1], [1, 0]]),  # Left → rotate 90° CW
            3: np.array([[1, 0], [0, 1]])    # Up → no rotation (up is forward)
        }
    
    def _get_grid_window(self):
        """
        Extract a window of the grid centered on the agent.
        The window is rotated so the agent's forward direction is always up.
        
        Returns:
            dict: Various masks for different object types
        """
        agent_pos = np.array(self.unwrapped.agent_pos)
        agent_dir = self.unwrapped.agent_dir
        grid = self.unwrapped.grid
        width, height = grid.width, grid.height
        
        # Initialize masks
        wall_mask = np.zeros((self.view_size, self.view_size), dtype=np.uint8)
        empty_mask = np.zeros((self.view_size, self.view_size), dtype=np.uint8)
        lava_mask = np.zeros((self.view_size, self.view_size), dtype=np.uint8)
        goal_mask = np.zeros((self.view_size, self.view_size), dtype=np.uint8)
        
        # Get rotation matrix for current agent direction
        rotation_matrix = self.rotations[agent_dir]
        
        # Loop through all positions in our view window
        for i in range(self.view_size):
            for j in range(self.view_size):
                # Position relative to center (agent)
                rel_pos = np.array([i - self.agent_view_center, j - self.agent_view_center])
                
                # Rotate to match agent's orientation and convert to world coordinates
                world_offset = rotation_matrix @ rel_pos
                world_pos = agent_pos + world_offset
                
                # Check if position is within bounds
                if (0 <= world_pos[0] < width and 0 <= world_pos[1] < height):
                    # Get the cell at this world position - note x,y order
                    cell = grid.get(int(world_pos[0]), int(world_pos[1]))
                    
                    if cell:
                        if cell.type == 'wall':
                            wall_mask[i, j] = 1
                        elif cell.type == 'lava':
                            lava_mask[i, j] = 1
                        elif cell.type == 'goal':
                            goal_mask[i, j] = 1
                        elif cell.type == 'empty':
                            empty_mask[i, j] = 1
                    else:
                        # Empty cell (no object)
                        empty_mask[i, j] = 1
                else:
                    # Out of bounds - mark as wall
                    wall_mask[i, j] = 1
        
        barrier_mask = wall_mask.copy()
        
        # Create the agent mask - agent is always at the center
        result = {
            'wall_mask': wall_mask,
            'empty_mask': empty_mask,
            'lava_mask': lava_mask,
            'goal_mask': goal_mask,
            'barrier_mask': barrier_mask,
            'agent_dir': agent_dir,
            'goal_visible': int(np.any(goal_mask == 1))
        }
        
        return result
    
    def observation(self, obs):
        """
        Generate the observation by adding our masks to the existing observation.
        """
        # Add our masks to the observation
        masks = self._get_grid_window()
        for key, value in masks.items():
            obs[key] = value
        
        return obs


# Example usage
if __name__ == "__main__":
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RobustMaskGenerator(env)
    obs, _ = env.reset()
    
    # You can access the masks directly from the observation
    wall_mask = obs['wall_mask']
    lava_mask = obs['lava_mask']
    goal_mask = obs['goal_mask'] 