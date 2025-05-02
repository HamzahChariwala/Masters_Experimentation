import gymnasium as gym
import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Wall
from minigrid.envs.empty import EmptyEnv
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt

from minigrid.envs.empty import EmptyEnv
from gymnasium.envs.registration import register

class CustomDiagonalEmptyEnv(EmptyEnv):
    def __init__(self, diag_stochastic=(0.1, 0.1), preserved_action_indices=[0, 1, 2], **kwargs):
        """
        diag_stochastic: Tuple (p_forward, p_side) for stochastic outcomes when taking a diagonal action.
                         The intended diagonal move is executed with probability 1 - (p_forward + p_side).
        preserved_action_indices: List of indices from the base action space that are retained (e.g., navigation: turn left, right, forward).
        kwargs: Additional keyword arguments for EmptyEnv (grid layout, max_steps, etc.) will be taken from the built-in defaults.
        """
        # Initialize the base EmptyEnv (which already sets up its mission and grid)
        super().__init__(**kwargs)
        
        # Store parameters for our custom diagonal logic.
        self.diag_stochastic = diag_stochastic
        self.p_forward, self.p_side = diag_stochastic
        self.p_success = 1 - (self.p_forward + self.p_side)
        self.preserved_action_indices = preserved_action_indices
        self.num_preserved_actions = len(preserved_action_indices)
        
        # We add two new actions for diagonal moves.
        self.num_new_actions = 2
        self.custom_action_space_size = self.num_preserved_actions + self.num_new_actions
        # Override the base action_space.
        self.action_space = gym.spaces.Discrete(self.custom_action_space_size)

        # Override observation_space to match gen_obs outputs.
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
            "direction": gym.spaces.Discrete(4),
            "mission": hasattr(gym.spaces, "Text") and gym.spaces.Text(max_length=100) or gym.spaces.Text(max_length=100)
        })
    
    def step(self, action):
        # If the action is one of the preserved ones, map it back to the original action.
        if action < self.num_preserved_actions:
            orig_action = self.preserved_action_indices[action]
            return super().step(orig_action)
        else:
            # For new actions (diagonal moves): index = num_preserved_actions -> diagonal left, 
            # index = num_preserved_actions+1 -> diagonal right.
            diag_type = action - self.num_preserved_actions  # 0 for left, 1 for right.
            return self._execute_diagonal(diag_type)
    
    def _execute_diagonal(self, diag_type):
        # Stochastically determine the outcome.
        r = np.random.rand()
        if r < self.p_forward:
            move_type = "forward"
        elif r < self.p_forward + self.p_side:
            move_type = "lateral"
        else:
            move_type = "diagonal"
        
        # Compute directional vectors based on self.agent_dir.
        # Convention: 0 = right, 1 = down, 2 = left, 3 = up.
        direction_vecs = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }
        forward_vec = direction_vecs[self.agent_dir]
        
        # Lateral vectors:
        left_vec = np.array([-forward_vec[1], forward_vec[0]])
        right_vec = np.array([forward_vec[1], -forward_vec[0]])
        lateral_vec = left_vec if diag_type == 0 else right_vec
        
        # Determine the offset.
        if move_type == "forward":
            offset = forward_vec
        elif move_type == "lateral":
            offset = lateral_vec
        else:
            offset = forward_vec + lateral_vec
        
        new_pos = self.agent_pos + offset
        diag_direction = "left" if diag_type == 0 else "right"

        # Check if the intended new position is within the grid.
        if self._is_walkable(new_pos):
            self.agent_pos = new_pos.copy()
            reward = 0
            info = {"action": "diagonal", "move_type": move_type, "diag_direction": diag_direction}
        else:
            # If move is invalid, do not move the agent.
            reward = 0
            info = {"action": "diagonal", "move_type": move_type, "diag_direction": diag_direction, "failed": True}
        
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        obs = self.gen_obs()
        return obs, reward, terminated, truncated, info
    
    def _is_walkable(self, pos):
        # Ensure pos is an integer position.
        pos = np.array(pos, dtype=int)
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return True

    def safe_encode(self):
        """
        Encode the grid safely. For out-of-bound lookups or cells that are None,
        return a default encoding (here, using zeros).
        """
        encoded = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for j in range(self.height):
            for i in range(self.width):
                cell = self.grid.get(i, j)
                if cell is None:
                    # Return a default encoding for empty cells; adjust the values if needed.
                    encoded[j, i, :] = np.array([0, 0, 0], dtype=np.uint8)
                else:
                    try:
                        encoded[j, i, :] = cell.encode()
                    except Exception as e:
                        # In case of any encoding errors, fallback to the default.
                        encoded[j, i, :] = np.array([0, 0, 0], dtype=np.uint8)
        return encoded

    def gen_obs(self):
        """
        Override gen_obs to use a safe version of grid encoding.
        """
        return {
            'image': self.safe_encode(),
            'direction': self.agent_dir,
            'mission': str(self.mission)  # Cast mission to str.
        }

    # Add a render override to accept arbitrary keyword arguments.
    def render(self, **kwargs):
        # Remove unsupported kwargs to avoid issues.
        kwargs.pop('render_mode', None)
        return super().render()


# Register the custom environment.
register(
    id='MiniGrid-Empty-5x5-CustomDiagonal-v0',
    entry_point='__main__:CustomDiagonalEmptyEnv',
    max_episode_steps=100
)

def visualize_environment(env, render_mode="rgb_array"):
    """
    Visualize the current state of the environment for debugging.
    
    Parameters:
      env: The environment instance.
      render_mode: The render mode to request from env.render(). Default is "rgb_array".
                   Ensure that your environment is created with a render_mode that supports this.
    """
    try:
        # Retrieve the rendered image.
        img = env.render(render_mode=render_mode)
    except Exception as e:
        print("Error during rendering:", e)
        return

    # Plot the image using matplotlib.
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Environment Rendering")
    plt.show()

# =============================================================================
# Example usage:
# =============================================================================

if __name__ == "__main__":
    env = gym.make('MiniGrid-Empty-5x5-CustomDiagonal-v0')
    obs, info = env.reset(seed=42)
    print("Initial observation keys:", list(obs.keys()))
    print("Action space:", env.action_space)

    # Run a few steps to test.
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action taken: {action}, Reward: {reward}, Info: {info}")
        if terminated or truncated:
            break

    env = gym.make('MiniGrid-Empty-5x5-CustomDiagonal-v0', render_mode="rgb_array")
    env.reset(seed=42)
    
    # Visualize the initial state of the environment.
    visualize_environment(env)





# For instance, you can now register a custom environment that is based on "MiniGrid-Empty-5x5-v0"
# but with our new action set and stochastic diagonal moves.
# if __name__ == "__main__":    

#     register(
#         id='MiniGrid-Empty-5x5-CustomDiagonal-v0',
#         entry_point='__main__:CustomDiagonalActionEnv',
#         max_episode_steps=100,
#         kwargs={'diag_stochastic': (0.1, 0.1)}
#     )

#     # Create the environment.
#     env = gym.make('MiniGrid-Empty-5x5-CustomDiagonal-v0')
#     obs, info = env.reset(seed=42)
#     print("Initial observation keys:", list(obs.keys()))
#     print("Action space:", env.action_space)

#     # Sample a few actions.
#     for _ in range(5):
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)
#         print(f"Action taken: {action}, Reward: {reward}, Info: {info}")
#         if terminated or truncated:
#             break
