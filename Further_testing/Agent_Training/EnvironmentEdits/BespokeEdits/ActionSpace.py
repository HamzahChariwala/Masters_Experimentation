import numpy as np
import gymnasium as gym

class CustomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, diagonal_success_reward=1.5, diagonal_failure_penalty=0.1):
        super().__init__(env)
        # New action space: 3 preserved + 2 diagonal actions.
        self.action_space = gym.spaces.Discrete(5)
        # Configurable reward values
        self.diagonal_success_reward = diagonal_success_reward
        self.diagonal_failure_penalty = diagonal_failure_penalty
    
    def _is_walkable(self, pos):
        """
        Determine if a position is walkable using MiniGrid's logic.
        A position is walkable if it's within bounds and either has no object or the object can be overlapped.
        """
        grid = self.env.grid if hasattr(self.env, 'grid') else None
        if grid is None:
            return False
        
        # Check bounds
        if not (0 <= pos[0] < grid.width and 0 <= pos[1] < grid.height):
            return False
        
        # Get cell at position
        cell = grid.get(pos[0], pos[1])
        
        # Cell is walkable if it's None or can be overlapped
        return cell is None or cell.can_overlap()
    
    def _check_diagonal_walkability(self, agent_pos, forward_pos, lateral_pos, diagonal_pos):
        """
        General algorithm for determining if a diagonal move is valid.
        
        A diagonal move is valid if:
        1. The diagonal position itself is walkable
        2. Both adjacent positions (forward and lateral) are walkable
        
        This ensures the agent cannot "cut corners" through walls.
        """
        # If the diagonal position itself is not walkable, the move is invalid
        if not self._is_walkable(diagonal_pos):
            return False
            
        # Get walkability of adjacent positions
        forward_walkable = self._is_walkable(forward_pos)
        lateral_walkable = self._is_walkable(lateral_pos)
        
        # Only allow diagonal moves if both adjacent positions are walkable
        # This prevents diagonal moves through walls/corners
        return forward_walkable and lateral_walkable
    
    def _diagonal_move(self, diag_direction: str):
        # Define direction vectors for the four cardinal directions
        direction_vecs = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1])   # up
        }
        
        # Get the agent's current direction vector
        agent_dir = self.env.agent_dir
        agent_pos = self.env.agent_pos
        forward_vec = direction_vecs[agent_dir]
        
        # Calculate left and right vectors relative to the agent's direction
        left_vec = np.array([-forward_vec[1], forward_vec[0]])
        right_vec = np.array([forward_vec[1], -forward_vec[0]])
        
        # Define diagonal movement based on direction
        if diag_direction == "left":
            # Forward + left diagonal
            diag_vec = forward_vec + left_vec
            lateral_vec = left_vec
        else:  # Right diagonal
            # Forward + right diagonal
            diag_vec = forward_vec + right_vec
            lateral_vec = right_vec
        
        # Calculate new position and intermediate positions
        new_pos = agent_pos + diag_vec
        forward_pos = agent_pos + forward_vec
        lateral_pos = agent_pos + lateral_vec
        
        # Check if the diagonal move is valid
        is_valid_diagonal = self._check_diagonal_walkability(
            agent_pos, forward_pos, lateral_pos, new_pos
        )

        if is_valid_diagonal:
            # Convert position to integer numpy array to ensure correct update
            new_pos_int = new_pos.astype(int)
            
            # Find the base environment to update directly
            base_env = self.env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            # Update position in all environment layers
            self._update_agent_position(new_pos_int)
            
            info = {"action": "diagonal", "diag_direction": diag_direction}
        else:
            info = {"action": "diagonal", "diag_direction": diag_direction, "failed": True}
            # Return the current observation but with a configurable negative reward for failed diagonal move
            return self.env.gen_obs(), -self.diagonal_failure_penalty, False, False, info

        if hasattr(self.env, "step_count"):
            self.env.step_count += 1

        terminated = hasattr(self.env, "max_steps") and (self.env.step_count >= self.env.max_steps)
        truncated = False
        obs = self.env.gen_obs() if hasattr(self.env, "gen_obs") else self.env.observation_space.sample()
        
        # Give a configurable higher reward for successful diagonal moves
        reward = self.diagonal_success_reward if is_valid_diagonal else 0
        return obs, reward, terminated, truncated, info

    def _update_agent_position(self, new_pos):
        """
        Update the agent's position in all environment wrapper layers.
        This ensures consistent state across the environment stack.
        """
        # First update the immediate environment's agent position
        if isinstance(self.env.agent_pos, np.ndarray):
            self.env.agent_pos = new_pos.copy()
        else:
            self.env.agent_pos = tuple(new_pos)
            
        # Then recursively update all nested environments
        env = self.env
        while hasattr(env, 'env'):
            inner_env = env.env
            if hasattr(inner_env, 'agent_pos'):
                if isinstance(inner_env.agent_pos, np.ndarray):
                    inner_env.agent_pos = new_pos.copy()
                else:
                    inner_env.agent_pos = tuple(new_pos)
            env = inner_env

    def step(self, action):
        if action < 3:
            return self.env.step(action)
        else:
            diag_direction = "left" if action == 3 else "right"
            return self._diagonal_move(diag_direction)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def get_wrapper_attr(self, name):
        """
        Fallback implementation for `get_wrapper_attr` to ensure compatibility with SubprocVecEnv.
        """
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"'{type(self.env).__name__}' object has no attribute '{name}'")