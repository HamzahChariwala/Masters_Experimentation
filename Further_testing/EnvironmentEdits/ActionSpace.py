import numpy as np
import gymnasium as gym

class CustomActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # New action space: 3 preserved + 2 diagonal actions.
        self.action_space = gym.spaces.Discrete(5)
    
    def _diagonal_move(self, diag_direction: str):
        # Compute vectors similar to CustomDiagonalEmptyEnv.
        direction_vecs = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }
        forward_vec = direction_vecs[self.env.agent_dir]
        left_vec = np.array([-forward_vec[1], forward_vec[0]])
        right_vec = np.array([forward_vec[1], -forward_vec[0]])
        lateral_vec = left_vec if diag_direction == "left" else right_vec
        diag_offset = forward_vec + lateral_vec
        new_pos = self.env.agent_pos + diag_offset

        # Ensure the new position is within bounds and walkable.
        if hasattr(self.env, 'width') and hasattr(self.env, 'height'):
            if not (0 <= new_pos[0] < self.env.width and 0 <= new_pos[1] < self.env.height):
                return self.env.gen_obs(), 0, False, False, {"action": "diagonal", "diag_direction": diag_direction, "failed": True}

        walkable = self.env._is_walkable(new_pos) if hasattr(self.env, "_is_walkable") else True

        if walkable:
            self.env.agent_pos = new_pos.copy().astype(int)
            info = {"action": "diagonal", "diag_direction": diag_direction}
            print(f"Diagonal move taken: {diag_direction}, New Position: {self.env.agent_pos}")
        else:
            info = {"action": "diagonal", "diag_direction": diag_direction, "failed": True}

        if hasattr(self.env, "step_count"):
            self.env.step_count += 1

        terminated = hasattr(self.env, "max_steps") and (self.env.step_count >= self.env.max_steps)
        truncated = False
        obs = self.env.gen_obs() if hasattr(self.env, "gen_obs") else self.env.observation_space.sample()
        reward = 0.1 if walkable else 0
        return obs, reward, terminated, truncated, info

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
    


class FlattenMultiDiscrete(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # capture the original branch sizes (e.g. [5,5,...,5])
        self.nvec = self.env.action_space.nvec
        # flatten into one big Discrete space
        self.action_space = gym.spaces.Discrete(int(np.prod(self.nvec)))

    def action(self, action: int):
        """
        Convert flat index → multi-discrete tuple,
        e.g. 1234 → (a0, a1, ..., a9) each in 0–4
        """
        return np.unravel_index(action, self.nvec)

    def reverse_action(self, action):
        """
        (Optional) Convert a multi-discrete vector back to a flat index,
        if you ever need to inspect the env’s raw action.
        """
        return int(np.ravel_multi_index(action, self.nvec))
