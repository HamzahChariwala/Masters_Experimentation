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
        # Here, we simulate a diagonal move as a combination.
        diag_offset = forward_vec + lateral_vec
        new_pos = self.env.agent_pos + diag_offset
        # Use env's own _is_walkable if available.
        walkable = self.env._is_walkable(new_pos) if hasattr(self.env, "_is_walkable") else True
        
        # Debug/log print.
        print(f"Diagonal move: {diag_direction}, new_pos: {new_pos}, walkable: {walkable}")
        
        if walkable:
            # Update underlying state similar to CustomDiagonalEmptyEnv.
            self.env.agent_pos = new_pos.copy().astype(int)
            info = {"action": "diagonal", "diag_direction": diag_direction}
        else:
            info = {"action": "diagonal", "diag_direction": diag_direction, "failed": True}
        if hasattr(self.env, "step_count"):
            self.env.step_count += 1
        terminated = hasattr(self.env, "max_steps") and (self.env.step_count >= self.env.max_steps)
        truncated = False
        # Use the environment's own observation generation.
        obs = self.env.gen_obs() if hasattr(self.env, "gen_obs") else self.env.observation_space.sample()
        reward = 0
        return obs, reward, terminated, truncated, info

    def step(self, action):
        if action < 3:
            # Preserve original actions: 
            # 0: forward, 1: turn left, 2: turn right.
            return self.env.step(action)
        else:
            diag_direction = "left" if action == 3 else "right"
            return self._diagonal_move(diag_direction)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
