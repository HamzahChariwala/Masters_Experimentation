import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

class OldGymCompatibility(gym.Env):
    def __init__(self, env: gym.Env):
        super().__init__()
        # Wrap in Monitor to get episode info in `info["episode"]`
        self.env = Monitor(env)
        self.action_space      = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        # Gymnasium.reset() → (obs, info)
        obs, info = self.env.reset(**kwargs)
        # Return both so DummyVecEnv.reset() can unpack correctly
        return obs, info

    def step(self, action):
        # Call the underlying Gymnasium env
        obs, reward, terminated, truncated, info = self.env.step(action)
        # **Return all five**, not a 4-tuple!
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


