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
        
    def get_wrapper_attr(self, name):
        """
        Recursive attribute getter that searches through the wrapper stack.
        Used by SubprocVecEnv to fetch attributes from the environment.
        """
        if hasattr(self, name):
            return getattr(self, name)
        elif hasattr(self.env, 'get_wrapper_attr'):
            return self.env.get_wrapper_attr(name)
        elif hasattr(self.env, name):
            return getattr(self.env, name)
        
        # Try to recurse through env chain if nested
        if hasattr(self.env, 'env'):
            env = self.env.env
            while env:
                if hasattr(env, name):
                    return getattr(env, name)
                if not hasattr(env, 'env'):
                    break
                env = env.env
                
        # If we reach here, the attribute wasn't found
        raise AttributeError(f"'{type(self).__name__}' object and its wrappers have no attribute '{name}'")


