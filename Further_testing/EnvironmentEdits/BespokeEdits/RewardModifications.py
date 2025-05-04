import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class EpisodeCompletionRewardWrapper(gym.Wrapper):
    """
    A wrapper that modifies the reward when an episode ends successfully.
    The reward is based on the number of steps taken, with three possible functions:
    1. Linear: 1 - m * num_steps
    2. Exponential: 1 / exp(m * num_steps)
    3. Sigmoid: 1 / (1 + exp(m * (num_steps - shift)))
    
    Parameters:
    -----------
    env : gym.Env
        The environment to wrap
    reward_type : str
        The type of reward function ('linear', 'exponential', or 'sigmoid')
    slope : float
        The slope (m) parameter that controls how quickly the reward decreases
    shift : float
        For sigmoid function, the shift parameter that controls the midpoint
    """
    
    def __init__(self, env, reward_type='linear', slope=0.01, shift=50):
        super().__init__(env)
        self.reward_type = reward_type
        self.slope = slope
        self.shift = shift
        self.step_count = 0
        
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Modify reward only when episode ends successfully
        if terminated and reward > 0:
            if self.reward_type == 'linear':
                # Linear: 1 - m * num_steps (clipped at 0)
                modified_reward = max(0, 1 - self.slope * self.step_count)
            elif self.reward_type == 'exponential':
                # Exponential: 1 / exp(m * num_steps)
                modified_reward = 1 / np.exp(self.slope * self.step_count)
            elif self.reward_type == 'sigmoid':
                # Sigmoid: 1 / (1 + exp(m * (num_steps - shift)))
                modified_reward = 1 / (1 + np.exp(self.slope * (self.step_count - self.shift)))
            else:
                raise ValueError(f"Invalid reward type: {self.reward_type}")
            
            return obs, modified_reward, terminated, truncated, info
        
        return obs, reward, terminated, truncated, info

