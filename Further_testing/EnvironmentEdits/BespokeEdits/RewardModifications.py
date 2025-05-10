import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class EpisodeCompletionRewardWrapper(gym.Wrapper):
    """
    A wrapper that modifies the reward when an episode ends successfully.
    The reward is based on the number of steps taken, with three possible functions:
    1. Linear: reward = max(0, y_intercept - (y_intercept/x_intercept) * num_steps)
    2. Exponential: reward = y_intercept / exp((-ln(0.01)/x_intercept) * num_steps)
    3. Sigmoid: reward = y_intercept / (1 + exp((4/transition_width) * (num_steps - x_intercept/2)))
    
    Parameters:
    -----------
    env : gym.Env
        The environment to wrap
    reward_type : str
        The type of reward function ('linear', 'exponential', or 'sigmoid')
    x_intercept : float
        Number of steps at which reward becomes approximately 0
    y_intercept : float
        Initial reward value at step 0
    transition_width : float
        For sigmoid function, controls how quickly the reward transitions from y_intercept to 0
    """
    
    def __init__(self, env, reward_type='linear', x_intercept=100, y_intercept=1.0, transition_width=10):
        super().__init__(env)
        self.reward_type = reward_type
        self.x_intercept = x_intercept
        self.y_intercept = y_intercept
        self.transition_width = transition_width
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
                # Linear: reward = max(0, y_intercept - (y_intercept/x_intercept) * num_steps)
                slope = self.y_intercept / self.x_intercept
                modified_reward = max(0, self.y_intercept - slope * self.step_count)
            elif self.reward_type == 'exponential':
                # Exponential: reward = y_intercept / exp((-ln(0.01)/x_intercept) * num_steps)
                slope = -np.log(0.01) / self.x_intercept
                modified_reward = self.y_intercept / np.exp(slope * self.step_count)
            elif self.reward_type == 'sigmoid':
                # Sigmoid: reward = y_intercept / (1 + exp((4/transition_width) * (num_steps - x_intercept/2)))
                slope = 4 / self.transition_width
                shift = self.x_intercept / 2
                modified_reward = self.y_intercept / (1 + np.exp(slope * (self.step_count - shift)))
            else:
                raise ValueError(f"Invalid reward type: {self.reward_type}")
            
            return obs, modified_reward, terminated, truncated, info
        
        return obs, reward, terminated, truncated, info

