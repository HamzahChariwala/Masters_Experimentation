import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class LavaStepCounterWrapper(gym.Wrapper):
    """
    A wrapper that counts the number of steps an agent spends in lava cells.
    
    This wrapper tracks:
    1. The total number of steps spent in lava
    2. Whether the agent is currently in lava
    
    Parameters:
    -----------
    env : gym.Env
        The environment to wrap
    lava_step_multiplier : float
        How many regular steps one lava step is considered equivalent to
    verbose : bool
        If True, prints basic information like initialization parameters
    debug_logging : bool
        If True, prints detailed step-by-step logging information
    """
    
    def __init__(self, env, lava_step_multiplier=2.0, verbose=True, debug_logging=False):
        super().__init__(env)
        self.lava_step_multiplier = lava_step_multiplier
        self.verbose = verbose
        self.debug_logging = debug_logging
        self.lava_steps = 0
        self.is_in_lava = False
        # Track this wrapper's own step count
        self.wrapper_step_count = 0
        
        if self.verbose:
            print(f"\nInitialized lava step counter with:")
            print(f"  Lava step multiplier: {lava_step_multiplier}")
            print(f"  Debug logging: {debug_logging}")
    
    def reset(self, **kwargs):
        # Reset lava step counter
        self.lava_steps = 0
        self.is_in_lava = False
        # Reset our internal step counter
        self.wrapper_step_count = 0
        
        obs, info = self.env.reset(**kwargs)
        
        # Check if agent starts in lava (rare but possible)
        self._update_lava_status(obs)
        
        # Add debug information in verbose mode
        if self.debug_logging:
            print(f"  [LavaStepCounter] Reset: lava_steps={self.lava_steps}, is_in_lava={self.is_in_lava}")
        
        return obs, info
    
    def step(self, action):
        # Execute action in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Increment our wrapper's step counter
        self.wrapper_step_count += 1
        
        # Update lava status based on new observation
        self._update_lava_status(obs)
        
        # Add lava steps information to info dict
        info['lava_steps'] = self.lava_steps
        info['is_in_lava'] = self.is_in_lava
        info['effective_steps'] = self.get_effective_steps()
        
        # Add debug information only if debug_logging is enabled
        if self.debug_logging and (self.is_in_lava or self.lava_steps > 0):
            print(f"  [LavaStepCounter] Step {self.wrapper_step_count}: lava_steps={self.lava_steps}, effective_steps={info['effective_steps']}")
        
        return obs, reward, terminated, truncated, info
    
    def _update_lava_status(self, obs):
        """Check if agent is currently standing in lava based on the observation."""
        # This assumes the MiniGrid convention where 'L' is lava
        grid = self.env.grid
        agent_pos = self.env.agent_pos
        
        if grid is not None and agent_pos is not None:
            cell = grid.get(*agent_pos)
            # Check if the cell is lava
            is_lava = cell is not None and cell.type == 'lava'
            
            # If this is a new lava step, increment the counter
            if is_lava:
                self.lava_steps += 1
                self.is_in_lava = True
            else:
                # Agent is not in lava
                self.is_in_lava = False
    
    def get_effective_steps(self):
        """
        Calculate effective steps considering lava penalty.
        A lava step counts as multiple regular steps based on the multiplier.
        
        USE OUR INTERNAL STEP COUNTER instead of relying on the base environment's counter
        which may not reset properly in all cases.
        """
        return self.wrapper_step_count + (self.lava_steps * self.lava_step_multiplier)


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
    count_lava_steps : bool
        If True, count lava steps with a penalty multiplier
    verbose : bool
        If True, prints debug information to the terminal
    """
    
    def __init__(self, env, reward_type='linear', x_intercept=100, y_intercept=1.0, 
                 transition_width=10, count_lava_steps=False, verbose=True):
        super().__init__(env)
        self.reward_type = reward_type
        self.x_intercept = x_intercept
        self.y_intercept = y_intercept
        self.transition_width = transition_width
        self.count_lava_steps = count_lava_steps
        self.verbose = verbose
        self.step_count = 0
        
        if self.verbose:
            print(f"\nInitialized reward wrapper with:")
            print(f"  Type: {reward_type}")
            print(f"  X-intercept: {x_intercept}")
            print(f"  Y-intercept: {y_intercept}")
            print(f"  Transition width: {transition_width}")
            print(f"  Count lava steps: {count_lava_steps}\n")
        
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Modify reward only when episode ends successfully
        if terminated and reward > 0:
            original_reward = reward
            
            # Determine effective steps (either regular steps or with lava penalty)
            effective_steps = self._get_effective_steps(info)
            
            if self.reward_type == 'linear':
                # Linear: reward = max(0, y_intercept - (y_intercept/x_intercept) * num_steps)
                slope = self.y_intercept / self.x_intercept
                modified_reward = max(0, self.y_intercept - slope * effective_steps)
            elif self.reward_type == 'exponential':
                # Exponential: reward = y_intercept / exp((-ln(0.01)/x_intercept) * num_steps)
                slope = -np.log(0.01) / self.x_intercept
                modified_reward = self.y_intercept / np.exp(slope * effective_steps)
            elif self.reward_type == 'sigmoid':
                # Sigmoid: reward = y_intercept / (1 + exp((4/transition_width) * (num_steps - x_intercept/2)))
                slope = 4 / self.transition_width
                shift = self.x_intercept / 2
                modified_reward = self.y_intercept / (1 + np.exp(slope * (effective_steps - shift)))
            else:
                raise ValueError(f"Invalid reward type: {self.reward_type}")
            
            if self.verbose:
                print(f"\nReward modification at step {self.step_count}:")
                print(f"  Original reward: {original_reward}")
                print(f"  Modified reward: {modified_reward}")
                print(f"  Steps taken: {self.step_count}")
                if self.count_lava_steps and 'lava_steps' in info:
                    print(f"  Lava steps: {info.get('lava_steps', 0)}")
                    print(f"  Effective steps: {effective_steps}")
                print(f"  Reward type: {self.reward_type}\n")
            
            return obs, modified_reward, terminated, truncated, info
        
        return obs, reward, terminated, truncated, info
    
    def _get_effective_steps(self, info):
        """
        Calculate effective steps, considering lava penalties if enabled.
        """
        if self.count_lava_steps and 'effective_steps' in info:
            return info['effective_steps']
        return self.step_count

