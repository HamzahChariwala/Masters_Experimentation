import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class CustomTerminationCallback(BaseCallback):
    """
    Custom callback for checking termination conditions during training.
    
    The callback first checks if both minimum thresholds (reward and episode length) 
    have been met. Only after both minimums are satisfied will the other 
    termination conditions be checked.
    
    Termination checks are applied in the following order (after minimum thresholds are met):
    1. Performance threshold - Stop when mean reward reaches min_reward_threshold
    2. No improvement checks - Stop if no improvement in reward or episode length
    3. Episode count - Stop after max_episodes
    4. Runtime - Stop after max_runtime seconds
    
    You can use any combination of these conditions by providing only the parameters
    you want to use. Any parameter set to None will be ignored.
    
    Parameters:
        eval_env: Environment to evaluate on (required)
        check_freq: Frequency to evaluate performance in timesteps (default: 10000)
        min_reward_threshold: Minimum mean reward that must be reached before termination conditions apply (default: None)
        min_episode_length_threshold: Minimum mean episode length that must be reached before termination conditions apply (default: None)
        target_reward_threshold: Target mean reward to reach before stopping (default: None)
        max_episodes: Maximum number of episodes before stopping (default: None)
        max_runtime: Maximum runtime in seconds (default: None)
        max_no_improvement_reward_steps: Stop if no improvement in reward for this many steps (default: None)
        max_no_improvement_length_steps: Stop if no improvement in episode length for this many steps (default: None)
        n_eval_episodes: Number of episodes to evaluate for performance checks (default: 10)
        verbose: Verbosity level (default: 1)
    """
    
    def __init__(
        self,
        eval_env,
        check_freq=10000,
        min_reward_threshold=None,
        min_episode_length_threshold=None,
        target_reward_threshold=None,
        max_episodes=None,
        max_runtime=None,
        max_no_improvement_reward_steps=None,
        max_no_improvement_length_steps=None,
        n_eval_episodes=10,
        verbose=1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        
        # Minimum thresholds that must be met before other conditions apply
        self.min_reward_threshold = min_reward_threshold
        self.min_episode_length_threshold = min_episode_length_threshold
        
        # Target conditions
        self.target_reward_threshold = target_reward_threshold
        self.max_episodes = max_episodes
        self.max_runtime = max_runtime
        
        # Improvement tracking parameters
        self.max_no_improvement_reward_steps = max_no_improvement_reward_steps
        self.max_no_improvement_length_steps = max_no_improvement_length_steps
        
        self.n_eval_episodes = n_eval_episodes
        
        # State tracking variables
        self.start_time = None
        self.best_mean_reward = -np.inf
        self.best_mean_length = np.inf  # Lower is better for episode length
        self.reward_steps_without_improvement = 0
        self.length_steps_without_improvement = 0
        self.episode_count = 0
        self.reward_threshold_met = False
        self.length_threshold_met = False
        
        # Safety variables
        self.evaluation_in_progress = False
        self.last_evaluation_time = 0
        
    def _init_callback(self):
        self.start_time = time.time()
        
    def _on_step(self):
        # Check for time-based termination immediately without expensive evaluation
        if self.max_runtime is not None:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.max_runtime:
                if self.verbose > 0:
                    print(f"Stopping training after {elapsed_time:.2f} seconds")
                return False
        
        # Track episode count from infos
        if len(self.model.ep_info_buffer) > 0:
            self.episode_count = len(self.model.ep_info_buffer)
            
        # Check episode count termination immediately without expensive evaluation
        if self.max_episodes is not None and self.episode_count >= self.max_episodes:
            if self.verbose > 0:
                print(f"Stopping training after {self.episode_count} episodes")
            return False
            
        # Only check other termination conditions at specified frequency
        if self.num_timesteps % self.check_freq != 0:
            return True
            
        # Safety check to prevent potential deadlocks
        current_time = time.time()
        if self.evaluation_in_progress and (current_time - self.last_evaluation_time < 120):
            if self.verbose > 0:
                print("Previous evaluation still in progress, skipping this one")
            return True
            
        try:
            self.evaluation_in_progress = True
            self.last_evaluation_time = current_time
            
            # Evaluate performance
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
            )
            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            if self.verbose > 0:
                print(f"Evaluation at step {self.num_timesteps}: Mean reward = {mean_reward:.2f}, Mean length = {mean_length:.1f}")
            
            # Check minimum thresholds
            if self.min_reward_threshold is not None and mean_reward >= self.min_reward_threshold:
                self.reward_threshold_met = True
                
            if self.min_episode_length_threshold is not None and mean_length <= self.min_episode_length_threshold:
                self.length_threshold_met = True
                
            # Track improvement regardless of thresholds
            # For reward, higher is better
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.reward_steps_without_improvement = 0
            else:
                self.reward_steps_without_improvement += self.check_freq
                
            # For episode length, lower is better (typically)
            if mean_length < self.best_mean_length:
                self.best_mean_length = mean_length
                self.length_steps_without_improvement = 0
            else:
                self.length_steps_without_improvement += self.check_freq
            
            # Only apply termination conditions if both minimum thresholds are met
            # or if the thresholds are not specified
            min_thresholds_met = (
                (self.min_reward_threshold is None or self.reward_threshold_met) and
                (self.min_episode_length_threshold is None or self.length_threshold_met)
            )
            
            if min_thresholds_met:
                # Check target reward threshold
                if self.target_reward_threshold is not None and mean_reward >= self.target_reward_threshold:
                    if self.verbose > 0:
                        print(f"Stopping training as mean reward {mean_reward:.2f} reached target {self.target_reward_threshold:.2f}")
                    return False
                    
                # Check for no improvement in reward
                if (self.max_no_improvement_reward_steps is not None and 
                    self.reward_steps_without_improvement >= self.max_no_improvement_reward_steps):
                    if self.verbose > 0:
                        print(f"Stopping training as no improvement in reward for {self.reward_steps_without_improvement} steps")
                    return False
                    
                # Check for no improvement in episode length
                if (self.max_no_improvement_length_steps is not None and 
                    self.length_steps_without_improvement >= self.max_no_improvement_length_steps):
                    if self.verbose > 0:
                        print(f"Stopping training as no improvement in episode length for {self.length_steps_without_improvement} steps")
                    return False
        
        except Exception as e:
            if self.verbose > 0:
                print(f"Error during evaluation: {e}")
        finally:
            self.evaluation_in_progress = False
                
        return True
