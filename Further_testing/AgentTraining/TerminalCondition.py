import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class CustomTerminationCallback(BaseCallback):
    """
    Custom callback for checking termination conditions during training.
    
    Termination checks are applied in the following order:
    1. Performance threshold - Stop when mean reward reaches min_reward_threshold
    2. No improvement check - Stop if no improvement for max_no_improvement_steps
    3. Episode count - Stop after max_episodes
    4. Runtime - Stop after max_runtime seconds
    
    You can use any combination of these conditions by providing only the parameters
    you want to use. Any parameter set to None will be ignored.
    
    Parameters:
        eval_env: Environment to evaluate on (required)
        check_freq: Frequency to evaluate performance in timesteps (default: 10000)
        min_reward_threshold: Stop when mean reward reaches this value (default: None)
        max_episodes: Maximum number of episodes before stopping (default: None)
        max_runtime: Maximum runtime in seconds (default: None)
        max_no_improvement_steps: Stop if no improvement for this many steps (default: None)
        n_eval_episodes: Number of episodes to evaluate for performance checks (default: 10)
        verbose: Verbosity level (default: 1)
    """
    
    def __init__(
        self,
        eval_env,
        check_freq=10000,
        min_reward_threshold=None,
        max_episodes=None,
        max_runtime=None,
        max_no_improvement_steps=None,
        n_eval_episodes=10,
        verbose=1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.min_reward_threshold = min_reward_threshold
        self.max_episodes = max_episodes
        self.max_runtime = max_runtime
        self.max_no_improvement_steps = max_no_improvement_steps
        self.n_eval_episodes = n_eval_episodes
        
        self.start_time = None
        self.best_mean_reward = -np.inf
        self.steps_without_improvement = 0
        self.episode_count = 0
        
    def _init_callback(self):
        self.start_time = time.time()
        
    def _on_step(self):
        # Track episode count from infos
        if len(self.model.ep_info_buffer) > 0:
            self.episode_count = len(self.model.ep_info_buffer)
            
        # Check termination conditions only at specified frequency
        if self.num_timesteps % self.check_freq != 0:
            return True
            
        # Check performance (mean reward)
        if self.min_reward_threshold is not None or self.max_no_improvement_steps is not None:
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
            
            # Check if performance exceeds threshold
            if self.min_reward_threshold is not None and mean_reward >= self.min_reward_threshold:
                if self.verbose > 0:
                    print(f"Stopping training as mean reward {mean_reward:.2f} reached threshold {self.min_reward_threshold:.2f}")
                return False
                
            # Check for improvement
            if self.max_no_improvement_steps is not None:
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.steps_without_improvement = 0
                else:
                    self.steps_without_improvement += self.check_freq
                    if self.steps_without_improvement >= self.max_no_improvement_steps:
                        if self.verbose > 0:
                            print(f"Stopping training as no improvement for {self.steps_without_improvement} steps")
                        return False
        
        # Check episode count
        if self.max_episodes is not None and self.episode_count >= self.max_episodes:
            if self.verbose > 0:
                print(f"Stopping training after {self.episode_count} episodes")
            return False
            
        # Check runtime
        if self.max_runtime is not None:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.max_runtime:
                if self.verbose > 0:
                    print(f"Stopping training after {elapsed_time:.2f} seconds")
                return False
                
        return True
