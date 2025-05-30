import os
import time
import numpy as np
import threading
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Import our safe version of evaluate_policy
from Environment_Tooling.EnvironmentGeneration import safe_evaluate_policy

# Import performance tracker using relative import
from .Tooling import PerformanceTracker

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
        eval_envs: List of environments to evaluate on (required)
        check_freq: Frequency to evaluate performance in timesteps (default: 10000)
        min_reward_threshold: Minimum mean reward that must be reached before termination conditions apply (default: None)
        min_episode_length_threshold: Minimum mean episode length that must be reached before termination conditions apply (default: None)
        target_reward_threshold: Target mean reward to reach before stopping (default: None)
        max_episodes: Maximum number of episodes before stopping (default: None)
        max_runtime: Maximum runtime in seconds (default: None)
        max_no_improvement_reward_steps: Stop if no improvement in reward for this many steps (default: None)
        max_no_improvement_length_steps: Stop if no improvement in episode length for this many steps (default: None)
        n_eval_episodes: Number of episodes to evaluate for performance checks (default: 10)
        eval_timeout: Maximum time in seconds to wait for evaluation to complete (default: 30)
        log_dir: Directory to save performance plots (default: "./logs/performance")
        verbose: Verbosity level (default: 1)
        disable_early_stopping: If True, disable all early stopping conditions but still run evaluations (default: False)
    """
    
    def __init__(
        self,
        eval_envs,
        check_freq=10000,
        min_reward_threshold=None,
        min_episode_length_threshold=None,
        target_reward_threshold=None,
        max_episodes=None,
        max_runtime=None,
        max_no_improvement_reward_steps=None,
        max_no_improvement_length_steps=None,
        n_eval_episodes=10,
        eval_timeout=30,
        log_dir="./logs/performance",
        verbose=1,
        disable_early_stopping=False
    ):
        super().__init__(verbose)
        # Handle both single environment and list of environments
        if isinstance(eval_envs, list):
            self.eval_envs = eval_envs
        else:
            self.eval_envs = [eval_envs]
            
        self.check_freq = check_freq
        
        # Flag to disable all early stopping conditions
        self.disable_early_stopping = disable_early_stopping
        
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
        self.eval_timeout = eval_timeout
        self.log_dir = log_dir
        
        # State tracking variables
        self.start_time = None
        self.best_mean_reward = -np.inf
        self.best_mean_length = np.inf  # Lower is better for episode length
        self.reward_steps_without_improvement = 0
        self.length_steps_without_improvement = 0
        self.episode_count = 0
        self.reward_threshold_met = False
        self.length_threshold_met = False
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker(log_dir=log_dir)
        
        # Safety variables
        self.evaluation_in_progress = False
        self.last_evaluation_time = 0
        self.evaluation_count = 0
        self.evaluation_results = None
        self.evaluation_complete = False
        self.evaluation_error = None
        
    def _init_callback(self):
        self.start_time = time.time()
        
    def _evaluate_agent(self):
        """Thread-safe evaluation function to run in a separate thread.
        Evaluates the agent on multiple environments and aggregates the results."""
        try:
            # Reset evaluation state
            self.evaluation_complete = False
            self.evaluation_error = None
            self.evaluation_results = None
            
            # Create lists to store results from all environments
            all_rewards = []
            all_lengths = []
            
            # Run evaluation on each environment
            for i, eval_env in enumerate(self.eval_envs):
                try:
                    if self.verbose > 0:
                        print(f"Evaluating on environment {i+1}/{len(self.eval_envs)}")
                    
                    # Run evaluation on this environment
                    env_rewards, env_lengths = safe_evaluate_policy(
                        self.model,
                        eval_env,
                        n_eval_episodes=self.n_eval_episodes,
                        deterministic=True,
                        return_episode_rewards=True,
                    )
                    
                    # Store results
                    all_rewards.extend(env_rewards)
                    all_lengths.extend(env_lengths)
                    
                    if self.verbose > 0:
                        print(f"  Environment {i+1} results: Mean reward = {np.mean(env_rewards):.2f}, Mean length = {np.mean(env_lengths):.1f}")
                        
                except Exception as env_error:
                    if self.verbose > 0:
                        print(f"Error evaluating on environment {i+1}: {env_error}")
            
            if len(all_rewards) > 0:
                # Store aggregated results
                self.evaluation_results = (all_rewards, all_lengths)
                self.evaluation_complete = True
            else:
                raise Exception("No valid evaluation results collected from any environment")
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Error during evaluation: {e}")
            self.evaluation_error = str(e)
            self.evaluation_complete = True
        
    def _on_step(self):
        # If early stopping is disabled, only run evaluations but don't terminate training
        if self.disable_early_stopping:
            # Only do evaluations at check_freq
            if self.num_timesteps % self.check_freq != 0:
                return True
                
            # Track episode count from infos for performance tracker
            if len(self.model.ep_info_buffer) > 0:
                self.episode_count = len(self.model.ep_info_buffer)
                
                # Update training metrics in the performance tracker
                train_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                train_lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
                self.performance_tracker.update_train_metrics(
                    self.num_timesteps, 
                    train_rewards, 
                    train_lengths, 
                    self.episode_count
                )
                
            # Skip safety check for deadlocks
            if self.evaluation_in_progress:
                current_time = time.time()
                time_since_last_eval = current_time - self.last_evaluation_time
                if time_since_last_eval > 120:  # If more than 2 minutes, something is wrong
                    if self.verbose > 0:
                        print(f"Warning: Previous evaluation has been running for {time_since_last_eval:.1f} seconds. Resetting state.")
                    self.evaluation_in_progress = False
                else:
                    if self.verbose > 0:
                        print("Previous evaluation still in progress, skipping this one")
                    return True
            
            # Mark that evaluation is in progress and record time
            self.evaluation_in_progress = True
            self.last_evaluation_time = time.time()
            
            # Run evaluation for tracking progress but don't terminate
            try:
                result = self._run_evaluation()
                
                # Explicitly generate performance plots after evaluation
                if result and hasattr(self, "performance_tracker"):
                    # Generate performance plots
                    self.performance_tracker.plot_performance(save=True, show=False)
                    if self.verbose > 0:
                        print(f"Performance plots updated and saved to {self.log_dir}")
                
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error during evaluation with early stopping disabled: {e}")
            finally:
                self.evaluation_in_progress = False
            
            # Always continue training when disable_early_stopping is True
            return True
        
        # Normal behavior when early stopping is enabled
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
            
            # Update training metrics in the performance tracker
            train_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            train_lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
            self.performance_tracker.update_train_metrics(
                self.num_timesteps, 
                train_rewards, 
                train_lengths, 
                self.episode_count
            )
            
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
        if self.evaluation_in_progress:
            time_since_last_eval = current_time - self.last_evaluation_time
            if time_since_last_eval > 120:  # If more than 2 minutes, something is wrong
                if self.verbose > 0:
                    print(f"Warning: Previous evaluation has been running for {time_since_last_eval:.1f} seconds. Resetting state.")
                self.evaluation_in_progress = False
            else:
                if self.verbose > 0:
                    print("Previous evaluation still in progress, skipping this one")
                return True
        
        try:
            self.evaluation_in_progress = True
            self.last_evaluation_time = current_time
            
            # Run the evaluation
            result = self._run_evaluation()
            
            # If evaluation failed, continue training
            if not result:
                self.evaluation_in_progress = False
                return True
            
            # Get evaluation results
            episode_rewards, episode_lengths = self.evaluation_results
            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            # Check minimum thresholds
            if self.min_reward_threshold is not None and mean_reward >= self.min_reward_threshold:
                self.reward_threshold_met = True
                
            if self.min_episode_length_threshold is not None and mean_length <= self.min_episode_length_threshold:
                self.length_threshold_met = True
                
            # Track improvement regardless of thresholds
            # For reward, higher is better
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    improvement = mean_reward - self.best_mean_reward
                    print(f"New best mean reward: {mean_reward:.2f} (improvement: +{improvement:.2f})")
                self.best_mean_reward = mean_reward
                self.reward_steps_without_improvement = 0
            else:
                self.reward_steps_without_improvement += self.check_freq
                if self.verbose > 0:
                    print(f"No improvement in reward for {self.reward_steps_without_improvement} steps. Current: {mean_reward:.2f}, Best: {self.best_mean_reward:.2f}")
                
            # For episode length, lower is better (typically)
            if mean_length < self.best_mean_length:
                if self.verbose > 0:
                    improvement = self.best_mean_length - mean_length
                    print(f"New best mean length: {mean_length:.1f} (improvement: -{improvement:.1f})")
                self.best_mean_length = mean_length
                self.length_steps_without_improvement = 0
            else:
                self.length_steps_without_improvement += self.check_freq
                if self.verbose > 0:
                    print(f"No improvement in episode length for {self.length_steps_without_improvement} steps. Current: {mean_length:.1f}, Best: {self.best_mean_length:.1f}")
            
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
                print(f"Error during callback logic: {e}")
        finally:
            self.evaluation_in_progress = False
                
        return True
        
    def _run_evaluation(self):
        """Run the evaluation and update performance metrics"""
        self.evaluation_count += 1
        
        if self.verbose > 0:
            print(f"\n===== Starting evaluation #{self.evaluation_count} at timestep {self.num_timesteps} =====")
            print(f"Evaluating on {len(self.eval_envs)} environments, {self.n_eval_episodes} episodes per environment")
            print(f"Total episodes: {len(self.eval_envs) * self.n_eval_episodes}, timeout: {self.eval_timeout}s")
        
        # Reset evaluation state
        self.evaluation_complete = False
        self.evaluation_error = None
        self.evaluation_results = None
        
        # Create and start evaluation thread
        eval_thread = threading.Thread(target=self._evaluate_agent)
        eval_thread.daemon = True  # Daemon thread will be killed if main thread exits
        eval_thread.start()
        
        # Wait for evaluation to complete with timeout
        start_wait = time.time()
        while not self.evaluation_complete and (time.time() - start_wait) < self.eval_timeout:
            time.sleep(0.1)  # Small sleep to avoid busy waiting
        
        # Check if evaluation completed or timed out
        if not self.evaluation_complete:
            if self.verbose > 0:
                print(f"Evaluation timed out after {self.eval_timeout} seconds")
            return False
        
        # Check if evaluation had an error
        if self.evaluation_error is not None:
            if self.verbose > 0:
                print(f"Evaluation failed with error: {self.evaluation_error}")
            return False
        
        # Get evaluation results
        episode_rewards, episode_lengths = self.evaluation_results
        mean_reward = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        # Update the performance tracker with evaluation results
        self.performance_tracker.update_eval_metrics(
            self.num_timesteps,
            episode_rewards,
            episode_lengths
        )
        
        # Generate performance plots
        self.performance_tracker.plot_performance(save=True, show=False)
        
        if self.verbose > 0:
            print(f"Aggregated results across all environments:")
            print(f"  Mean reward = {mean_reward:.2f}, Std = {np.std(episode_rewards):.2f}")
            print(f"  Mean length = {mean_length:.1f}, Std = {np.std(episode_lengths):.1f}")
            print(f"  Min reward = {np.min(episode_rewards):.2f}, Max reward = {np.max(episode_rewards):.2f}")
            print(f"  Performance plots saved to {self.log_dir}")
            print(f"===== Evaluation #{self.evaluation_count} completed =====\n")
        
        return True
