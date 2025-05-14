import os
import sys
import numpy as np
import time
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import torch
import traceback
import threading
from collections import Counter
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from minigrid.wrappers import FullyObsWrapper, NoDeath

# Add the parent directory to sys.path to ensure Environment_Tooling can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import SafeMonitor from Environment_Tooling
from Environment_Tooling.EnvironmentGeneration import SafeMonitor, safe_evaluate_policy

from Environment_Tooling.BespokeEdits.CustomWrappers import (
    GoalAngleDistanceWrapper, 
    PartialObsWrapper, 
    ExtractAbstractGrid, 
    PartialRGBObsWrapper, 
    PartialGrayObsWrapper, 
    ForceFloat32,
    DiagonalMoveMonitor
)
from Environment_Tooling.BespokeEdits.FeatureExtractor import SelectiveObservationWrapper
from Environment_Tooling.BespokeEdits.ActionSpace import CustomActionWrapper
from Environment_Tooling.BespokeEdits.GymCompatibility import OldGymCompatibility

class PerformanceTracker:
    """
    Track and plot agent performance metrics during training and evaluation.
    """
    def __init__(self, log_dir="./logs/performance"):
        """
        Initialize the performance tracker.
        
        Parameters:
        ----------
        log_dir : str
            Directory to save plots
        """
        os.makedirs(log_dir, exist_ok=True)
        
        # Track metrics
        self.train_timesteps = []
        self.train_rewards = []
        self.train_lengths = []
        self.train_episode_counts = []
        
        self.eval_timesteps = []
        self.eval_rewards = []
        self.eval_lengths = []
        self.eval_rewards_std = []
        self.eval_lengths_std = []
        
    def update_train_metrics(self, timestep, rewards, lengths, episode_count):
        """
        Update training metrics.
        
        Parameters:
        ----------
        timestep : int
            Current timestep
        rewards : list
            List of episode rewards
        lengths : list
            List of episode lengths
        episode_count : int
            Total episodes so far
        """
        self.train_timesteps.append(timestep)
        self.train_rewards.append(np.mean(rewards) if len(rewards) > 0 else 0)
        self.train_lengths.append(np.mean(lengths) if len(lengths) > 0 else 0)
        self.train_episode_counts.append(episode_count)
        
    def update_eval_metrics(self, timestep, rewards, lengths):
        """
        Update evaluation metrics.
        
        Parameters:
        ----------
        timestep : int
            Current timestep
        rewards : list
            List of evaluation episode rewards
        lengths : list
            List of evaluation episode lengths
        """
        self.eval_timesteps.append(timestep)
        self.eval_rewards.append(np.mean(rewards) if len(rewards) > 0 else 0)
        self.eval_lengths.append(np.mean(lengths) if len(lengths) > 0 else 0)
        self.eval_rewards_std.append(np.std(rewards) if len(rewards) > 0 else 0)
        self.eval_lengths_std.append(np.std(lengths) if len(lengths) > 0 else 0)
        
    def plot_performance(self, save=True, show=False):
        """
        Plot performance metrics comparing training and evaluation.
        
        Parameters:
        ----------
        save : bool
            Whether to save the plots to files
        show : bool
            Whether to display the plots
        """
        # Create new figure for rewards
        plt.figure(figsize=(12, 6))
        plt.title("Average Reward during Training and Evaluation")
        
        # Plot train rewards
        if len(self.train_timesteps) > 0:
            plt.plot(self.train_timesteps, self.train_rewards, 'b-', alpha=0.5, label="Train")
        
        # Plot eval rewards with error bars
        if len(self.eval_timesteps) > 0:
            plt.errorbar(
                self.eval_timesteps, 
                self.eval_rewards, 
                yerr=self.eval_rewards_std,
                fmt='ro-', 
                capsize=5, 
                label="Evaluation"
            )
        
        plt.xlabel("Timesteps")
        plt.ylabel("Average Reward")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if save:
            plt.savefig(f"{self.log_dir}/rewards.png")
        if show:
            plt.show()
        else:
            plt.close()
            
        # Create new figure for episode lengths
        plt.figure(figsize=(12, 6))
        plt.title("Average Episode Length during Training and Evaluation")
        
        # Plot train lengths
        if len(self.train_timesteps) > 0:
            plt.plot(self.train_timesteps, self.train_lengths, 'b-', alpha=0.5, label="Train")
        
        # Plot eval lengths with error bars
        if len(self.eval_timesteps) > 0:
            plt.errorbar(
                self.eval_timesteps, 
                self.eval_lengths, 
                yerr=self.eval_lengths_std,
                fmt='go-', 
                capsize=5, 
                label="Evaluation"
            )
        
        plt.xlabel("Timesteps")
        plt.ylabel("Average Episode Length")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if save:
            plt.savefig(f"{self.log_dir}/episode_lengths.png")
        if show:
            plt.show()
        else:
            plt.close()
            
        # Create combined plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot rewards
        ax1.set_title("Agent Performance Metrics")
        
        if len(self.train_timesteps) > 0:
            ax1.plot(self.train_timesteps, self.train_rewards, 'b-', alpha=0.5, label="Train")
        
        if len(self.eval_timesteps) > 0:
            ax1.errorbar(
                self.eval_timesteps, 
                self.eval_rewards, 
                yerr=self.eval_rewards_std,
                fmt='ro-', 
                capsize=5, 
                label="Evaluation"
            )
            
        ax1.set_ylabel("Average Reward")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot episode lengths
        if len(self.train_timesteps) > 0:
            ax2.plot(self.train_timesteps, self.train_lengths, 'b-', alpha=0.5, label="Train")
        
        if len(self.eval_timesteps) > 0:
            ax2.errorbar(
                self.eval_timesteps, 
                self.eval_lengths, 
                yerr=self.eval_lengths_std,
                fmt='go-', 
                capsize=5, 
                label="Evaluation"
            )
            
        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Average Episode Length")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.log_dir}/combined_metrics.png")
        if show:
            plt.show()
        else:
            plt.close()
            
        # Print performance comparison
        print("\n===== Performance Summary =====")
        
        if len(self.eval_rewards) > 0:
            print(f"Latest evaluation metrics (timestep {self.eval_timesteps[-1]}):")
            print(f"  Reward: {self.eval_rewards[-1]:.2f} ± {self.eval_rewards_std[-1]:.2f}")
            print(f"  Episode Length: {self.eval_lengths[-1]:.1f} ± {self.eval_lengths_std[-1]:.1f}")
            
            if len(self.train_rewards) > 0:
                # Find train metrics closest to last evaluation
                closest_idx = np.argmin(np.abs(np.array(self.train_timesteps) - self.eval_timesteps[-1]))
                print(f"Comparable training metrics (timestep {self.train_timesteps[closest_idx]}):")
                print(f"  Reward: {self.train_rewards[closest_idx]:.2f}")
                print(f"  Episode Length: {self.train_lengths[closest_idx]:.1f}")
                
                # Calculate performance gaps
                reward_gap = self.eval_rewards[-1] - self.train_rewards[closest_idx]
                length_gap = self.eval_lengths[-1] - self.train_lengths[closest_idx]
                
                print(f"Performance gap (eval - train):")
                print(f"  Reward: {reward_gap:.2f} ({reward_gap/max(0.01, self.train_rewards[closest_idx])*100:.1f}%)")
                print(f"  Episode Length: {length_gap:.1f} ({length_gap/max(0.01, self.train_lengths[closest_idx])*100:.1f}%)")
                
        print("=============================")
        
        return True

def evaluate_with_timeout(model, env, n_eval_episodes=10, timeout=30, deterministic=True):
    """
    Evaluate a model on an environment with a timeout to prevent getting stuck.
    
    Parameters:
    ----------
    model : stable_baselines3 model
        The trained agent to evaluate
    env : gymnasium.Env
        The evaluation environment
    n_eval_episodes : int
        Number of episodes to evaluate
    timeout : int
        Timeout in seconds for the evaluation
    deterministic : bool
        Whether to use deterministic actions
        
    Returns:
    -------
    tuple
        Tuple of (rewards, episode_lengths) if successful,
        or ([], []) if timeout or error occurs
    """
    # Variables to store results and status
    eval_results = None
    eval_complete = False
    eval_error = None
    
    # Function to run in a separate thread
    def run_evaluation():
        nonlocal eval_results, eval_complete, eval_error
        try:
            rewards, lengths = safe_evaluate_policy(
                model,
                env,
                n_eval_episodes=n_eval_episodes,
                deterministic=deterministic,
                return_episode_rewards=True,
            )
            eval_results = (rewards, lengths)
            eval_complete = True
        except Exception as e:
            eval_error = str(e)
            eval_complete = True
    
    # Start evaluation in a separate thread
    eval_thread = threading.Thread(target=run_evaluation)
    eval_thread.daemon = True
    eval_thread.start()
    
    # Wait for evaluation to complete with timeout
    start_time = time.time()
    while not eval_complete and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    # Check evaluation status
    if not eval_complete:
        return [], [], "TIMEOUT"
        
    if eval_error:
        return [], [], eval_error
        
    rewards, lengths = eval_results
    return rewards, lengths, None

def visualize_agent_behavior(model, env_id, observation_params, seed=42, num_episodes=5, render=True):
    """
    Visualize the agent's behavior and print detailed information about each action.
    
    Parameters:
    ----------
    model : stable_baselines3 model
        The trained agent
    env_id : str
        Environment ID to create
    observation_params : dict
        Dictionary of observation parameters
    seed : int
        Seed for the environment
    num_episodes : int
        Number of episodes to visualize
    render : bool
        Whether to render the environment (may require extra setup)
    """
    # Create a single environment for visualization
    print(f"\n===== DETAILED AGENT BEHAVIOR ANALYSIS =====")
    print(f"\n===== Visualizing Agent Behavior ({num_episodes} episodes) =====")
    
    # Use the observation parameters but enable monitoring
    eval_params = observation_params.copy()
    eval_params["use_random_spawn"] = False  # Use standard spawn for visualization
    eval_params["monitor_diagonal_moves"] = True
    
    # Create environment with human rendering if requested
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    
    # Apply our wrappers
    if eval_params.get("max_episode_steps") is not None:
        env = TimeLimit(env, max_episode_steps=eval_params["max_episode_steps"])
    
    if eval_params.get("use_no_death", True):
        env = NoDeath(env, 
                     no_death_types=eval_params.get("no_death_types", ("lava",)), 
                     death_cost=eval_params.get("death_cost", -0.25))
    
    env = CustomActionWrapper(env)
    env = DiagonalMoveMonitor(env)
    env = RecordEpisodeStatistics(env)
    env = FullyObsWrapper(env)
    
    env = GoalAngleDistanceWrapper(env)
    env = PartialObsWrapper(env, eval_params.get("window_size", 5))
    env = ExtractAbstractGrid(env)
    env = PartialRGBObsWrapper(env, eval_params.get("window_size", 5))
    env = PartialGrayObsWrapper(env, eval_params.get("window_size", 5))
    
    env = SelectiveObservationWrapper(
        env,
        cnn_keys=eval_params.get("cnn_keys", []),
        mlp_keys=eval_params.get("mlp_keys", [])
    )
    env = ForceFloat32(env)
    env = OldGymCompatibility(env)
    env = SafeMonitor(env)
    
    # Helper function to safely get agent position and direction
    def get_agent_info(env):
        # Start with the environment
        current_env = env
        
        # Keep unwrapping until we find agent_pos or run out of unwrapped envs
        while hasattr(current_env, 'unwrapped'):
            # Check if current level has the attributes
            if hasattr(current_env, 'agent_pos') and hasattr(current_env, 'agent_dir'):
                return current_env.agent_pos, current_env.agent_dir
            
            # Move to the next unwrapped level
            current_env = current_env.unwrapped
            
            # Check again at this level
            if hasattr(current_env, 'agent_pos') and hasattr(current_env, 'agent_dir'):
                return current_env.agent_pos, current_env.agent_dir
        
        # If we got here, we couldn't find the attributes
        return None, None
    
    # Track action statistics across all episodes
    action_counter = Counter()
    successful_diag_counter = 0
    failed_diag_counter = 0
    total_steps = 0
    total_rewards = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        step = 0
        done = False
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        # Get agent position and direction safely
        agent_pos, agent_dir = get_agent_info(env)
        if agent_pos is not None and agent_dir is not None:
            print(f"Starting position: {agent_pos}, direction: {agent_dir}")
        else:
            print("Could not determine agent position and direction")
        
        while not done:
            # Get the action from the model
            action, _ = model.predict(obs, deterministic=True)
            action_counter[int(action)] += 1
            
            # Log the action being taken
            action_name = {
                0: "Turn left",
                1: "Turn right",
                2: "Move forward",
                3: "Move diagonal left",
                4: "Move diagonal right"
            }.get(action, f"Unknown({action})")
            
            print(f"Step {step}: Action {action} ({action_name})")
            
            # Take the action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Log the result
            if "action" in info and info["action"] == "diagonal":
                if "failed" in info and info["failed"]:
                    print(f"  Diagonal move FAILED: {info['diag_direction']}")
                    failed_diag_counter += 1
                else:
                    print(f"  Diagonal move SUCCESS: {info['diag_direction']}")
                    successful_diag_counter += 1
            
            # Print position after move (safely)
            agent_pos, agent_dir = get_agent_info(env)
            if agent_pos is not None and agent_dir is not None:
                print(f"  Position: {agent_pos}, Direction: {agent_dir}")
            
            episode_reward += reward
            total_rewards += reward
            step += 1
            total_steps += 1
            done = terminated or truncated
            
            if render:
                time.sleep(0.2)  # Slow down rendering
        
        print(f"Episode {episode+1} finished: Steps={step}, Reward={episode_reward:.2f}")
    
    # Print overall statistics
    print("\n===== Overall Statistics =====")
    print(f"Total steps: {total_steps}")
    print(f"Total reward: {total_rewards:.2f}")
    print(f"Average reward per episode: {total_rewards/num_episodes:.2f}")
    print(f"Action distribution: {dict(action_counter)}")
    
    # Calculate percentages
    action_percentages = {action: (count/total_steps)*100 for action, count in action_counter.items()}
    print(f"Action percentages: {action_percentages}")
    
    # Diagonal move statistics
    total_diag_attempts = successful_diag_counter + failed_diag_counter
    if total_diag_attempts > 0:
        success_rate = (successful_diag_counter / total_diag_attempts) * 100
        print(f"Diagonal attempts: {total_diag_attempts} ({total_diag_attempts/total_steps*100:.1f}% of all actions)")
        print(f"Successful diagonal moves: {successful_diag_counter} ({success_rate:.1f}% success rate)")
    else:
        print("No diagonal moves attempted")
    
    print("=============================")
    
    env.close()
    return env.get_episode_history() if hasattr(env, "get_episode_history") else None 