import gymnasium as gym
import numpy as np
import time
from collections import Counter
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from stable_baselines3.common.monitor import Monitor

from minigrid.wrappers import FullyObsWrapper, NoDeath

from EnvironmentEdits.BespokeEdits.CustomWrappers import (
    GoalAngleDistanceWrapper, 
    PartialObsWrapper, 
    ExtractAbstractGrid, 
    PartialRGBObsWrapper, 
    PartialGrayObsWrapper, 
    ForceFloat32,
    DiagonalMoveMonitor
)
from EnvironmentEdits.BespokeEdits.FeatureExtractor import SelectiveObservationWrapper
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.BespokeEdits.GymCompatibility import OldGymCompatibility

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
    env = Monitor(env)
    
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
        print(f"Starting position: {env.unwrapped.agent_pos}, direction: {env.unwrapped.agent_dir}")
        
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
            
            # Print position after move
            print(f"  Position: {env.unwrapped.agent_pos}, Direction: {env.unwrapped.agent_dir}")
            
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