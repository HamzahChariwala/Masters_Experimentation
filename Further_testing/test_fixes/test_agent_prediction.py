#!/usr/bin/env python3
"""
Test script to verify agent prediction and path generation.
This script will test the agent's behavior in a single environment and verify that it generates
proper paths and model inputs.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from Agent_Evaluation.EnvironmentTooling.extract_grid import extract_grid_from_env
from Agent_Evaluation.EnvironmentTooling.import_vars import create_evaluation_env, extract_env_config

def load_agent(agent_path, config=None):
    """
    Load agent from specified path.
    
    Args:
        agent_path: Path to agent directory
        config: Optional config dictionary
    
    Returns:
        Loaded agent model
    """
    from Agent_Evaluation.generate_evaluations import load_agent as original_load_agent
    from Agent_Evaluation.generate_evaluations import load_config
    
    if config is None:
        config = load_config(agent_path)
    
    return original_load_agent(agent_path, config)

def run_agent_prediction_test(agent_path, env_id, seed, max_steps=20):
    """
    Test agent prediction by running it in a single environment.
    
    Args:
        agent_path: Path to agent directory
        env_id: Environment ID
        seed: Environment seed
        max_steps: Maximum number of steps to run
    """
    print(f"Testing agent prediction for {agent_path} in {env_id} with seed {seed}")
    
    # Load agent config
    from Agent_Evaluation.generate_evaluations import load_config
    config = load_config(agent_path)
    
    # Extract environment settings
    env_settings = extract_env_config(config, override_env_id=env_id)
    
    # Create environment
    print("Creating environment...")
    env = create_evaluation_env(env_settings, seed=seed, override_rank=0)
    
    # Extract environment layout
    print("Extracting environment layout...")
    env_tensor = extract_grid_from_env(env)
    
    # Visualize environment
    print_env_tensor(env_tensor)
    
    # Load agent
    print("Loading agent...")
    agent = load_agent(agent_path, config)
    
    # Run prediction test
    print("\nRunning agent prediction test...")
    run_prediction_in_env(agent, env, env_tensor, max_steps)
    
def print_env_tensor(env_tensor):
    """
    Print the environment tensor in a readable format.
    
    Args:
        env_tensor: Environment tensor
    """
    if env_tensor is None:
        print("Empty environment tensor")
        return
    
    print("\nEnvironment Layout:")
    height, width = env_tensor.shape
    
    for y in range(height):
        row = [env_tensor[y, x][0].upper() for x in range(width)]
        print(' '.join(row))
    print()

def print_state_info(env, state_key, info, action=None, reward=None):
    """
    Print detailed information about a state.
    
    Args:
        env: Environment
        state_key: State key (x,y,orientation)
        info: Environment info dictionary
        action: Optional action taken
        reward: Optional reward received
    """
    print(f"\nState: {state_key}")
    
    if action is not None:
        # Convert numpy array to int if needed
        if isinstance(action, np.ndarray):
            action = int(action)
            
        action_name = {
            0: "Turn left",
            1: "Turn right",
            2: "Move forward",
            3: "Move diagonal left",
            4: "Move diagonal right"
        }.get(action, f"Unknown({action})")
        print(f"Action: {action} ({action_name})")
    
    if reward is not None:
        print(f"Reward: {reward}")
    
    # Extract log_data if available
    if "log_data" in info and isinstance(info["log_data"], dict):
        log_data = info["log_data"]
        
        # Print goal direction if available
        if "four_way_goal_direction" in log_data:
            print(f"Goal direction: {log_data['four_way_goal_direction']}")
        
        # Print angle alignment if available
        if "four_way_angle_alignment" in log_data:
            print(f"Angle alignment: {log_data['four_way_angle_alignment']}")
        
        # Print barrier and lava masks in compact form if available
        for mask_name in ["barrier_mask", "lava_mask"]:
            if mask_name in log_data and log_data[mask_name] is not None:
                mask = log_data[mask_name]
                if isinstance(mask, np.ndarray):
                    print(f"{mask_name} shape: {mask.shape}")
                elif isinstance(mask, list):
                    print(f"{mask_name} dimensions: {len(mask)}x{len(mask[0]) if mask else 0}")

def run_prediction_in_env(agent, env, env_tensor, max_steps=20):
    """
    Run agent prediction in the environment.
    
    Args:
        agent: Agent model
        env: Environment
        env_tensor: Environment tensor
        max_steps: Maximum number of steps to run
    """
    # Reset environment
    obs, info = env.reset()
    
    # Get initial state
    initial_x, initial_y, initial_orientation = get_agent_position_orientation(env)
    path = [(initial_x, initial_y, initial_orientation)]
    
    # Print initial state info
    state_key = f"{initial_x},{initial_y},{initial_orientation}"
    print(f"Initial state: {state_key}")
    print_state_info(env, state_key, info)
    
    # Run agent prediction
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < max_steps:
        # Get action from agent
        action, _ = agent.predict(obs, deterministic=True)
        
        # Print raw action for debugging
        print(f"Raw action from agent: {action} (type: {type(action)})")
        
        # Ensure action is an integer
        if isinstance(action, np.ndarray):
            action_int = int(action)
        else:
            action_int = action
            
        # Take action in environment
        obs, reward, terminated, truncated, info = env.step(action_int)
        
        # Update step count and reward
        step += 1
        total_reward += reward
        
        # Get current state
        x, y, orientation = get_agent_position_orientation(env)
        state_key = f"{x},{y},{orientation}"
        
        # Check cell type
        cell_type = "unknown"
        if 0 <= y < env_tensor.shape[0] and 0 <= x < env_tensor.shape[1]:
            cell_type = env_tensor[y, x]
        
        # Add to path
        path.append((x, y, orientation))
        
        # Print state info
        print(f"\nStep {step}:")
        print_state_info(env, state_key, info, action_int, reward)
        print(f"Cell type: {cell_type}")
        
        # Check if done
        done = terminated or truncated
        if done:
            print(f"\nEpisode ended after {step} steps")
            print(f"Final state: {state_key}")
            print(f"Total reward: {total_reward}")
    
    # Print final path
    print("\nFinal path:")
    for i, (x, y, orientation) in enumerate(path):
        print(f"  {i}: ({x},{y},{orientation})")
    
    return path

def get_agent_position_orientation(env):
    """
    Get agent position and orientation from environment.
    
    Args:
        env: Environment
    
    Returns:
        Tuple of (x, y, orientation)
    """
    # Try to unwrap the environment to get agent position and orientation
    current_env = env
    max_depth = 10
    
    for _ in range(max_depth):
        if hasattr(current_env, 'agent_pos') and hasattr(current_env, 'agent_dir'):
            return current_env.agent_pos[0], current_env.agent_pos[1], current_env.agent_dir
        
        if hasattr(current_env, 'env'):
            current_env = current_env.env
        elif hasattr(current_env, 'unwrapped'):
            current_env = current_env.unwrapped
        else:
            break
    
    # Default values if agent position and orientation not found
    return 1, 1, 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test agent prediction")
    parser.add_argument("--agent_path", default="LavaTests/Standard", help="Path to agent directory")
    parser.add_argument("--env_id", default="MiniGrid-LavaCrossingS11N5-v0", help="Environment ID")
    parser.add_argument("--seed", type=int, default=81102, help="Environment seed")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum steps to run")
    
    args = parser.parse_args()
    
    run_agent_prediction_test(args.agent_path, args.env_id, args.seed, args.max_steps) 