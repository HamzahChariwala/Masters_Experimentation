"""Module for handling trial evaluations in hyperparameter tuning."""

import os
import json
import numpy as np
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Tuple
import yaml
import shutil
import time

def create_trial_directory(base_dir: str, trial_number: int) -> str:
    """Create a directory for the trial results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = os.path.join(base_dir, f"trial_{trial_number}_{timestamp}")
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

def create_agent_config(base_config: Dict[str, Any], params: Dict[str, Any], 
                       output_dir: str, agent_number: int, reduced_timesteps_factor: float) -> str:
    """Create a configuration file for a single agent."""
    # Deep copy the base config
    config = {k: v.copy() if isinstance(v, dict) else v for k, v in base_config.items()}
    
    # Update with trial parameters
    for param_path, value in params.items():
        path_parts = param_path.split('.')
        current = config
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[path_parts[-1]] = value
    
    # Apply timesteps reduction
    total_timesteps = config.get('experiment', {}).get('output', {}).get('total_timesteps', 1_000_000)
    config['experiment']['output']['total_timesteps'] = int(total_timesteps * reduced_timesteps_factor)
    
    # Create agent directory and save config
    agent_dir = os.path.join(output_dir, f"agent_{agent_number}")
    os.makedirs(agent_dir, exist_ok=True)
    config_path = os.path.join(agent_dir, "config.yaml")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return agent_dir

def train_and_evaluate_agent(agent_dir: str) -> Dict[str, Any]:
    """Train and evaluate a single agent."""
    # Get relative path from Agent_Storage
    relative_path = os.path.relpath(agent_dir, "Agent_Storage")
    
    # Train the agent
    print(f"Training agent: {relative_path}")
    train_cmd = ["python", "Agent_Training/train.py", "--path", relative_path]
    subprocess.run(train_cmd, check=True)
    
    # Evaluate the agent - IMPORTANT: Use absolute path with Agent_Storage prefix
    print(f"Evaluating agent: {relative_path}")
    eval_cmd = ["python", "Agent_Evaluation/generate_evaluations.py", "--path", f"Agent_Storage/{relative_path}"]
    subprocess.run(eval_cmd, check=True)
    
    # The correct path to the performance summary file
    summary_path = os.path.join(agent_dir, "evaluation_summary", "performance_all_states.json")
    
    # Wait for the file to be created (it might take a moment)
    max_wait = 60  # Maximum wait time in seconds
    wait_time = 0
    while not os.path.exists(summary_path) and wait_time < max_wait:
        time.sleep(1)
        wait_time += 1
        if wait_time % 10 == 0:
            print(f"Waiting for summary file... ({wait_time}s)")
    
    if not os.path.exists(summary_path):
        # List files in the agent directory to help debug
        print(f"Checking directories for: {agent_dir}")
        if os.path.exists(agent_dir):
            print(f"Agent directory exists. Contents: {os.listdir(agent_dir)}")
            eval_summary_dir = os.path.join(agent_dir, "evaluation_summary")
            if os.path.exists(eval_summary_dir):
                print(f"Evaluation summary directory exists. Contents: {os.listdir(eval_summary_dir)}")
            else:
                print("Evaluation summary directory doesn't exist")
                
            # Check for evaluation logs
            eval_logs_dir = os.path.join(agent_dir, "evaluation_logs")
            if os.path.exists(eval_logs_dir):
                print(f"Evaluation logs directory exists. Contents: {os.listdir(eval_logs_dir)}")
        
        # If the file doesn't exist after waiting, create a default summary
        print("Creating default summary as evaluation failed")
        os.makedirs(os.path.join(agent_dir, "evaluation_summary"), exist_ok=True)
        default_summary = {
            "overall_summary": {
                "goal_reached_proportion": 0.0,
                "next_cell_lava_proportion": 1.0
            }
        }
        with open(summary_path, 'w') as f:
            json.dump(default_summary, f, indent=2)
    
    # Load and return evaluation results
    with open(summary_path, 'r') as f:
        results = json.load(f)
    
    # Get the overall summary
    overall_summary = results.get('overall_summary', {})
    
    # Create a fallback if the overall summary is missing required metrics
    if 'goal_reached_proportion' not in overall_summary or 'next_cell_lava_proportion' not in overall_summary:
        overall_summary = {
            'goal_reached_proportion': 0.0,
            'next_cell_lava_proportion': 1.0
        }
    
    return overall_summary

def evaluate_trial(base_config: Dict[str, Any], params: Dict[str, Any], 
                  trial_dir: str, n_agents: int, reduced_timesteps_factor: float) -> Tuple[float, float]:
    """Train and evaluate multiple agents for a trial."""
    agent_results = []
    
    # Create a directory to store all configurations and results
    os.makedirs(trial_dir, exist_ok=True)
    
    # Save trial parameters
    params_path = os.path.join(trial_dir, "trial_params.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    # Train and evaluate each agent
    for i in range(n_agents):
        agent_dir = create_agent_config(
            base_config, 
            params, 
            trial_dir, 
            i+1, 
            reduced_timesteps_factor
        )
        try:
            results = train_and_evaluate_agent(agent_dir)
            agent_results.append(results)
            
            # Save individual agent results
            results_path = os.path.join(agent_dir, "agent_summary.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Successfully evaluated agent {i+1}")
        except Exception as e:
            print(f"Error evaluating agent {i+1}: {str(e)}")
            print(f"Agent directory: {agent_dir}")
    
    # Calculate mean metrics across all agents
    if agent_results:
        goal_proportions = [r.get('goal_reached_proportion', 0) for r in agent_results]
        lava_proportions = [r.get('next_cell_lava_proportion', 1) for r in agent_results]
        
        mean_goal = float(np.mean(goal_proportions))
        mean_lava = float(np.mean(lava_proportions))
    else:
        mean_goal = 0.0
        mean_lava = 1.0
    
    # Save aggregated results
    summary = {
        'params': params,
        'individual_results': agent_results,
        'mean_metrics': {
            'goal_reached_proportion': mean_goal,
            'next_cell_lava_proportion': mean_lava
        }
    }
    
    summary_path = os.path.join(trial_dir, "trial_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Trial summary saved to: {summary_path}")
    print(f"Mean goal proportion: {mean_goal:.3f}, Mean lava proportion: {mean_lava:.3f}")
    
    return mean_goal, mean_lava 