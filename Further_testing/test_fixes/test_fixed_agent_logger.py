#!/usr/bin/env python3
"""
Test script to verify the fixed agent_logger implementation with real agent prediction.
This script will test whether agent logs now contain proper paths and model inputs.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil
import tempfile

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import fixed AgentLogger implementation
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


def compare_paths(original_path, newly_generated_path):
    """
    Compare two paths to see if they differ substantially.
    
    Args:
        original_path: Original path from existing agent logs
        newly_generated_path: Newly generated path from fixed AgentLogger
        
    Returns:
        bool: True if paths differ substantially
    """
    if len(original_path) != len(newly_generated_path):
        return True
    
    # Check if any states are different
    different_states = 0
    for i in range(min(len(original_path), len(newly_generated_path))):
        if original_path[i] != newly_generated_path[i]:
            different_states += 1
    
    # Consider paths different if more than 20% of states differ
    return different_states > len(original_path) * 0.2


def create_backup_dir():
    """
    Create a backup directory for temporary files.
    
    Returns:
        str: Path to backup directory
    """
    backup_dir = os.path.join(project_root, "test_fixes", "agent_logger_backup")
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir


def test_fixed_agent_logger(agent_path, env_id, seed, max_states=5):
    """
    Test the fixed agent_logger implementation.
    
    Args:
        agent_path: Path to agent directory
        env_id: Environment ID
        seed: Environment seed
        max_states: Maximum number of states to test
    """
    print(f"Testing fixed agent_logger for {agent_path} in {env_id} with seed {seed}")
    
    # Create a backup of the existing agent_logger.py file
    backup_dir = create_backup_dir()
    original_file = os.path.join(project_root, "Agent_Evaluation", "AgentTooling", "agent_logger.py")
    backup_file = os.path.join(backup_dir, "agent_logger.py.bak")
    print(f"Backing up original file to {backup_file}")
    shutil.copy2(original_file, backup_file)
    
    # Create a temporary directory for output files
    temp_dir = tempfile.mkdtemp(dir=os.path.join(project_root, "test_fixes"))
    print(f"Created temporary directory for outputs: {temp_dir}")
    
    try:
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
        
        # Load agent
        print("Loading agent...")
        agent = load_agent(agent_path, config)
        
        # Get existing log file for comparison
        if os.path.isabs(agent_path) or agent_path.startswith("Agent_Storage/"):
            full_agent_dir = agent_path
        else:
            full_agent_dir = os.path.join(project_root, "Agent_Storage", agent_path)
            
        existing_log_file = os.path.join(full_agent_dir, "evaluation_logs", f"{env_id}-{seed}.json")
        
        # Load existing log file if it exists
        existing_log_data = None
        if os.path.exists(existing_log_file):
            print(f"Loading existing log file: {existing_log_file}")
            with open(existing_log_file, 'r') as f:
                existing_log_data = json.load(f)
        
        # Create an instance of the fixed AgentLogger
        from Agent_Evaluation.AgentTooling.agent_logger import AgentLogger
        logger = AgentLogger(env_id, seed, env_tensor, agent)
        
        # Test a limited number of states
        states_tested = 0
        
        # Iterate through a small region of the environment to test a few states
        height, width = env_tensor.shape
        for y in range(1, min(5, height - 1)):
            for x in range(1, min(5, width - 1)):
                cell_type = env_tensor[y, x]
                
                # Skip walls and goal as starting positions
                if cell_type == "wall" or cell_type == "goal":
                    continue
                
                # Test just one orientation for simplicity
                orientation = 0
                state_key = f"{x},{y},{orientation}"
                print(f"\nTesting from state: {state_key}...")
                
                # Run a single episode from this state
                try:
                    logger._run_episode_from_state(env, (x, y, orientation))
                    states_tested += 1
                except Exception as e:
                    print(f"Error running episode from state {state_key}: {e}")
                
                # Stop after testing max_states
                if states_tested >= max_states:
                    break
            
            if states_tested >= max_states:
                break
        
        # Compare with existing data if available
        if existing_log_data and "environment" in existing_log_data:
            print("\nComparing with existing log data...")
            
            old_env_data = existing_log_data["environment"]
            
            for state_key, state_data in logger.all_states_data.items():
                print(f"\nState: {state_key}")
                
                # Check if state exists in original data
                if state_key in old_env_data:
                    # Extract paths
                    new_path = state_data.get("path_taken", [])
                    
                    # Get old path (could be under "path_taken" or "path")
                    if "path_taken" in old_env_data[state_key]:
                        old_path = old_env_data[state_key]["path_taken"]
                    elif "path" in old_env_data[state_key]:
                        old_path = old_env_data[state_key]["path"]
                    else:
                        old_path = []
                    
                    # Compare paths
                    paths_differ = compare_paths(old_path, new_path)
                    
                    print(f"  Paths differ: {paths_differ}")
                    print(f"  Old path ({len(old_path)} steps): {old_path[:3]}{'...' if len(old_path) > 3 else ''}")
                    print(f"  New path ({len(new_path)} steps): {new_path[:3]}{'...' if len(new_path) > 3 else ''}")
                    
                    # Check model inputs
                    old_has_inputs = "model_inputs" in old_env_data[state_key]
                    new_has_inputs = "model_inputs" in state_data
                    
                    print(f"  Has model inputs - Old: {old_has_inputs}, New: {new_has_inputs}")
                    
                    # Check mask consistency in model inputs
                    if old_has_inputs and new_has_inputs:
                        old_inputs = old_env_data[state_key]["model_inputs"]
                        new_inputs = state_data["model_inputs"]
                        
                        # Check barrier_mask and lava_mask
                        for mask_name in ["barrier_mask", "lava_mask"]:
                            if mask_name in old_inputs and mask_name in new_inputs:
                                old_mask = old_inputs[mask_name]
                                new_mask = new_inputs[mask_name]
                                
                                # Simple check for mask consistency
                                old_shape = (len(old_mask), len(old_mask[0]) if old_mask else 0)
                                new_shape = (len(new_mask), len(new_mask[0]) if new_mask else 0)
                                
                                print(f"  {mask_name} dimensions - Old: {old_shape}, New: {new_shape}")
                                
                                # Check if masks for different states are different
                                masks_differ = (old_mask != new_mask)
                                print(f"  {mask_name} differs: {masks_differ}")
                else:
                    print("  Not present in original data")
        
        # Export to temporary file
        output_path = os.path.join(temp_dir, f"{env_id}-{seed}-test.json")
        
        # Export the data to JSON with the agent directory
        export_path = logger.export_to_json(output_path=output_path)
        print(f"\nAgent behavior data exported to: {export_path}")
        
        # Print summary of results
        print("\nTesting results:")
        print(f"States tested: {states_tested}")
        print(f"States with data: {len(logger.all_states_data)}")
        
    finally:
        # Restore original file
        print(f"Restoring original file from {backup_file}")
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, original_file)
        else:
            print("Warning: Backup file not found!")
        
        # Create a tarball of test results
        tarball_path = os.path.join(project_root, "test_fixes", "agent_logger_test_results.tar.gz")
        import tarfile
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(temp_dir, arcname=os.path.basename(temp_dir))
        
        print(f"Test results saved to: {tarball_path}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        print("Testing complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test fixed agent_logger implementation")
    parser.add_argument("--agent_path", default="LavaTests/Standard", help="Path to agent directory")
    parser.add_argument("--env_id", default="MiniGrid-LavaCrossingS11N5-v0", help="Environment ID")
    parser.add_argument("--seed", type=int, default=81102, help="Environment seed")
    parser.add_argument("--max_states", type=int, default=5, help="Maximum number of states to test")
    
    args = parser.parse_args()
    
    test_fixed_agent_logger(args.agent_path, args.env_id, args.seed, args.max_states) 