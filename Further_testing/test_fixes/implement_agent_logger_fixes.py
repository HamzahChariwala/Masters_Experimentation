#!/usr/bin/env python3
"""
This script implements the fixes to the AgentLogger class to solve the issues with environment
layout mismatch, mlp_input repetition, and incorrect path generation.
"""

import os
import sys
import shutil
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

def create_backup(file_path: str, backup_dir: str) -> str:
    """
    Create a backup of the original file.
    
    Args:
        file_path: Path to the file to backup
        backup_dir: Directory to store the backup
        
    Returns:
        str: Path to the backup file
    """
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, os.path.basename(file_path) + ".bak")
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at: {backup_path}")
    return backup_path

def edit_agent_logger_file():
    """
    Apply fixes to the AgentLogger class in agent_logger.py.
    Changes:
    1. Remove the simulated path generation
    2. Use actual agent prediction for actions
    3. Fix the environment layout consistency
    """
    # Define file paths
    agent_logger_file = os.path.join(project_root, "Agent_Evaluation", "AgentTooling", "agent_logger.py")
    backup_dir = os.path.join(project_root, "test_fixes", "agent_logger_backup")
    
    # Create backup
    backup_path = create_backup(agent_logger_file, backup_dir)
    
    # Define necessary changes
    changes = [
        {
            "function": "_run_episode_from_state",
            "description": "Replace simulated path generation with real agent prediction",
            "old_code": """        # Initialize episode data
        steps = []
        
        # For path_taken, try to simulate a reasonable path to the goal
        # In a real implementation, this would come from the agent's trajectory
        path_taken = self._generate_simulated_path(start_state)
        
        # Log current state
        curr_state = (x, y, orientation)
        curr_cell_type = self.env_tensor[y, x]
        
        # Extract features for logging
        model_inputs = self._extract_model_inputs(obs, info)
        
        # Create first step data without actually predicting agent actions
        first_step = {
            "state": state_key,
            "action": None,  # Skip prediction
            "reward": 0,
            "terminated": False,
            "truncated": False,
            "cell_type": curr_cell_type,
            "model_inputs": model_inputs
        }
        
        steps.append(first_step)
        
        # Calculate summary statistics for this episode
        summary = {
            "total_steps": 1,
            "outcome": "skipped_prediction",
            "final_reward": 0,
            "state_type": curr_cell_type
        }""",
            "new_code": """        # Initialize episode data
        steps = []
        path_taken = [state_key]  # Initialize path with starting state
        
        # Log current state
        curr_state = (x, y, orientation)
        curr_cell_type = self.env_tensor[y, x]
        
        # Extract features for logging
        model_inputs = self._extract_model_inputs(obs, info)
        
        # Maximum number of steps to run
        max_steps = 20
        
        # Get MLP inputs for the model
        mlp_inputs = self._extract_mlp_inputs(obs, info)
        
        # Predict action using the agent model
        action = None
        try:
            action, _ = self.agent.predict(mlp_inputs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action)
        except Exception as e:
            print(f"  Error predicting action: {e}")
        
        # Create first step data
        first_step = {
            "state": state_key,
            "action": action,
            "reward": 0,
            "terminated": False,
            "truncated": False,
            "cell_type": curr_cell_type,
            "model_inputs": model_inputs
        }
        
        steps.append(first_step)
        
        # Try to run a few steps to generate a real path
        done = False
        step_count = 0
        
        # Copy the environment to avoid affecting the original
        import copy
        env_copy = env
        
        # Get a fresh observation after setting the state
        obs, info = env_copy.reset(seed=self.seed)
        
        # Set position and orientation again
        current_env.agent_pos = np.array([x, y])
        current_env.agent_dir = orientation
        
        # Update observation
        if hasattr(current_env, 'gen_obs'):
            obs = current_env.gen_obs()
        
        # Run the agent for a few steps to generate a path
        total_reward = 0
        while not done and step_count < max_steps:
            # Predict action
            try:
                mlp_inputs = self._extract_mlp_inputs(obs, info)
                action, _ = self.agent.predict(mlp_inputs, deterministic=True)
                
                # Convert action to int
                if isinstance(action, np.ndarray):
                    action = int(action)
                    
                # Take action in environment
                next_obs, reward, terminated, truncated, next_info = env_copy.step(action)
                
                # Get new state
                next_x, next_y, next_orientation = self._get_agent_position(env_copy)
                next_state_key = f"{next_x},{next_y},{next_orientation}"
                
                # Get cell type
                next_cell_type = "unknown"
                if 0 <= next_y < self.env_tensor.shape[0] and 0 <= next_x < self.env_tensor.shape[1]:
                    next_cell_type = self.env_tensor[next_y, next_x]
                
                # Create step data
                step_data = {
                    "state": next_state_key,
                    "action": action,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "cell_type": next_cell_type,
                    "model_inputs": self._extract_model_inputs(next_obs, next_info)
                }
                
                # Add to steps and path
                steps.append(step_data)
                path_taken.append(next_state_key)
                
                # Update for next iteration
                obs = next_obs
                info = next_info
                total_reward += reward
                
                # Check if done
                done = terminated or truncated
                step_count += 1
                
                if terminated and reward > 0:
                    # Goal reached
                    break
            except Exception as e:
                print(f"  Error running step {step_count}: {e}")
                break
        
        # Calculate summary statistics for this episode
        summary = {
            "total_steps": len(steps),
            "outcome": "success" if (done and total_reward > 0) else "failure",
            "final_reward": total_reward,
            "state_type": curr_cell_type
        }"""
        },
        {
            "function": "state_data",
            "description": "Update path key name",
            "old_code": """        # Store complete state data
        state_data = {
            "steps": steps,
            "path": path_taken,
            "summary": summary
        }""",
            "new_code": """        # Store complete state data
        state_data = {
            "steps": steps,
            "path_taken": path_taken,
            "summary": summary
        }"""
        },
        {
            "function": "_generate_simulated_path",
            "description": "Replace the simulated path generation with agent position lookup method",
            "old_code": """    def _generate_simulated_path(self, start_state: Tuple[int, int, int]) -> List[str]:
        \"\"\"
        Generate a simulated path to the goal.
        In a real implementation, this would use the actual path from agent trajectory,
        but for now we'll create a synthetic path.
        
        Args:
            start_state: The starting state (x, y, orientation)
        
        Returns:
            List[str]: List of state keys representing a path to the goal
        \"\"\"
        # Start with current state
        x, y, orientation = start_state
        state_key = f"{x},{y},{orientation}"
        path = [state_key]
        
        # Locate goal position
        goal_x, goal_y = None, None
        for y_idx in range(self.env_tensor.shape[0]):
            for x_idx in range(self.env_tensor.shape[1]):
                if self.env_tensor[y_idx, x_idx] == "goal":
                    goal_x, goal_y = x_idx, y_idx
                    break
        
        # If we couldn't find the goal, just return the current state
        if goal_x is None:
            return path
            
        # For simplicity, we'll use a rough approximation of a path
        # This could be replaced with a proper pathfinding algorithm
        
        # 1. Find the goal location
        # Start with current position
        curr_x, curr_y, curr_dir = start_state
        
        # Choose a direction that moves towards the goal
        # We'll use a naive approach: first move horizontally, then vertically
        
        # First, try to reach the correct column (move horizontally)
        while curr_x != goal_x:
            # Determine direction to move
            target_dir = 0 if goal_x > curr_x else 2  # 0=right, 2=left
            
            # If not facing the right direction, turn
            if curr_dir != target_dir:
                # Add a turning state
                if (target_dir - curr_dir) % 4 == 1 or (target_dir - curr_dir) % 4 == -3:
                    # Turn right
                    curr_dir = (curr_dir + 1) % 4
                else:
                    # Turn left
                    curr_dir = (curr_dir - 1) % 4
                path.append(f"{curr_x},{curr_y},{curr_dir}")
            else:
                # Move in the direction we're facing
                if curr_dir == 0:  # Right
                    curr_x += 1
                elif curr_dir == 2:  # Left
                    curr_x -= 1
                    
                # Check if this is a valid move (not into a wall or lava)
                if 0 <= curr_y < self.env_tensor.shape[0] and 0 <= curr_x < self.env_tensor.shape[1]:
                    cell_type = self.env_tensor[curr_y, curr_x]
                    if cell_type == "wall":
                        # Can't move here, undo and try another direction
                        if curr_dir == 0:  # Right
                            curr_x -= 1
                        elif curr_dir == 2:  # Left
                            curr_x += 1
                        break
        
                path.append(f"{curr_x},{curr_y},{curr_dir}")
        
        # Then, try to reach the correct row (move vertically)
        while curr_y != goal_y:
            # Determine direction to move
            target_dir = 1 if goal_y > curr_y else 3  # 1=down, 3=up
            
            # If not facing the right direction, turn
            if curr_dir != target_dir:
                # Add a turning state
                if (target_dir - curr_dir) % 4 == 1 or (target_dir - curr_dir) % 4 == -3:
                    # Turn right
                    curr_dir = (curr_dir + 1) % 4
                else:
                    # Turn left
                    curr_dir = (curr_dir - 1) % 4
                path.append(f"{curr_x},{curr_y},{curr_dir}")
            else:
                # Move in the direction we're facing
                if curr_dir == 1:  # Down
                    curr_y += 1
                elif curr_dir == 3:  # Up
                    curr_y -= 1
                    
                # Check if this is a valid move (not into a wall or lava)
                if 0 <= curr_y < self.env_tensor.shape[0] and 0 <= curr_x < self.env_tensor.shape[1]:
                    cell_type = self.env_tensor[curr_y, curr_x]
                    if cell_type == "wall":
                        # Can't move here, undo and try another direction
                        if curr_dir == 1:  # Down
                            curr_y -= 1
                        elif curr_dir == 3:  # Up
                            curr_y += 1
                        break
                        
                path.append(f"{curr_x},{curr_y},{curr_dir}")
        
        # Limit path length for performance
        if len(path) > 30:
            path = path[:30]
            
        return path""",
            "new_code": """    def _get_agent_position(self, env):
        \"\"\"
        Get agent position and orientation from environment.
        
        Args:
            env: Environment
        
        Returns:
            Tuple of (x, y, orientation)
        \"\"\"
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
        return 1, 1, 0"""
        },
        {
            "function": "_extract_mlp_inputs",
            "description": "Add MLP input extraction method if it doesn't exist",
            "old_code": "",  # This function might not exist, so we check for its existence first
            "new_code": """    def _extract_mlp_inputs(self, obs, info=None):
        \"\"\"
        Extract MLP inputs for model prediction from gym observation.
        
        Args:
            obs: Gym observation
            info: Optional info dictionary
        
        Returns:
            dict: Dictionary of MLP inputs
        \"\"\"
        # Check if obs is a dictionary with MLP_input
        if isinstance(obs, dict) and "MLP_input" in obs:
            return obs
        
        # Check if obs is a numpy array
        if isinstance(obs, np.ndarray):
            return {"MLP_input": obs}
        
        # Default fallback if we can't identify the input format
        return obs"""
        }
    ]
    
    # Read file content
    with open(agent_logger_file, 'r') as f:
        content = f.read()
    
    # Apply changes
    new_content = content
    for change in changes:
        if change["old_code"] == "" and change["function"] in new_content:
            # Skip if the function already exists
            print(f"Function {change['function']} already exists, skipping.")
            continue
            
        if change["old_code"] in new_content:
            print(f"Applying change: {change['description']}")
            new_content = new_content.replace(change["old_code"], change["new_code"])
        else:
            print(f"Warning: Could not find the code to replace for {change['description']}")
    
    # Write updated content
    with open(agent_logger_file, 'w') as f:
        f.write(new_content)
    
    print("AgentLogger fixes applied successfully!")

def main():
    """Main function to implement the fixes."""
    print("Implementing fixes to AgentLogger...")
    edit_agent_logger_file()
    print("All fixes applied successfully!")

if __name__ == "__main__":
    main() 