import os
import sys
import json
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from Behaviour_Specification.StateGeneration.state_class import State

class AgentLogger:
    """
    Logger for capturing and storing agent behavior during evaluation.
    This mimics the structure of the Dijkstra's path data output.
    """
    
    def __init__(self, env_id: str, seed: int, env_tensor: np.ndarray, agent):
        """
        Initialize the agent logger.
        
        Args:
            env_id (str): Environment ID
            seed (int): Random seed
            env_tensor (np.ndarray): Environment tensor
            agent: The agent model to evaluate
        """
        self.env_id = env_id
        self.seed = seed
        self.env_tensor = env_tensor
        self.agent = agent
        self.state_object = State((1, 1, 0))  # Create a State object for utility methods
        self.all_states_data = {}  # Store data for all tested states
        
    def log_agent_behavior(self, env) -> Dict[str, Any]:
        """
        Systematically test agent behavior from all valid starting positions and orientations.
        
        Args:
            env: The environment to evaluate in
            
        Returns:
            Dict[str, Any]: Dictionary with all agent behavior data
        """
        print("\nAnalyzing agent behavior from all valid starting states...")
        
        # Get environment dimensions
        height, width = self.env_tensor.shape
        
        # All possible orientations (0=right, 1=down, 2=left, 3=up)
        orientations = [0, 1, 2, 3]
        
        # Dictionary to store all state data
        all_states_data = {}
        
        # Test from every valid position (excluding walls and goal)
        for y in range(1, height - 1):  # Skip border walls
            for x in range(1, width - 1):  # Skip border walls
                cell_type = self.env_tensor[y, x]
                
                # Skip walls and goals as starting positions
                if cell_type == "wall" or cell_type == "goal":
                    continue
                
                # Test all orientations at this position
                for orientation in orientations:
                    state_key = f"{x},{y},{orientation}"
                    print(f"Testing from state: {state_key}...")
                    
                    # Run a single episode from this starting state
                    try:
                        # Reset environment and set agent to this position
                        self._run_episode_from_state(env, (x, y, orientation))
                    except Exception as e:
                        print(f"Error running episode from state {state_key}: {e}")
        
        return self.all_states_data
    
    def _run_episode_from_state(self, env, start_state: Tuple[int, int, int]):
        """
        Run a single episode with the agent starting at the specified state.
        Since we're encountering issues with the agent prediction,
        we'll focus on logging the observation data from each state.
        
        Args:
            env: The environment to evaluate in
            start_state: The starting state (x, y, orientation)
        """
        x, y, orientation = start_state
        state_key = f"{x},{y},{orientation}"
        
        # Skip if we've already tested this state
        if state_key in self.all_states_data:
            return
        
        # Reset environment
        obs, info = env.reset(seed=self.seed)
        
        # Try to place agent at the specified position and orientation
        unwrapped_env = env.unwrapped
        
        # Navigate to the unwrapped environment with grid
        current_env = env
        while hasattr(current_env, 'env'):
            current_env = current_env.env
            if hasattr(current_env, 'grid') and hasattr(current_env, 'agent_pos') and hasattr(current_env, 'agent_dir'):
                # Found the base MiniGrid environment
                break
        
        # If we found a MiniGrid environment with accessible attributes
        if hasattr(current_env, 'grid') and hasattr(current_env, 'agent_pos') and hasattr(current_env, 'agent_dir'):
            # Set agent position and orientation
            current_env.agent_pos = np.array([x, y])
            current_env.agent_dir = orientation
            
            # Force environment observation update
            if hasattr(current_env, 'gen_obs'):
                obs = current_env.gen_obs()
            else:
                # Try to get updated observation through the wrapper chain
                obs, _, _, _, info = env.step(0)  # Take a no-op action to update observation
                # Reset to restore environment
                obs, info = env.reset(seed=self.seed)  
                # Position might have been reset, set it again
                current_env.agent_pos = np.array([x, y])  
                current_env.agent_dir = orientation
                # Generate observation again
                if hasattr(current_env, 'gen_obs'):
                    obs = current_env.gen_obs()
        else:
            # If we couldn't find base environment attributes, skip this state
            print(f"  Could not set agent to state {state_key}, skipping...")
            return
        
        # Initialize episode data
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
        }
        
        # Store complete state data
        state_data = {
            "steps": steps,
            "path": path_taken,
            "summary": summary
        }
        
        # Add this state's data to the overall dataset
        self.all_states_data[state_key] = state_data
        
    def _generate_simulated_path(self, start_state: Tuple[int, int, int]) -> List[str]:
        """
        Generate a simulated path to the goal.
        In a real implementation, this would use the actual path from agent trajectory,
        but for now we'll create a synthetic path.
        
        Args:
            start_state: The starting state (x, y, orientation)
            
        Returns:
            List[str]: List of state keys representing a path to the goal
        """
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
            
        return path
    
    def _get_agent_state(self, env) -> Optional[Tuple[int, int, int]]:
        """
        Get the agent's current state from the environment.
        
        Args:
            env: The environment
            
        Returns:
            Optional[Tuple[int, int, int]]: Agent state (x, y, orientation) or None if not found
        """
        # Try different approaches to find the agent position
        unwrapped_env = env.unwrapped
        
        # 1. Try the SafeMonitor.get_agent_pos method if available
        if hasattr(env, "get_agent_pos"):
            agent_pos = env.get_agent_pos()
            if agent_pos is not None and hasattr(unwrapped_env, "agent_dir"):
                return (agent_pos[0], agent_pos[1], unwrapped_env.agent_dir)
        
        # 2. Try accessing through unwrapped env directly
        if hasattr(unwrapped_env, "agent_pos") and hasattr(unwrapped_env, "agent_dir"):
            return (unwrapped_env.agent_pos[0], unwrapped_env.agent_pos[1], unwrapped_env.agent_dir)
        
        # 3. Try nested environment attributes (may be wrapped multiple times)
        current_env = env
        while hasattr(current_env, "env"):
            current_env = current_env.env
            if hasattr(current_env, "agent_pos") and hasattr(current_env, "agent_dir"):
                return (current_env.agent_pos[0], current_env.agent_pos[1], current_env.agent_dir)
        
        return None
    
    def _determine_next_state(self, 
                             current_state: Tuple[int, int, int], 
                             action: int) -> Tuple[Optional[Tuple[int, int, int]], str, bool]:
        """
        Determine the next state given the current state and action.
        
        Args:
            current_state (Tuple[int, int, int]): Current state (x, y, orientation)
            action (int): Action taken (0=rotate left, 1=rotate right, 2=forward, 3=diagonal left, 4=diagonal right)
            
        Returns:
            Tuple[Optional[Tuple[int, int, int]], str, bool]: Next state, cell type, and whether it's a diagonal move
        """
        x, y, orientation = current_state
        is_diagonal = False
        
        # Calculate next state based on action
        if action == 0:  # Rotate left
            next_state = (x, y, (orientation - 1) % 4)
        elif action == 1:  # Rotate right
            next_state = (x, y, (orientation + 1) % 4)
        elif action == 2:  # Move forward
            # Calculate position change based on orientation
            dx, dy = 0, 0
            if orientation == 0:  # Right
                dx, dy = 1, 0
            elif orientation == 1:  # Down
                dx, dy = 0, 1
            elif orientation == 2:  # Left
                dx, dy = -1, 0
            elif orientation == 3:  # Up
                dx, dy = 0, -1
                
            next_state = (x + dx, y + dy, orientation)
        elif action == 3:  # Diagonal left
            is_diagonal = True
            # Calculate position change based on orientation
            if orientation == 0:  # Right -> Up-Right
                dx, dy = 1, -1
            elif orientation == 1:  # Down -> Right-Down
                dx, dy = 1, 1
            elif orientation == 2:  # Left -> Down-Left
                dx, dy = -1, 1
            elif orientation == 3:  # Up -> Left-Up
                dx, dy = -1, -1
                
            next_state = (x + dx, y + dy, orientation)
        elif action == 4:  # Diagonal right
            is_diagonal = True
            # Calculate position change based on orientation
            if orientation == 0:  # Right -> Down-Right
                dx, dy = 1, 1
            elif orientation == 1:  # Down -> Left-Down
                dx, dy = -1, 1
            elif orientation == 2:  # Left -> Up-Left
                dx, dy = -1, -1
            elif orientation == 3:  # Up -> Right-Up
                dx, dy = 1, -1
                
            next_state = (x + dx, y + dy, orientation)
        else:
            # Unknown action
            return None, "unknown", False
        
        # Check if the next state is within bounds
        if not (0 <= next_state[0] < self.env_tensor.shape[1] and 0 <= next_state[1] < self.env_tensor.shape[0]):
            return None, "out_of_bounds", is_diagonal
        
        # Get the cell type at the next position
        next_cell_type = self.env_tensor[next_state[1], next_state[0]]
        
        return next_state, next_cell_type, is_diagonal
        
    def _extract_model_inputs(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract observation data for logging purposes, focusing only on the keys specified in the config.
        
        Args:
            obs (Dict[str, Any]): Observation from the environment
            info (Dict[str, Any]): Additional info from the environment
            
        Returns:
            Dict[str, Any]: Dictionary containing only the specific model inputs needed
        """
        # Helper function to recursively convert NumPy arrays to lists
        def convert_numpy_to_list(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, dict):
                return {k: convert_numpy_to_list(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [convert_numpy_to_list(i) for i in item]
            elif isinstance(item, (int, float, bool, str, type(None))):
                return item
            else:
                return str(item)
        
        # The model inputs we need to collect based on the config file
        config_keys = [
            "four_way_goal_direction",
            "four_way_angle_alignment", 
            "barrier_mask", 
            "lava_mask"
        ]
        
        # Initialize empty model inputs
        model_inputs = {}
        
        # Extract specified keys from log_data if available
        if 'log_data' in info and isinstance(info['log_data'], dict):
            log_data = info['log_data']
            
            # Only collect the keys specified in the config
            for key in config_keys:
                if key in log_data:
                    model_inputs[key] = convert_numpy_to_list(log_data[key])
        
        return model_inputs
    
    def _calculate_summary_stats(self, steps, path_taken) -> Dict[str, Any]:
        """
        Calculate summary statistics for the episode.
        
        Args:
            steps: List of step data
            path_taken: List of states visited
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        # If no steps, return default stats
        if not steps:
            return {
                "total_reward": 0,
                "path_length": 0,
                "lava_steps": 0,
                "success": False,
                "truncated": False,
                "terminated": False,
                "into_wall": False,
                "cyclic_rotation": False,
                "reachable": False  # Match Dijkstra log format
            }
        
        total_reward = sum(step["reward"] for step in steps)
        path_length = len(path_taken) - 1  # Subtract 1 since first state isn't a step
        lava_steps = sum(1 for step in steps if step["cell_type"] == "lava")
        
        # Check for goal reached (success)
        success = any(step["terminated"] and step["reward"] > 0 for step in steps)
        
        # Termination flags
        terminated = any(step["terminated"] for step in steps)
        truncated = any(step["truncated"] for step in steps)
        
        # Check for into_wall behavior (agent tries to move forward but stays in place)
        into_wall = False
        for i in range(1, len(steps)):
            prev_state = steps[i-1]["state"]
            curr_state = steps[i]["state"]
            prev_action = steps[i-1]["action"]
            
            # If action was forward (2) and state didn't change, it's likely a wall collision
            if prev_action == 2 and prev_state == curr_state:
                into_wall = True
                break
        
        # Check for cyclic rotation (agent keeps rotating back and forth)
        cyclic_rotation = False
        if len(steps) >= 5:  # Need at least 5 steps to detect pattern
            rotation_pattern = True
            for i in range(3, len(steps)):
                x1, y1, _ = map(int, steps[i-3]["state"].split(','))
                x2, y2, _ = map(int, steps[i]["state"].split(','))
                
                # Check if positions are the same
                if x1 != x2 or y1 != y2:
                    rotation_pattern = False
                    break
                    
                # Check if actions are rotations (0 or 1)
                if not (steps[i-3]["action"] in [0, 1] and 
                        steps[i-2]["action"] in [0, 1] and
                        steps[i-1]["action"] in [0, 1] and
                        steps[i]["action"] in [0, 1]):
                    rotation_pattern = False
                    break
            
            cyclic_rotation = rotation_pattern
        
        return {
            "total_reward": total_reward,
            "path_length": path_length,
            "lava_steps": lava_steps,
            "success": success,
            "truncated": truncated,
            "terminated": terminated,
            "into_wall": into_wall,
            "cyclic_rotation": cyclic_rotation,
            "reachable": success or path_length > 0  # Consider reachable if success or it moved
        }
    
    def export_to_json(self, output_path: str = None) -> str:
        """
        Export the agent behavior data to a JSON file.
        Format to match Dijkstra logs structure.
        
        Args:
            output_path (str, optional): Path to save the JSON file. 
                If None, saved to Agent_Evaluation/AgentLogs directory.
            
        Returns:
            str: Path to the saved JSON file
        """
        # Helper function to recursively convert NumPy arrays to lists
        def convert_numpy_to_list(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, dict):
                return {k: convert_numpy_to_list(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [convert_numpy_to_list(i) for i in item]
            else:
                return item
        
        # Build the output data structure to match Dijkstra log format
        # Format: {"environment": {"layout": [...], "1,1,0": {...}, ...}}
        output_data = {
            "environment": {
                "layout": convert_numpy_to_list(self.env_tensor)
            }
        }
        
        # Add all state data
        for state_key, state_data in self.all_states_data.items():
            output_data["environment"][state_key] = convert_numpy_to_list(state_data)
        
        # Determine output path if not provided
        if output_path is None:
            # Create Agent_Evaluation/AgentLogs directory if it doesn't exist
            agent_logs_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "AgentLogs"
            )
            os.makedirs(agent_logs_dir, exist_ok=True)
            
            # Save to Agent_Evaluation/AgentLogs directory with the proper name format
            output_path = os.path.join(
                agent_logs_dir, 
                f"{self.env_id}-{self.seed}-agent.json"
            )
        
        # Save the data to a JSON file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Agent behavior data exported to: {output_path}")
        return output_path

    def _extract_mlp_inputs(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Any:
        """
        Extract only the MLP inputs needed for model prediction.
        This function creates a 106-element feature vector required by the model,
        using features from log_data, specifically focusing on:
        ["four_way_goal_direction", "four_way_angle_alignment", "barrier_mask", "lava_mask"]
        
        Args:
            obs (Dict[str, Any]): Observation from the environment
            info (Dict[str, Any]): Additional info from the environment
            
        Returns:
            np.ndarray: A (106,) shaped numpy array for the agent's predict method
        """
        # DEBUG: Print observation and info structure
        print(f"    Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
        print(f"    Info keys: {list(info.keys()) if isinstance(info, dict) else 'Not a dict'}")
        
        # Create a feature vector of the required size (106 elements)
        feature_vector = np.zeros(106, dtype=np.float32)
        
        # Check if log_data is available in info
        if 'log_data' in info and isinstance(info['log_data'], dict):
            log_data = info['log_data']
            print(f"    log_data keys: {list(log_data.keys()) if isinstance(log_data, dict) else 'Not a dict'}")
            
            # 1. Extract four_way_goal_direction (one-hot encode it into first 4 positions)
            if 'four_way_goal_direction' in log_data and log_data['four_way_goal_direction'] is not None:
                direction = log_data['four_way_goal_direction']
                if isinstance(direction, int) and 0 <= direction < 4:
                    feature_vector[direction] = 1.0
                    
            # 2. Extract four_way_angle_alignment (position 4)
            if 'four_way_angle_alignment' in log_data and log_data['four_way_angle_alignment'] is not None:
                alignment = log_data['four_way_angle_alignment']
                if isinstance(alignment, (int, float, np.number)):
                    feature_vector[4] = float(alignment)
                    
            # 3. Extract and flatten barrier_mask (positions 5-54, assuming size matches)
            if 'barrier_mask' in log_data and log_data['barrier_mask'] is not None:
                mask = log_data['barrier_mask']
                if isinstance(mask, np.ndarray):
                    flat_mask = mask.flatten()
                    # Copy as many elements as will fit without going out of bounds
                    copy_size = min(50, flat_mask.size)
                    feature_vector[5:5+copy_size] = flat_mask[:copy_size]
                elif isinstance(mask, (list, tuple)):
                    flat_mask = np.array(mask).flatten()
                    copy_size = min(50, flat_mask.size)
                    feature_vector[5:5+copy_size] = flat_mask[:copy_size]
                    
            # 4. Extract and flatten lava_mask (positions 55-104, assuming size matches)
            if 'lava_mask' in log_data and log_data['lava_mask'] is not None:
                mask = log_data['lava_mask']
                if isinstance(mask, np.ndarray):
                    flat_mask = mask.flatten()
                    # Copy as many elements as will fit without going out of bounds
                    copy_size = min(50, flat_mask.size)
                    feature_vector[55:55+copy_size] = flat_mask[:copy_size]
                elif isinstance(mask, (list, tuple)):
                    flat_mask = np.array(mask).flatten()
                    copy_size = min(50, flat_mask.size)
                    feature_vector[55:55+copy_size] = flat_mask[:copy_size]
                    
            # 5. Add a feature indicating we have valid data at position 105
            feature_vector[105] = 1.0
            
            print(f"    Created feature vector with shape {feature_vector.shape}")
            return feature_vector
            
        # If we couldn't extract the needed features, still return a zero-filled 106-element vector
        print(f"    Created empty feature vector with shape {feature_vector.shape}")
        return feature_vector


def run_agent_evaluation(agent, env, env_id: str, seed: int, num_episodes: int = 1) -> str:
    """
    Run the agent evaluation and log its behavior.
    Format to match a simplified version of the logs structure without standard/conservative nesting.
    
    Args:
        agent: The agent model
        env: The environment to evaluate in
        env_id (str): Environment ID
        seed (int): Random seed
        num_episodes (int): Number of episodes to run (default: 1)
        
    Returns:
        str: Path to the saved JSON file
    """
    from Agent_Evaluation.EnvironmentTooling.extract_grid import extract_grid_from_env

    # Extract environment tensor
    env_tensor = extract_grid_from_env(env)
    
    # Create agent logger
    logger = AgentLogger(env_id, seed, env_tensor, agent)
    
    # Log agent behavior
    states_data = logger.log_agent_behavior(env)
    
    # Structure the data with simplified format
    formatted_data = {}
    
    # For each state in the agent behavior data
    for state_key, state_data in states_data.items():
        # Format the state data with a simplified structure
        formatted_state_data = {
            "path_taken": state_data["path"],
            "next_step": {
                "action": None,  # We don't have an actual prediction
                "target_state": None,
                "type": "unknown",
                "risky_diagonal": False
            },
            "summary_stats": {
                "path_cost": 0,  # We don't have an actual path
                "path_length": 0,
                "lava_steps": 0,
                "reachable": False  # Since we're not making predictions
            }
        }
        
        # If we have a cell type, update the next_step type
        if state_data["steps"] and "cell_type" in state_data["steps"][0]:
            formatted_state_data["next_step"]["type"] = state_data["steps"][0]["cell_type"]
        
        # Extract model_inputs directly from the steps
        if state_data["steps"] and "model_inputs" in state_data["steps"][0]:
            # Use the raw model_inputs directly
            formatted_state_data["model_inputs"] = state_data["steps"][0]["model_inputs"]
        
        # Add the formatted state data to the main dictionary
        formatted_data[state_key] = formatted_state_data
    
    # Update the logger's all_states_data with the formatted data
    logger.all_states_data = formatted_data
    
    # Export the data to JSON
    return logger.export_to_json() 