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
        This version actually runs the agent model to predict actions and generate real paths.
        
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
        max_depth = 10
        found_base_env = False
        
        for _ in range(max_depth):
            if hasattr(current_env, 'grid') and hasattr(current_env, 'agent_pos') and hasattr(current_env, 'agent_dir'):
                # Found the base MiniGrid environment
                found_base_env = True
                break
            
            if hasattr(current_env, 'env'):
                current_env = current_env.env
            elif hasattr(current_env, 'unwrapped'):
                current_env = current_env.unwrapped
            else:
                break
        
        # If we found a MiniGrid environment with accessible attributes
        if found_base_env:
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
        }
        
        # Store complete state data
        state_data = {
            "steps": steps,
            "path_taken": path_taken,
            "summary": summary
        }
        
        # Add this state's data to the overall dataset
        self.all_states_data[state_key] = state_data
    
    def _get_agent_position(self, env):
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
                    # For barrier_mask and lava_mask, transpose them like we do for env_tensor
                    if key in ["barrier_mask", "lava_mask"] and isinstance(log_data[key], np.ndarray):
                        model_inputs[key] = convert_numpy_to_list(log_data[key].T)
                    else:
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
    
    def export_to_json(self, output_path: str = None, agent_dir: str = None) -> str:
        """
        Export the agent behavior data to a JSON file.
        Format to match Dijkstra logs structure.
        
        Args:
            output_path (str, optional): Path to save the JSON file. 
                If None, saved to agent_dir/evaluation_logs directory.
            agent_dir (str, optional): Path to the agent directory.
                If None, defaults to saving in Agent_Evaluation/AgentLogs.
            
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
                # Transpose the env_tensor before converting to list
                "layout": convert_numpy_to_list(self.env_tensor.T)
            }
        }
        
        # Add all state data
        for state_key, state_data in self.all_states_data.items():
            output_data["environment"][state_key] = convert_numpy_to_list(state_data)
        
        # Determine output path if not provided
        if output_path is None:
            if agent_dir is not None:
                # Create evaluation_logs directory in the agent's folder
                # First, verify that the agent_dir exists
                if not os.path.exists(agent_dir):
                    print(f"Warning: Agent directory {agent_dir} does not exist. Creating it...")
                    os.makedirs(agent_dir, exist_ok=True)
                
                agent_logs_dir = os.path.join(agent_dir, "evaluation_logs")
                os.makedirs(agent_logs_dir, exist_ok=True)
                print(f"Created evaluation_logs directory: {agent_logs_dir}")
                
                # Save to agent's evaluation_logs directory with the name format without "-agent"
                output_path = os.path.join(
                    agent_logs_dir, 
                    f"{self.env_id}-{self.seed}.json"
                )
            else:
                # Fallback to the previous behavior if agent_dir is not provided
                agent_logs_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    "AgentLogs"
                )
                os.makedirs(agent_logs_dir, exist_ok=True)
                
                # Save to Agent_Evaluation/AgentLogs directory with the old name format
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
                    # Transpose mask to be consistent with env_tensor handling
                    mask = mask.T
                    flat_mask = mask.flatten()
                    # Copy as many elements as will fit without going out of bounds
                    copy_size = min(50, flat_mask.size)
                    feature_vector[5:5+copy_size] = flat_mask[:copy_size]
                elif isinstance(mask, (list, tuple)):
                    # Convert to numpy array, transpose, and flatten
                    mask_array = np.array(mask).T
                    flat_mask = mask_array.flatten()
                    copy_size = min(50, flat_mask.size)
                    feature_vector[5:5+copy_size] = flat_mask[:copy_size]
                    
            # 4. Extract and flatten lava_mask (positions 55-104, assuming size matches)
            if 'lava_mask' in log_data and log_data['lava_mask'] is not None:
                mask = log_data['lava_mask']
                if isinstance(mask, np.ndarray):
                    # Transpose mask to be consistent with env_tensor handling
                    mask = mask.T
                    flat_mask = mask.flatten()
                    # Copy as many elements as will fit without going out of bounds
                    copy_size = min(50, flat_mask.size)
                    feature_vector[55:55+copy_size] = flat_mask[:copy_size]
                elif isinstance(mask, (list, tuple)):
                    # Convert to numpy array, transpose, and flatten
                    mask_array = np.array(mask).T
                    flat_mask = mask_array.flatten()
                    copy_size = min(50, flat_mask.size)
                    feature_vector[55:55+copy_size] = flat_mask[:copy_size]
                    
            # 5. Add a feature indicating we have valid data at position 105
            feature_vector[105] = 1.0
            
            print(f"    Created feature vector with shape {feature_vector.shape}")
            return feature_vector
            
        # If we couldn't extract the needed features, still return a zero-filled 106-element vector
        print(f"    Created empty feature vector with shape {feature_vector.shape}")
        return feature_vector


def run_agent_evaluation(agent, env, env_id: str, seed: int, num_episodes: int = 1, logger=None, agent_dir: str = None) -> str:
    """
    Run the agent evaluation and log its behavior.
    Format to match a simplified version of the logs structure without standard/conservative nesting.
    All orientation values use the agent/environment convention (0=right, 1=down, 2=left, 3=up).
    
    Args:
        agent: The agent model
        env: The environment to evaluate in
        env_id (str): Environment ID
        seed (int): Random seed
        num_episodes (int): Number of episodes to run (default: 1)
        logger (AgentLogger, optional): Pre-populated logger instance. If provided, skips data collection.
        agent_dir (str, optional): Path to the agent directory for saving logs.
            If provided, logs will be saved to {agent_dir}/evaluation_logs/{env_id}-{seed}.json
        
    Returns:
        str: Path to the saved JSON file
    """
    from Agent_Evaluation.EnvironmentTooling.extract_grid import extract_grid_from_env

    # If a logger is provided, use it
    if logger is not None:
        print("Using pre-populated logger")
        states_data = logger.all_states_data
    else:
        # Extract environment tensor
        env_tensor = extract_grid_from_env(env)
        
        # Create agent logger
        logger = AgentLogger(env_id, seed, env_tensor, agent)
        
        # Log agent behavior
        states_data = logger.log_agent_behavior(env)
    
    # Structure the data with simplified format
    formatted_data = {}
    
    # For tracking the number of risky diagonals found
    risky_diagonals_found = 0
    
    # For each state in the agent behavior data
    for state_key, state_data in states_data.items():
        # Get path_taken from state_data
        path_taken = state_data["path_taken"] if "path_taken" in state_data else state_data.get("path", [])
        
        # Calculate actual path_length based on path_taken
        path_length = len(path_taken) - 1 if len(path_taken) > 0 else 0
        
        # Initialize next step values
        next_action = None
        next_target_state = None
        next_cell_type = "unknown"
        
        # Get the current cell type 
        x, y, _ = map(int, state_key.split(","))
        current_cell_type = logger.env_tensor[y, x]
        
        # If we have a path with at least one step
        if path_length > 0 and len(path_taken) >= 2:
            # Get the first and second states in the path
            curr_state_str = path_taken[0]
            next_state_str = path_taken[1]
            
            # Parse current and next state 
            curr_x, curr_y, curr_orient = map(int, curr_state_str.split(","))
            next_x, next_y, next_orient = map(int, next_state_str.split(","))
            
            # Create a State object to use its methods for move identification
            curr_state_tuple = (curr_x, curr_y, curr_orient)
            next_state_tuple = (next_x, next_y, next_orient)
            
            # Create State object for utility methods
            state_obj = State(curr_state_tuple)
            
            # Get Dijkstra orientation (since State class uses Dijkstra orientation)
            # Convert from agent orientation (0=right, 1=down, 2=left, 3=up)
            # to Dijkstra orientation (0=up, 1=right, 2=down, 3=left)
            agent_to_dijkstra = {0: 1, 1: 2, 2: 3, 3: 0}
            dijkstra_orient = agent_to_dijkstra[curr_orient]
            
            # Similarly convert next orientation
            dijkstra_next_orient = agent_to_dijkstra[next_orient]
            
            # Create state tuples with Dijkstra orientation for use with State methods
            dijkstra_curr_state = (curr_x, curr_y, dijkstra_orient)
            dijkstra_next_state = (next_x, next_y, dijkstra_next_orient)
            
            # Determine if the move is a rotation
            if (curr_x, curr_y) == (next_x, next_y):
                # Rotation action - determine direction
                if (dijkstra_orient + 1) % 4 == dijkstra_next_orient:
                    next_action = 1  # Rotate right
                else:
                    next_action = 0  # Rotate left
            else:
                # Position change - use State object to identify move type
                move_type = state_obj.identify_move_type(dijkstra_curr_state, dijkstra_next_state, dijkstra_orient)
                
                if move_type == "forward":
                    next_action = 2  # Forward
                elif move_type == "diagonal-left":
                    next_action = 3  # Diagonal left
                elif move_type == "diagonal-right":
                    next_action = 4  # Diagonal right
                else:
                    # Default to forward for unknown move types
                    next_action = 2
            
            next_target_state = next_state_str
            next_cell_type = logger.env_tensor[next_y, next_x]
        
        # Count lava steps in the path
        lava_steps = 0
        for i in range(1, len(path_taken)):
            step_x, step_y, _ = map(int, path_taken[i].split(","))
            if logger.env_tensor[step_y, step_x] == "lava":
                lava_steps += 1
        
        # Check if risky diagonal
        risky_diagonal = False
        
        # Only process diagonal moves (next_action 3 or 4)
        if next_action in [3, 4] and path_length > 0:  # Diagonal left or right
            # Get current state from path
            curr_x, curr_y, curr_orient = map(int, path_taken[0].split(","))
            
            # Create a fresh State object
            state_obj = State((curr_x, curr_y, curr_orient))
            
            # Convert to Dijkstra orientation for use with State methods
            agent_to_dijkstra = {0: 1, 1: 2, 2: 3, 3: 0}
            dijkstra_orient = agent_to_dijkstra[curr_orient]
            dijkstra_curr_state = (curr_x, curr_y, dijkstra_orient)
            
            # Get the next state
            next_x, next_y, next_orient = map(int, path_taken[1].split(","))
            dijkstra_next_state = (next_x, next_y, agent_to_dijkstra[next_orient])
            
            # Determine move type
            move_type = "diagonal-left" if next_action == 3 else "diagonal-right"
            
            # Debug info
            print(f"\nChecking state {state_key} for risky diagonal...")
            print(f"  Action: {next_action} ({move_type})")
            print(f"  Dijkstra orientation: {dijkstra_orient}")
            
            # IMPORTANT: Directly use the is_diagonal_safe method
            # If safe returns True, risky diagonal is False
            # If NOT safe (returns False), risky diagonal is True
            safe = state_obj.is_diagonal_safe(
                dijkstra_curr_state, 
                dijkstra_next_state, 
                dijkstra_orient, 
                move_type, 
                logger.env_tensor,
                debug=True  # Enable debug mode to see which cells are being checked
            )
            
            risky_diagonal = not safe
            if risky_diagonal:
                risky_diagonals_found += 1
                print(f"  RISKY DIAGONAL DETECTED for state {state_key}")
        
        # Format the state data with proper values
        formatted_state_data = {
            "path_taken": path_taken,
            "next_step": {
                "action": next_action,
                "target_state": next_target_state,
                "type": next_cell_type,
                "risky_diagonal": risky_diagonal
            },
            "summary_stats": {
                "path_cost": path_length,  # Use path_length as a proxy for path_cost (assuming cost 1 per step)
                "path_length": path_length,
                "lava_steps": lava_steps,
                "reachable": path_length > 0 or current_cell_type == "goal"  # Reachable if there's a path or we're already at the goal
            }
        }
        
        # Extract model_inputs directly from the steps
        if "steps" in state_data and state_data["steps"] and "model_inputs" in state_data["steps"][0]:
            # Use the raw model_inputs directly
            formatted_state_data["model_inputs"] = state_data["steps"][0]["model_inputs"]
        
        # Add the formatted state data to the main dictionary
        formatted_data[state_key] = formatted_state_data
    
    print(f"\nTotal risky diagonals found: {risky_diagonals_found}")
    
    # Update the logger's all_states_data with the formatted data
    logger.all_states_data = formatted_data
    
    # Export the data to JSON with the agent directory
    return logger.export_to_json(agent_dir=agent_dir)