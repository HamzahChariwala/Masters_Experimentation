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
# Import make_env and CustomCombinedExtractor
from Environment_Tooling.EnvironmentGeneration import make_env
from Environment_Tooling.BespokeEdits.FeatureExtractor import CustomCombinedExtractor
from Agent_Evaluation.EnvironmentTooling.extract_grid import extract_grid_from_env

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
        
    def log_agent_behavior(self, eval_env_fn) -> Dict[str, Any]:
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
                
                # Test all orientations at this position on a brand-new env each time
                for orientation in orientations:
                    state_key = f"{x},{y},{orientation}"
                    print(f"Testing from state: {state_key}...")

                    try:
                        # spawn a fresh env instance so env.reset(options=…) is always honored
                        env = eval_env_fn()
                        self._run_episode_from_state(env, (x, y, orientation))
                    except Exception as e:
                        print(f"Error running episode from state {state_key}: {e}")
        
        return self.all_states_data
    
    def _run_episode_from_state(self, env, start_state: Tuple[int, int, int]):
        """
        Run a single episode with the agent starting at the specified state.
        Args:
            env: The fully wrapped environment to evaluate in
            start_state: The starting state (x, y, orientation)
        """
        x, y, orientation = start_state

        # Safety: make sure env is not a VecEnv
        if hasattr(env, "envs") and len(env.envs) == 1:
            env = env.envs[0]

        state_key = f"{x},{y},{orientation}"
        
        if state_key in self.all_states_data:
            return

        print(f"\nAttempting to set state {state_key} using env.reset(options=...)")
        reset_options = {
            'agent_pos': (x, y),
            'agent_dir': orientation
        }

        try:
            # Attempt to reset the environment to the specific start_state using options
            obs, info = env.reset(options=reset_options)
            print(f"  DEBUG: For state {state_key}, after env.reset(options={reset_options}):")
            if isinstance(obs, dict):
                print(f"    obs keys: {list(obs.keys())}")
                if 'MLP_input' in obs:
                    print(f"    obs['MLP_input'] shape: {obs['MLP_input'].shape}, dtype: {obs['MLP_input'].dtype}")
                    # print(f"    obs['MLP_input'] content: {obs['MLP_input']}") # Potentially too verbose
                else:
                    print(f"    WARNING: 'MLP_input' key NOT found in obs after reset with options.")
            else:
                print(f"    WARNING: obs is not a dict after reset with options. Type: {type(obs)}")
            
            if isinstance(info, dict):
                print(f"    info keys: {list(info.keys())}")
                if 'log_data' in info and isinstance(info['log_data'], dict):
                    print(f"    info['log_data'] keys: {list(info['log_data'].keys())}")
                    # Log specific items from log_data for comparison if they exist
                    for key_to_check in ['four_way_goal_direction', 'barrier_mask']:
                        if key_to_check in info['log_data']:
                            print(f"      log_data['{key_to_check}'] (type: {type(info['log_data'][key_to_check])}): {str(info['log_data'][key_to_check])[:100]}")
                        else:
                            print(f"      log_data['{key_to_check}'] not found.")
                else:
                    print(f"    WARNING: info['log_data'] not found or not a dict.")
            else:
                print(f"    WARNING: info is not a dict. Type: {type(info)}")

        except Exception as e_reset:
            print(f"  CRITICAL ERROR during env.reset(options={reset_options}) for state {state_key}: {e_reset}")
            self.all_states_data[state_key] = {"error": f"Error in reset(options): {e_reset}", "steps": [], "path_taken": [], "summary": {}}
            return

        curr_cell_type = self.env_tensor[y, x] 
        # simply record the actual 106-d input vector the model saw
        model_inputs_for_json = obs['MLP_input'].tolist()

        # FOR PREDICTION: We now strictly expect obs['MLP_input'] to be valid from the wrapped environment
        if not (isinstance(obs, dict) and 'MLP_input' in obs and 
                isinstance(obs['MLP_input'], np.ndarray) and obs['MLP_input'].shape == (106,)):
            print(f"  CRITICAL ERROR: For state {state_key}, obs['MLP_input'] is missing or invalid after reset(options=...). Cannot proceed with prediction.")
            self.all_states_data[state_key] = {
                "error": "obs['MLP_input'] missing or invalid for first prediction", 
                "steps": [], "path_taken": [state_key],
                "summary": {"outcome": "failure_mlp_input_missing"},
                "model_inputs_at_error": model_inputs_for_json # Log what we extracted for JSON
            }
            return
        
        observation_for_predict = obs # Use the whole obs dict, which contains MLP_input

        steps = []
        path_taken = [state_key]
        first_step_log_entry = {
            "state": state_key, "action": None, "reward": 0, "terminated": False, "truncated": False,
            "cell_type": curr_cell_type, "model_inputs": model_inputs_for_json
        }
        
        action = None
        try:
            print(f"  DEBUG: Passing to model.predict for state {state_key}. Type of observation_for_predict: {type(observation_for_predict)}")
            if isinstance(observation_for_predict, dict):
                 print(f"         Keys: {list(observation_for_predict.keys())}")

            action, _ = self.agent.predict(observation_for_predict, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action)
            first_step_log_entry["action"] = action
        except Exception as e_predict:
            print(f"  ERROR predicting first action for state {state_key}: {e_predict}")
            self.all_states_data[state_key] = {
                "error": f"Prediction error: {e_predict}", 
                "steps": [first_step_log_entry],
                "path_taken": path_taken,
                "summary": {"outcome": "failure_predict_error"},
                "model_inputs_at_error": model_inputs_for_json,
                "mlp_vector_at_error": obs.get('MLP_input', np.array([])).tolist() # Log MLP_input if available
            }
            return 

        steps.append(first_step_log_entry)

        # --- Episode Stepping Loop ---
        # CRITICAL FIX: Use the original environment instead of creating a new one
        # This ensures we maintain the correct agent position and direction throughout the episode
        print(f"  DEBUG: Using original environment for stepping loop for state {state_key}")
        
        # Set variables for the stepping loop
        current_obs = obs
        current_info = info
        done = False
        step_count = 0
        max_steps = 20
        total_reward = 0

        while not done and step_count < max_steps:
            try:
                if not (isinstance(current_obs, dict) and 'MLP_input' in current_obs):
                    print(f"  CRITICAL ERROR: Loop: current_obs['MLP_input'] missing for {state_key}. Aborting step.")
                    steps.append({"error": "Loop: MLP_input missing in current_obs"})
                    break

                action_loop, _ = self.agent.predict(current_obs, deterministic=True)
                if isinstance(action_loop, np.ndarray):
                    action_loop = int(action_loop)

                next_obs, reward_loop, terminated_loop, truncated_loop, next_info = env.step(action_loop)
                
                # Get the agent's position after the step
                next_x, next_y, next_orientation = self._get_agent_position(env)
                next_state_key_str = f"{next_x},{next_y},{next_orientation}"
                next_cell_type = self.env_tensor[next_y, next_x] if 0 <= next_y < self.env_tensor.shape[0] and 0 <= next_x < self.env_tensor.shape[1] else "unknown"
                
                step_model_inputs_json = next_obs['MLP_input'].tolist()

                step_data = {
                    "state": next_state_key_str, "action": action_loop, "reward": reward_loop,
                    "terminated": terminated_loop, "truncated": truncated_loop, "cell_type": next_cell_type,
                    "model_inputs": step_model_inputs_json
                }
                steps.append(step_data)
                path_taken.append(next_state_key_str)
                
                current_obs = next_obs
                current_info = next_info
                total_reward += reward_loop
                done = terminated_loop or truncated_loop
                step_count += 1
                if terminated_loop and reward_loop > 0: break
                
            except Exception as e_loop_step:
                print(f"  ERROR during episode loop step {step_count} for state {state_key}: {e_loop_step}")
                # append a “fake” step record with all the keys your summary needs
                steps.append({
                    "state": path_taken[-1],
                    "action": None,
                    "reward": 0.0,
                    "terminated": True,
                    "truncated": False,
                    "cell_type": curr_cell_type,
                    "model_inputs": current_obs['MLP_input'].tolist(),
                    "error": str(e_loop_step),
                })
                done = True

        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(steps, path_taken)
        
        # Store all data for this state
        state_data = {
            "steps": steps,
            "path_taken": path_taken,
            "summary": summary
        }
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

        # ──────────────────────────────────────────────────────────────────────────
        # If someone accidentally passed in a VecEnv, pull out the real env
        if hasattr(env, "envs") and len(env.envs) == 1:
            current_env = env.envs[0]
        else:
            current_env = env
        # ──────────────────────────────────────────────────────────────────────────
        
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
        # Format: {"environment": {"layout": [...]}, "states": {"1,1,0": {...}, ...}}
        output_data = {
            "environment": {
                # IMPORTANT: DON'T transpose the layout here, as it's already in the correct format
                # The env_tensor is in shape (height, width) where [0,0] is the top-left corner
                # This matches the format used in Dijkstra logs
                "layout": convert_numpy_to_list(self.env_tensor)
            },
            "performance": {
                "__comment": "Each state maps to an array with the following values in order: [cell_type, path_length, lava_steps, reaches_goal, next_cell_is_lava, risky_diagonal, target_state, action_taken]"
            },
            "states": {}  # Add a states key for all state data
        }
        
        # Add all state data to the states object
        for state_key, state_data in self.all_states_data.items():
            # Ensure state_data has the correct structure for processed data
            if isinstance(state_data, dict):
                # Convert state data to the expected format
                output_data["states"][state_key] = convert_numpy_to_list(state_data)
        
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
        
        # Verify that the output structure includes states with path data
        state_count = len(output_data.get("states", {}))
        paths_with_data = sum(1 for state in output_data.get("states", {}).values() 
                            if "path_taken" in state and len(state["path_taken"]) > 0)
        print(f"Exporting data for {state_count} states, {paths_with_data} with path data")
        
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
        # Print debug info without full arrays
        print(f"    Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
        print(f"    Info keys: {list(info.keys()) if isinstance(info, dict) else 'Not a dict'}")
        
        # Create a brand new feature vector for this specific state
        feature_vector = np.zeros(106, dtype=np.float32)
        
        # Check if log_data is available in info
        if 'log_data' in info and isinstance(info['log_data'], dict):
            log_data = info['log_data']
            print(f"    log_data keys: {list(log_data.keys()) if isinstance(log_data, dict) else 'Not a dict'}")
            
            # Create a completely new copy of log_data to prevent any shared references
            import copy
            safe_log_data = copy.deepcopy(log_data)
            
            # 1. Extract four_way_goal_direction as a one-hot vector (first 4 positions)
            try:
                direction_processed = False
                if 'four_way_goal_direction' in safe_log_data and safe_log_data['four_way_goal_direction'] is not None:
                    direction_data = safe_log_data['four_way_goal_direction']
                    print(f"    _extract_mlp_inputs: four_way_goal_direction raw value: {direction_data}, type: {type(direction_data)}")
                    if isinstance(direction_data, (list, np.ndarray)) and np.array(direction_data).ndim == 1 and np.array(direction_data).size > 0:
                        # Assuming the array contains scores/probabilities for directions, take the argmax
                        direction_idx = np.argmax(direction_data)
                        if 0 <= direction_idx < 4:
                            feature_vector[direction_idx] = 1.0
                            direction_processed = True
                            print(f"    Info: Processed four_way_goal_direction (argmax): index {direction_idx}")
                        else:
                            print(f"    Warning: argmax of four_way_goal_direction array {direction_idx} is out of range 0-3.")
                    elif isinstance(direction_data, (int, np.integer)) and 0 <= int(direction_data) < 4:
                        # If it's already a valid integer index
                        feature_vector[int(direction_data)] = 1.0
                        direction_processed = True
                        print(f"    Info: Processed four_way_goal_direction (scalar): index {int(direction_data)}")
                    else:
                        print(f"    Warning: four_way_goal_direction value '{direction_data}' is not a valid integer (0-3) or a processable array.")
                else:
                    print(f"    Warning: four_way_goal_direction not found in log_data or is None.")
                if not direction_processed:
                    print(f"    Info: Setting default for four_way_goal_direction (all zeros in one-hot).")
            except Exception as e:
                print(f"    Warning: Error processing four_way_goal_direction: {e}. Using default.")
                    
            # 2. Extract four_way_angle_alignment (position 4)
            try:
                alignment_processed = False
                if 'four_way_angle_alignment' in safe_log_data and safe_log_data['four_way_angle_alignment'] is not None:
                    alignment_data = safe_log_data['four_way_angle_alignment']
                    print(f"    _extract_mlp_inputs: four_way_angle_alignment raw value: {alignment_data}, type: {type(alignment_data)}")
                    if isinstance(alignment_data, (list, np.ndarray)) and np.array(alignment_data).ndim == 1 and np.array(alignment_data).size > 0:
                        # Assuming we should take the first element if it's an array
                        value_to_use = float(np.array(alignment_data)[0])
                        feature_vector[4] = value_to_use
                        alignment_processed = True
                        print(f"    Info: Processed four_way_angle_alignment (first element of array): {value_to_use}")
                    elif isinstance(alignment_data, (int, float, np.number)):
                        # If it's already a scalar number
                        feature_vector[4] = float(alignment_data)
                        alignment_processed = True
                        print(f"    Info: Processed four_way_angle_alignment (scalar): {float(alignment_data)}")
                    else:
                        print(f"    Warning: four_way_angle_alignment value '{alignment_data}' is not a valid number or processable array.")
                else:
                    print(f"    Warning: four_way_angle_alignment not found in log_data or is None.")
                if not alignment_processed:
                    feature_vector[4] = 0.0 # Default to 0.0
                    print(f"    Info: Setting four_way_angle_alignment to default 0.0.")
            except Exception as e:
                print(f"    Warning: Error processing four_way_angle_alignment: {e}. Using default 0.0.")
                feature_vector[4] = 0.0
                    
            # 3. Extract and flatten barrier_mask (positions 5-54)
            try:
                if 'barrier_mask' in safe_log_data and safe_log_data['barrier_mask'] is not None:
                    barrier_mask_data = safe_log_data['barrier_mask']
                    print(f"    _extract_mlp_inputs: barrier_mask_data type: {type(barrier_mask_data)}, content (first 5 elements if list/array): {str(barrier_mask_data)[:100] if not isinstance(barrier_mask_data, (str, bool, int, float)) else barrier_mask_data}")

                    # Ensure it's a numpy array of numbers
                    if isinstance(barrier_mask_data, (list, tuple)):
                        barrier_mask_data = np.array(barrier_mask_data, dtype=np.float32)
                    elif not isinstance(barrier_mask_data, np.ndarray):
                         # Attempt to convert if it's some other type, or default to zeros
                        try:
                            barrier_mask_data = np.array(barrier_mask_data, dtype=np.float32)
                        except:
                            print(f"    Warning: barrier_mask could not be converted to np.ndarray. Using zeros. Original type: {type(barrier_mask_data)}")
                            barrier_mask_data = np.zeros((7,7), dtype=np.float32) # Assuming 7x7, adjust if different

                    if barrier_mask_data.ndim == 2: # Should be 2D
                        barrier_mask = barrier_mask_data.T.astype(np.float32) # Transpose and ensure float
                        flat_mask = barrier_mask.flatten()
                        
                        # Copy values - limit to 50 elements
                        copy_size = min(50, flat_mask.size)
                        for i in range(copy_size):
                            feature_vector[5 + i] = float(flat_mask[i]) # Ensure float
                    else:
                        print(f"    Warning: barrier_mask was not 2D. Shape: {barrier_mask_data.shape}. Using zeros for barrier_mask features.")

            except Exception as e:
                print(f"    Warning: Error processing barrier_mask: {e}")
                    
            # 4. Extract and flatten lava_mask (positions 55-104)
            try:
                if 'lava_mask' in safe_log_data and safe_log_data['lava_mask'] is not None:
                    lava_mask_data = safe_log_data['lava_mask']
                    print(f"    _extract_mlp_inputs: lava_mask_data type: {type(lava_mask_data)}, content (first 5 elements if list/array): {str(lava_mask_data)[:100] if not isinstance(lava_mask_data, (str, bool, int, float)) else lava_mask_data}")

                    # Ensure it's a numpy array of numbers
                    if isinstance(lava_mask_data, (list, tuple)):
                        lava_mask_data = np.array(lava_mask_data, dtype=np.float32)
                    elif not isinstance(lava_mask_data, np.ndarray):
                        try:
                            lava_mask_data = np.array(lava_mask_data, dtype=np.float32)
                        except:
                            print(f"    Warning: lava_mask could not be converted to np.ndarray. Using zeros. Original type: {type(lava_mask_data)}")
                            lava_mask_data = np.zeros((7,7), dtype=np.float32) # Assuming 7x7, adjust if different
                    
                    if lava_mask_data.ndim == 2: # Should be 2D
                        lava_mask = lava_mask_data.T.astype(np.float32) # Transpose and ensure float
                        flat_mask = lava_mask.flatten()
                        
                        # Copy values - limit to 50 elements
                        copy_size = min(50, flat_mask.size)
                        for i in range(copy_size):
                            feature_vector[55 + i] = float(flat_mask[i]) # Ensure float
                    else:
                        print(f"    Warning: lava_mask was not 2D. Shape: {lava_mask_data.shape}. Using zeros for lava_mask features.")
            except Exception as e:
                print(f"    Warning: Error processing lava_mask: {e}")
                    
            # 5. Add a feature indicating we have valid data at position 105
            feature_vector[105] = 1.0
            
            print(f"    Created feature vector with shape {feature_vector.shape}")
            return feature_vector  # Return the newly created feature vector
        
        # If we couldn't extract the needed features, still return a zero-filled 106-element vector
        print(f"    Created empty feature vector with shape {feature_vector.shape}")
        return feature_vector  # Return an empty feature vector


def run_agent_evaluation(agent, env, env_id: str, seed: int, num_episodes: int = 1, logger=None, agent_dir: str = None) -> str:
    """
    Run the agent evaluation and log its behavior.
    Uses a fully wrapped environment created by make_env to ensure dynamic log_data.
    
    Args:
        agent: The agent model
        env: This environment instance is IGNORED. A new one is created with make_env.
        env_id (str): Environment ID
        seed (int): Random seed
        num_episodes (int): Number of episodes to run (default: 1)
        logger (AgentLogger, optional): Pre-populated logger instance. If provided, skips data collection.
        agent_dir (str, optional): Path to the agent directory for saving logs.
        
    Returns:
        str: Path to the saved JSON file
    """
    print(f"INFO: run_agent_evaluation called for env_id={env_id}, seed={seed}")

    # Create a new, fully wrapped environment for this evaluation run
    # This ensures that all necessary wrappers for log_data generation are present.
    # The mlp_keys should match those used during training and expected by _extract_mlp_inputs.
    print(f"DEBUG: Creating fully wrapped environment for evaluation using make_env for {env_id} with seed {seed}")
    base_fn = make_env(
        env_id=env_id,
        rank=0,  # Typically 0 for a single environment
        env_seed=seed,
        window_size=7, # TODO: This should ideally come from a config or be more generic
        cnn_keys=[],   # Assuming no CNN keys for this agent, adjust if necessary
        mlp_keys=["four_way_goal_direction",
                  "four_way_angle_alignment",
                  "barrier_mask",
                  "lava_mask"],
        max_episode_steps=250,  # A reasonable default for evaluation episodes
        # Add other parameters for make_env if they were used during training for this agent
        # e.g., use_random_spawn, use_no_death, etc. For now, using common defaults.
    )

    # wrap *every* new env in your PositionAwareWrapper
    from Agent_Evaluation.EnvironmentTooling.position_aware_wrapper import PositionAwareWrapper
    def eval_env_fn():
        env = base_fn()
        return PositionAwareWrapper(env)

    evaluation_env = eval_env_fn()
    evaluation_env.reset(seed=seed)
    env_tensor = extract_grid_from_env(evaluation_env)

    print(f"DEBUG: Fully wrapped environment for evaluation created: {evaluation_env}")

    # ──────────────────────────────────────────────────────────────────────────
    # If make_env returned a DummyVecEnv/SubprocVecEnv, pull out the single sub-env
    if hasattr(evaluation_env, "envs") and len(evaluation_env.envs) == 1:
        print("DEBUG: Unwrapping vectorized env → using env.envs[0]")
        evaluation_env = evaluation_env.envs[0]
    # ──────────────────────────────────────────────────────────────────────────

    # If a logger is provided, use it (though it might have been initialized with a different env_tensor)
    # For consistency, we'll re-initialize the logger if states_data is not already populated.
    if logger is not None and logger.all_states_data:
        print("Using pre-populated logger with existing states_data.")
        # Ensure env_tensor in logger matches the new evaluation_env if possible,
        # but this is tricky as logger might be for a different seed/layout.
        # For now, if logger is pre-populated, we assume it's correct for its data.
        pass
    else:
        print("DEBUG: Creating new AgentLogger instance.")
        # Extract environment tensor from the new, fully wrapped evaluation environment
        env_tensor = extract_grid_from_env(evaluation_env)
        
        # Create agent logger with the new environment and its tensor
        logger = AgentLogger(env_id, seed, env_tensor, agent)
        
        # Log agent behavior using the fully wrapped environment
        # This 'evaluation_env' should provide dynamic log_data
        logger.log_agent_behavior(eval_env_fn) 
    
    # Structure the data (this part remains largely the same, but operates on new data)
    formatted_states = {}
    
    # For tracking the number of risky diagonals found
    risky_diagonals_found = 0
    
    # For each state in the agent behavior data
    for state_key, state_data in logger.all_states_data.items():
        # Get path_taken from state_data - preserve what's already there
        path_taken = state_data.get("path_taken", [])
        
        # If the path isn't there but it's in the original data structure under "steps"
        if not path_taken and "steps" in state_data:
            # Collect state keys from each step to build the path
            path_taken = [step.get("state", "") for step in state_data["steps"] if "state" in step]
        
        # Fallback to old "path" key if available
        if not path_taken and "path" in state_data:
            path_taken = state_data["path"]
            
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
            # Make a deep copy of the model_inputs to ensure each state has unique inputs
            import copy
            formatted_state_data["model_inputs"] = copy.deepcopy(state_data["steps"][0]["model_inputs"])
        
        # Add the formatted state data to the main dictionary
        formatted_states[state_key] = formatted_state_data
    
    print(f"\nTotal risky diagonals found: {risky_diagonals_found}")
    print(f"Total states processed: {len(formatted_states)}")
    
    # Update the logger's all_states_data with the formatted data
    logger.all_states_data = formatted_states
    
    # Export the data to JSON with the agent directory
    return logger.export_to_json(agent_dir=agent_dir)