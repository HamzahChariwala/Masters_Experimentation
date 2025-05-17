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
        
        # All possible orientations
        orientations = [0, 1, 2, 3]  # Right, Down, Left, Up
        
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
            elif hasattr(env, 'observation'):
                # Try to get updated observation through the wrapper chain
                obs, _, _, _, _ = env.step(0)  # Take a no-op action to update observation
                obs, info = env.reset(seed=self.seed)  # Reset to restore environment
                current_env.agent_pos = np.array([x, y])  # Position might have been reset, set it again
                current_env.agent_dir = orientation
        else:
            # If we couldn't find base environment attributes, skip this state
            print(f"  Could not set agent to state {state_key}, skipping...")
            return
        
        # Initialize episode data
        steps = []
        path_taken = [state_key]
        
        # Run until done or max steps reached
        done = False
        max_steps = 30  # Limit episode length to avoid infinite loops
        step_count = 0
        
        # First step - log initial state
        curr_state = (x, y, orientation)
        curr_cell_type = self.env_tensor[y, x]
        
        # For the first step, preview the action but don't take it
        action, _states = self.agent.predict(obs, deterministic=True)
        
        # Get the next state that would result from this action
        next_state, next_state_type, is_diagonal = self._determine_next_state(curr_state, action)
        
        # Check if the next state involves a risky diagonal move
        risky_diagonal = False
        if is_diagonal:
            # Get the adjacent cells for the diagonal move
            move_type = "diagonal-left" if action == 3 else "diagonal-right"
            adjacent_cells = self.state_object.get_adjacent_cells_for_diagonal(curr_state, move_type)
            
            # Check if any adjacent cells are lava
            for adj_x, adj_y in adjacent_cells:
                if (0 <= adj_y < self.env_tensor.shape[0] and 
                    0 <= adj_x < self.env_tensor.shape[1] and 
                    self.env_tensor[adj_y, adj_x] == "lava"):
                    risky_diagonal = True
                    break
        
        # Extract model inputs from observation directly
        model_inputs = self._extract_model_inputs(obs, info)
        
        # Create first step data
        first_step = {
            "state": state_key,
            "action": int(action),
            "reward": 0,
            "terminated": False,
            "truncated": False,
            "cell_type": curr_cell_type,
            "next_step": {
                "action": int(action),
                "target_state": ','.join(map(str, next_state)) if next_state else None,
                "type": next_state_type,
                "risky_diagonal": risky_diagonal
            },
            "model_inputs": model_inputs
        }
        
        steps.append(first_step)
        
        # Now actually take steps in the environment
        while not done and step_count < max_steps:
            # Take action
            action, _states = self.agent.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Get the agent's current position and orientation
            env_state = self._get_agent_state(env)
            if env_state is None:
                # If we can't determine agent state, use best guess from last state and action
                if next_state:
                    env_state = next_state
                else:
                    # We're stuck, just use the current state
                    env_state = curr_state
            
            # Convert state to string representation
            new_state_key = ','.join(map(str, env_state))
            
            # Add to path if state changed
            if new_state_key != path_taken[-1]:
                path_taken.append(new_state_key)
            
            # Get cell type
            cell_x, cell_y, _ = env_state
            cell_type = self.env_tensor[cell_y, cell_x]
            
            # Determine what the next state would be
            next_state, next_state_type, is_diagonal = self._determine_next_state(env_state, action)
            
            # Check for risky diagonal
            risky_diagonal = False
            if is_diagonal:
                move_type = "diagonal-left" if action == 3 else "diagonal-right"
                adjacent_cells = self.state_object.get_adjacent_cells_for_diagonal(env_state, move_type)
                
                for adj_x, adj_y in adjacent_cells:
                    if (0 <= adj_y < self.env_tensor.shape[0] and 
                        0 <= adj_x < self.env_tensor.shape[1] and 
                        self.env_tensor[adj_y, adj_x] == "lava"):
                        risky_diagonal = True
                        break
            
            # Extract model inputs
            model_inputs = self._extract_model_inputs(next_obs, info)
            
            # Create step data
            step_data = {
                "state": new_state_key,
                "action": int(action),
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "cell_type": cell_type,
                "next_step": {
                    "action": int(action),
                    "target_state": ','.join(map(str, next_state)) if next_state else None,
                    "type": next_state_type,
                    "risky_diagonal": risky_diagonal
                },
                "model_inputs": model_inputs
            }
            
            steps.append(step_data)
            
            # Update for next iteration
            obs = next_obs
            curr_state = env_state
            done = terminated or truncated
            step_count += 1
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(steps, path_taken)
        
        # Store agent behavior for this state in the format matching Dijkstra logs
        self.all_states_data[state_key] = {
            "standard": {
                "path_taken": path_taken,
                "next_step": first_step["next_step"],
                "summary_stats": summary_stats,
                "steps": steps
            }
        }
    
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
        Extract model inputs from observation and info.
        
        Args:
            obs (Dict[str, Any]): Observation from the environment
            info (Dict[str, Any]): Additional info from the environment
            
        Returns:
            Dict[str, Any]: Model inputs dictionary with actual values
        """
        model_inputs = {}
        
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
        
        # Directly include the entire observation dictionary with actual values
        if isinstance(obs, dict):
            for key, value in obs.items():
                model_inputs[key] = convert_numpy_to_list(value)
        elif isinstance(obs, np.ndarray):
            model_inputs["observation"] = convert_numpy_to_list(obs)
        
        # Try to extract MLP_Keys from info if available
        if "MLP_Keys" in info:
            model_inputs["MLP_Keys"] = convert_numpy_to_list(info["MLP_Keys"])
        
        # Try to extract flattened vector if available
        if "flattened_vector" in info:
            model_inputs["flattened_vector"] = convert_numpy_to_list(info["flattened_vector"])
            
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
        output_data = {
            "environment": {
                "layout": convert_numpy_to_list(self.env_tensor),
                "legend": {
                    "wall": "Wall cell - impassable",
                    "floor": "Floor cell - normal traversal",
                    "lava": "Lava cell - avoided or penalized",
                    "goal": "Goal cell - destination"
                }
            },
            "states": convert_numpy_to_list(self.all_states_data)
        }
        
        # Determine output path if not provided
        if output_path is None:
            # Create Agent_Evaluation/AgentLogs directory if it doesn't exist
            agent_logs_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "AgentLogs"
            )
            os.makedirs(agent_logs_dir, exist_ok=True)
            
            # Save to Agent_Evaluation/AgentLogs directory
            output_path = os.path.join(
                agent_logs_dir, 
                f"{self.env_id}-{self.seed}.json"
            )
        
        # Save the data to a JSON file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Agent behavior data exported to: {output_path}")
        return output_path


def run_agent_evaluation(agent, env, env_id: str, seed: int, num_episodes: int = 1) -> str:
    """
    Run the agent evaluation and log its behavior.
    
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
    
    # Log agent behavior from all valid starting positions
    logger.log_agent_behavior(env)
    
    # Export the data to JSON
    return logger.export_to_json() 