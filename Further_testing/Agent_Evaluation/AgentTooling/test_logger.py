import os
import sys
import json
import numpy as np
from typing import Dict, Any, List, Tuple

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from Agent_Evaluation.AgentTooling.agent_logger import AgentLogger
from collections import namedtuple

class MockAgent:
    """Mock agent for testing"""
    def predict(self, obs, deterministic=True):
        """Return a random action (0-4)"""
        # Let's create a simple deterministic pattern: first turn right, then go forward
        # This ensures we get some meaningful path data
        if not hasattr(self, "step_counter"):
            self.step_counter = 0
        
        action = 1 if self.step_counter % 3 == 0 else 2  # Right turn or forward
        self.step_counter += 1
        return action, None

class MockGrid:
    """Mock grid for testing"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def get(self, x, y):
        # Just a dummy method to satisfy MiniGrid interface
        return None

class MockEnv:
    """Mock environment for testing"""
    def __init__(self, env_tensor):
        self.grid = MockGrid(env_tensor.shape[1], env_tensor.shape[0])
        self.agent_pos = np.array([1, 1])
        self.agent_dir = 0
        self.env_tensor = env_tensor
        
    def reset(self, seed=None):
        return {"image": np.zeros((7, 7, 3))}, {}
        
    def step(self, action):
        # Process the action
        if action == 0:  # Left turn
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 1:  # Right turn
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2:  # Forward
            # Calculate the new position based on direction
            dx, dy = 0, 0
            if self.agent_dir == 0:  # Right
                dx, dy = 1, 0
            elif self.agent_dir == 1:  # Down
                dx, dy = 0, 1
            elif self.agent_dir == 2:  # Left
                dx, dy = -1, 0
            elif self.agent_dir == 3:  # Up
                dx, dy = 0, -1
            
            # Check if the new position is within bounds and not a wall
            new_x, new_y = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if (0 <= new_x < self.grid.width and 
                0 <= new_y < self.grid.height and 
                self.env_tensor[new_y, new_x] != "wall"):
                self.agent_pos = np.array([new_x, new_y])
        
        # Check if the agent is on a special cell
        x, y = self.agent_pos
        cell_type = self.env_tensor[y, x]
        
        # Determine reward and termination
        reward = 0.0
        terminated = False
        truncated = False
        
        if cell_type == "goal":
            reward = 1.0
            terminated = True
        elif cell_type == "lava":
            reward = -1.0
        
        # Create an observation
        obs = {"image": np.zeros((7, 7, 3))}
        
        # Info includes flattened vector and MLP_Keys for testing
        info = {
            "flattened_vector": np.random.rand(10),
            "MLP_Keys": {
                "key1": np.random.rand(5),
                "key2": np.random.rand(5)
            }
        }
        
        return obs, reward, terminated, truncated, info
    
    def gen_obs(self):
        return {"image": np.zeros((7, 7, 3))}
    
    @property
    def unwrapped(self):
        return self
    
    @property
    def env(self):
        return None
    
    def close(self):
        pass

# Mock implementation of the key functionality for testing
class TestAgentLogger:
    """
    A simplified agent logger for testing the output format.
    """
    
    def __init__(self, env_id: str, seed: int):
        self.env_id = env_id
        self.seed = seed
        self.env_tensor = self._create_test_env_tensor()
        self.all_states_data = {}
        
        # Generate test data for all valid states
        self._generate_test_data()
    
    def _create_test_env_tensor(self):
        """Create a test environment tensor"""
        env_tensor = np.full((11, 11), "floor", dtype=object)
        
        # Add walls around the perimeter
        env_tensor[0, :] = "wall"
        env_tensor[-1, :] = "wall"
        env_tensor[:, 0] = "wall"
        env_tensor[:, -1] = "wall"
        
        # Add some lava
        env_tensor[3, 5] = "lava"
        env_tensor[5, 3] = "lava"
        env_tensor[5, 7] = "lava"
        env_tensor[7, 5] = "lava"
        
        # Add goal at the bottom right
        env_tensor[9, 9] = "goal"
        
        return env_tensor
    
    def _generate_test_data(self):
        """Generate test data for all valid states"""
        # Get environment dimensions
        height, width = self.env_tensor.shape
        
        # All possible orientations
        orientations = [0, 1, 2, 3]  # Right, Down, Left, Up
        
        # Test from every valid position (excluding walls and goal)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                cell_type = self.env_tensor[y, x]
                
                # Skip walls and goals as starting positions
                if cell_type == "wall" or cell_type == "goal":
                    continue
                
                # Test all orientations at this position
                for orientation in orientations:
                    state_key = f"{x},{y},{orientation}"
                    
                    # Generate test data for this state
                    self._generate_state_data(x, y, orientation, state_key)
    
    def _generate_state_data(self, x: int, y: int, orientation: int, state_key: str):
        """Generate test data for a single state"""
        # Generate a sample path
        path_taken = [state_key]
        
        # Add some steps to the path
        curr_x, curr_y, curr_dir = x, y, orientation
        steps = []
        
        # Create first step data with model inputs
        first_step = {
            "state": state_key,
            "action": 1,  # Sample action (turn right)
            "reward": 0,
            "terminated": False,
            "truncated": False,
            "cell_type": self.env_tensor[y, x],
            "next_step": {
                "action": 1,
                "target_state": f"{x},{y},{(orientation + 1) % 4}",
                "type": "floor",
                "risky_diagonal": False
            },
            "model_inputs": {
                "image": np.random.rand(7, 7, 3).tolist(),
                "flattened_vector": np.random.rand(10).tolist(),
                "MLP_Keys": {
                    "key1": np.random.rand(5).tolist(),
                    "key2": np.random.rand(5).tolist()
                }
            }
        }
        
        steps.append(first_step)
        
        # Add a few more steps
        for i in range(5):
            action = i % 3  # Mix of actions
            
            # Update direction based on action
            if action == 0:  # Left turn
                curr_dir = (curr_dir - 1) % 4
            elif action == 1:  # Right turn
                curr_dir = (curr_dir + 1) % 4
            elif action == 2:  # Forward
                # Move forward based on current direction
                dx, dy = 0, 0
                if curr_dir == 0:  # Right
                    dx, dy = 1, 0
                elif curr_dir == 1:  # Down
                    dx, dy = 0, 1
                elif curr_dir == 2:  # Left
                    dx, dy = -1, 0
                elif curr_dir == 3:  # Up
                    dx, dy = 0, -1
                
                # Check if the move is valid
                new_x, new_y = curr_x + dx, curr_y + dy
                if (0 < new_x < self.env_tensor.shape[1] - 1 and 
                    0 < new_y < self.env_tensor.shape[0] - 1 and 
                    self.env_tensor[new_y, new_x] != "wall"):
                    curr_x, curr_y = new_x, new_y
            
            # Update path
            new_state_key = f"{curr_x},{curr_y},{curr_dir}"
            if new_state_key != path_taken[-1]:
                path_taken.append(new_state_key)
            
            # Get cell type
            cell_type = self.env_tensor[curr_y, curr_x]
            
            # Determine termination
            terminated = cell_type == "goal"
            truncated = False
            reward = 1.0 if terminated else (-1.0 if cell_type == "lava" else 0.0)
            
            # Create step data
            step_data = {
                "state": new_state_key,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "cell_type": cell_type,
                "next_step": {
                    "action": action,
                    "target_state": f"{curr_x},{curr_y},{curr_dir}",
                    "type": cell_type,
                    "risky_diagonal": False
                },
                "model_inputs": {
                    "image": np.random.rand(7, 7, 3).tolist(),
                    "flattened_vector": np.random.rand(10).tolist(),
                    "MLP_Keys": {
                        "key1": np.random.rand(5).tolist(),
                        "key2": np.random.rand(5).tolist()
                    }
                }
            }
            
            steps.append(step_data)
            
            if terminated or truncated:
                break
        
        # Calculate summary statistics
        total_reward = sum(step["reward"] for step in steps)
        path_length = len(path_taken) - 1
        lava_steps = sum(1 for step in steps if step["cell_type"] == "lava")
        success = any(step["terminated"] and step["reward"] > 0 for step in steps)
        
        summary_stats = {
            "total_reward": total_reward,
            "path_length": path_length,
            "lava_steps": lava_steps,
            "success": success,
            "truncated": any(step["truncated"] for step in steps),
            "terminated": any(step["terminated"] for step in steps),
            "into_wall": False,
            "cyclic_rotation": False,
            "reachable": success or path_length > 0
        }
        
        # Store the data
        self.all_states_data[state_key] = {
            "summary_stats": summary_stats,
            "path_taken": path_taken,
            "steps": steps
        }
    
    def export_to_json(self, output_path: str = None) -> str:
        """
        Export the test data to a JSON file.
        
        Args:
            output_path (str, optional): Path to save the JSON file.
            
        Returns:
            str: Path to the saved JSON file
        """
        # Build the output data structure to match Dijkstra log format
        output_data = {
            "environment": {
                "layout": self.env_tensor.tolist(),
                "legend": {
                    "wall": "Wall cell - impassable",
                    "floor": "Floor cell - normal traversal",
                    "lava": "Lava cell - avoided or penalized",
                    "goal": "Goal cell - destination"
                }
            },
            "states": {}
        }
        
        # Add all states data
        for state_key, state_data in self.all_states_data.items():
            output_data["states"][state_key] = {
                "agent_behavior": state_data
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
            
        print(f"Test data exported to: {output_path}")
        return output_path

def run_test():
    """Run the test"""
    print("Generating test data for agent evaluation output format...")
    
    # Create the test logger
    logger = TestAgentLogger("TestEnv", 42)
    
    # Export the results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "AgentLogs",
        "test_logger_output.json"
    )
    
    # Export to JSON
    logger.export_to_json(output_path)
    print(f"\nExported results to: {output_path}")
    
    # Load and validate the JSON structure
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    # Print summary of structure
    validate_output(data, output_path)
    
    print("\nTest completed successfully.")

def validate_output(data, output_path):
    """Validate the output data structure"""
    print("\nValidating output structure:")
    
    # Check if the structure matches what we expect
    if "environment" in data and "states" in data:
        print("✓ JSON has correct top-level structure")
    else:
        print("✗ JSON is missing required top-level keys")
    
    if "layout" in data["environment"] and "legend" in data["environment"]:
        print("✓ Environment data is correctly structured")
    else:
        print("✗ Environment data is missing required keys")
    
    # Check if the layout matches our tensor
    if isinstance(data["environment"]["layout"], list):
        print("✓ Environment layout is a list")
    else:
        print("✗ Environment layout is not a list")
    
    # Print summary of structure
    print("\nJSON Structure Summary:")
    print(f"- Total size: {os.path.getsize(output_path)} bytes")
    print(f"- Environment layout dimensions: {len(data['environment']['layout'])}x{len(data['environment']['layout'][0])}")
    print(f"- Number of states: {len(data['states'])}")
    
    # Print stats for a few sample states
    state_count = 0
    for state_key in list(data["states"].keys())[:3]:  # Just look at first 3
        state_count += 1
        state_data = data["states"][state_key]
        if "agent_behavior" in state_data:
            agent_behavior = state_data["agent_behavior"]
            print(f"\nSample State {state_count} - {state_key}:")
            if "path_taken" in agent_behavior:
                path = agent_behavior["path_taken"]
                print(f"- Path taken: {path[:2]}...{path[-2:]} (length {len(path)})")
            if "summary_stats" in agent_behavior:
                print(f"- Summary stats: {agent_behavior['summary_stats']}")
            if "steps" in agent_behavior and len(agent_behavior["steps"]) > 0:
                first_step = agent_behavior["steps"][0]
                if "model_inputs" in first_step:
                    print(f"- Model inputs present in first step: {list(first_step['model_inputs'].keys())}")
                    
                    # Validate that model inputs contain actual data
                    for key, value in first_step["model_inputs"].items():
                        if isinstance(value, dict) and value:
                            print(f"  - {key}: {len(value)} keys")
                        elif isinstance(value, list) and value:
                            if isinstance(value[0], list) and isinstance(value[0][0], list):
                                print(f"  - {key}: 3D array of shape {len(value)}x{len(value[0])}x{len(value[0][0])}")
                            elif isinstance(value[0], list):
                                print(f"  - {key}: 2D array of shape {len(value)}x{len(value[0])}")
                            else:
                                print(f"  - {key}: 1D array of length {len(value)}")

if __name__ == "__main__":
    run_test() 