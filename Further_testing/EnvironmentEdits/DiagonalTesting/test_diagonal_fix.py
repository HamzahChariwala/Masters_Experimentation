#!/usr/bin/env python3
"""
Test script to verify that the fixed diagonal movement implementation works correctly.
This script tests both basic movement and wall interaction cases.
"""

import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

# Add the root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the environment generation functions
from EnvironmentEdits.EnvironmentGeneration import make_env
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper
from minigrid.core.world_object import Wall

# Set seeds for reproducibility
SEED = 12345

# Direction names for better readability
DIRECTION_NAMES = {
    0: "right",
    1: "down",
    2: "left",
    3: "up"
}

# Action names
ACTION_NAMES = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "diagonal left",
    4: "diagonal right"
}

def ensure_array(value):
    """Convert a tuple or other iterable to a numpy array"""
    if isinstance(value, np.ndarray):
        return value
    else:
        return np.array(value)

def to_list(value):
    """Convert numpy array to list safely"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, tuple):
        return list(value)
    else:
        return value

class DiagonalTester:
    """Class to test diagonal movement implementation"""
    
    def __init__(self, env_id='MiniGrid-Empty-8x8-v0', max_episode_steps=150):
        """Initialize the tester with the specified environment"""
        self.env_id = env_id
        
        # Create environment with custom parameters
        self.env = make_env(
            env_id=env_id,
            rank=0,
            env_seed=SEED,
            window_size=7,
            cnn_keys=[],
            mlp_keys=["four_way_goal_direction",
                      "four_way_angle_alignment",
                      "barrier_mask",
                      "lava_mask"],
            max_episode_steps=max_episode_steps,
            use_random_spawn=False,
            use_no_death=True,
            no_death_types=("lava",),
            death_cost=0,
            monitor_diagonal_moves=True,
            diagonal_success_reward=1.5,
            diagonal_failure_penalty=0.1
        )
        
        # Find the CustomActionWrapper in the environment stack
        self.action_wrapper = self._find_wrapper(CustomActionWrapper)
        
        if not self.action_wrapper:
            raise ValueError("Could not find CustomActionWrapper in the environment stack")
        
        # Access the base environment
        self.base_env = self._get_base_env()
        
        # Keep track of wall positions
        self.wall_positions = []
    
    def _find_wrapper(self, wrapper_class):
        """Find a specific wrapper in the environment stack"""
        env = self.env
        while env is not None:
            if isinstance(env, wrapper_class):
                return env
            if hasattr(env, 'env'):
                env = env.env
            else:
                break
        return None
    
    def _get_base_env(self):
        """Get the base environment instance"""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env
    
    def _print_grid(self, highlight_positions=None):
        """Print the current grid layout (simplified)"""
        print("\nGrid Layout:")
        grid = self.base_env.grid
        
        width, height = grid.width, grid.height
        output = []
        
        # Print coordinates header
        coord_header = "   "
        for i in range(width):
            coord_header += str(i)
        print(coord_header)
        
        for j in range(height):
            row = []
            # Add row coordinate
            row_str = f"{j}: "
            
            for i in range(width):
                cell = grid.get(i, j)
                
                # Default character
                ch = "."  # Empty
                
                if cell is not None:
                    if cell.type == 'wall':
                        ch = "#"  # Wall
                    elif cell.type == 'lava':
                        ch = "L"  # Lava
                    elif cell.type == 'goal':
                        ch = "G"  # Goal
                    elif cell.type == 'door':
                        ch = "D"  # Door
                    else:
                        ch = cell.type[0]  # First letter of type
                
                # Mark agent position
                agent_pos = ensure_array(self.base_env.agent_pos)
                if np.array_equal(agent_pos, np.array([i, j])):
                    ch = "A"
                
                # Mark highlighted positions if any
                if highlight_positions:
                    for pos, marker in highlight_positions:
                        if np.array_equal(np.array([i, j]), np.array(pos)):
                            ch = marker
                            
                row.append(ch)
            
            output.append(row_str + "".join(row))
        
        print("\n".join(output))
        print(f"Agent position: {self.base_env.agent_pos}, direction: {self.base_env.agent_dir} ({DIRECTION_NAMES[self.base_env.agent_dir]})")
    
    def set_agent_position_and_direction(self, position, direction):
        """Set agent position and direction, ensuring it's updated in all environment layers"""
        position_array = ensure_array(position)
        
        # Set in base environment
        self.base_env.agent_pos = position_array.copy()
        self.base_env.agent_dir = direction
        
        # Set in all wrapper layers
        env = self.env
        while env is not None and hasattr(env, 'env'):
            if hasattr(env, 'agent_pos'):
                if isinstance(env.agent_pos, np.ndarray):
                    env.agent_pos = position_array.copy()
                elif hasattr(env, 'agent_pos'):
                    env.agent_pos = tuple(position_array)
            if hasattr(env, 'agent_dir'):
                env.agent_dir = direction
            env = env.env
    
    def place_wall(self, x, y):
        """Place a wall at the specified position"""
        self.base_env.grid.set(x, y, Wall())
        self.wall_positions.append([x, y])
    
    def calculate_diagonal_move_positions(self, agent_pos, agent_dir, diag_direction):
        """Calculate the expected positions for a diagonal move"""
        # Define direction vectors
        direction_vecs = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1])   # up
        }
        
        forward_vec = direction_vecs[agent_dir]
        left_vec = np.array([-forward_vec[1], forward_vec[0]])
        right_vec = np.array([forward_vec[1], -forward_vec[0]])
        
        if diag_direction == "left":
            diag_vec = forward_vec + left_vec
            lateral_vec = left_vec
        else:  # right
            diag_vec = forward_vec + right_vec
            lateral_vec = right_vec
        
        # Calculate positions
        diagonal_pos = agent_pos + diag_vec
        forward_pos = agent_pos + forward_vec
        lateral_pos = agent_pos + lateral_vec
        
        return {
            "forward_pos": forward_pos,
            "lateral_pos": lateral_pos,
            "diagonal_pos": diagonal_pos
        }
    
    def perform_diagonal_action(self, diag_direction):
        """Perform a diagonal action and return the result"""
        action = 3 if diag_direction == "left" else 4
        original_pos = self.base_env.agent_pos.copy()
        
        # Get current direction
        agent_dir = self.base_env.agent_dir
        
        # Calculate expected positions
        positions = self.calculate_diagonal_move_positions(
            original_pos, agent_dir, diag_direction
        )
        
        # Perform the action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Determine if the move was successful
        new_pos = self.base_env.agent_pos.copy()
        success = not np.array_equal(original_pos, new_pos)
        
        # Check if the position matches the expected position if successful
        position_match = False
        if success:
            position_match = np.array_equal(new_pos, positions["diagonal_pos"])
        
        result = {
            "agent_direction": agent_dir,
            "agent_direction_name": DIRECTION_NAMES[agent_dir],
            "diagonal_direction": diag_direction,
            "original_position": to_list(original_pos),
            "expected_position": to_list(positions["diagonal_pos"]),
            "actual_position": to_list(new_pos),
            "forward_position": to_list(positions["forward_pos"]),
            "forward_walkable": self.action_wrapper._is_walkable(positions["forward_pos"]),
            "lateral_position": to_list(positions["lateral_pos"]),
            "lateral_walkable": self.action_wrapper._is_walkable(positions["lateral_pos"]),
            "diagonal_position": to_list(positions["diagonal_pos"]),
            "diagonal_walkable": self.action_wrapper._is_walkable(positions["diagonal_pos"]),
            "success": success,
            "position_match": position_match,
            "reward": reward
        }
        
        return result
    
    def create_wall_configuration(self):
        """Create a specific wall configuration for testing corner cases"""
        # Reset any existing walls
        self.wall_positions = []
        
        # Reset the environment
        self.env.reset()
        
        # Place walls at strategic positions to test corner movement
        wall_positions = [
            [3, 3],  # Center wall
            [3, 4],  # Wall south of center
            [4, 3],  # Wall east of center
            [3, 6],  # Isolated wall
            [4, 1],  # Isolated wall
            [6, 3]   # Isolated wall
        ]
        
        for x, y in wall_positions:
            self.place_wall(x, y)
        
        return wall_positions
    
    def test_basic_diagonal_moves(self):
        """Test basic diagonal movement implementation without walls"""
        print("Testing basic diagonal movement...")
        self.env.reset()
        
        # Center position for testing
        center_pos = np.array([3, 3])
        
        # Test in all four directions
        results = []
        for direction in range(4):  # 0: right, 1: down, 2: left, 3: up
            self.set_agent_position_and_direction(center_pos, direction)
            
            # Test left diagonal
            left_result = self.perform_diagonal_action("left")
            
            # Reset position
            self.set_agent_position_and_direction(center_pos, direction)
            
            # Test right diagonal
            right_result = self.perform_diagonal_action("right")
            
            result = {
                "agent_direction": direction,
                "agent_direction_name": DIRECTION_NAMES[direction],
                "original_position": to_list(center_pos),
                "left_diagonal": left_result,
                "right_diagonal": right_result
            }
            
            results.append(result)
        
        # Calculate success rate
        total_tests = len(results) * 2  # Left and right for each direction
        successful_tests = sum(r["left_diagonal"]["success"] and r["left_diagonal"]["position_match"] for r in results)
        successful_tests += sum(r["right_diagonal"]["success"] and r["right_diagonal"]["position_match"] for r in results)
        
        success_rate = (successful_tests / total_tests) * 100
        
        print(f"Basic movement test complete: {successful_tests}/{total_tests} tests passed ({success_rate:.2f}%)")
        
        return results, success_rate
    
    def test_diagonal_wall_blocking(self):
        """Test if diagonal moves are properly blocked by walls"""
        print("Testing diagonal movement around walls...")
        # Create wall configuration
        wall_positions = self.create_wall_configuration()
        
        # Test cases for diagonal movement around walls
        test_cases = [
            {
                "description": "Moving diagonally into a corner from top left (blocked by walls)",
                "position": [2, 2],
                "direction": 0,  # right
                "diagonal_direction": "right",
                "expected_success": False
            },
            {
                "description": "Moving diagonally into a corner from bottom right (blocked by walls)",
                "position": [5, 5],
                "direction": 2,  # left
                "diagonal_direction": "left",
                "expected_success": False
            },
            {
                "description": "Diagonal move blocked by wall in forward direction",
                "position": [2, 3],
                "direction": 0,  # right
                "diagonal_direction": "right",
                "expected_success": False
            },
            {
                "description": "Diagonal move blocked by wall in lateral direction",
                "position": [3, 2],
                "direction": 1,  # down
                "diagonal_direction": "left",
                "expected_success": False
            },
            {
                "description": "Diagonal path is clear despite adjacent walls",
                "position": [2, 4],
                "direction": 0,  # right
                "diagonal_direction": "left",
                "expected_success": True
            },
            {
                "description": "Diagonal path is clear despite adjacent walls (opposite side)",
                "position": [4, 2],
                "direction": 1,  # down
                "diagonal_direction": "right",
                "expected_success": True
            },
            {
                "description": "Diagonal move into an isolated wall",
                "position": [2, 6],
                "direction": 0,  # right
                "diagonal_direction": "left",
                "expected_success": False
            },
            {
                "description": "Diagonal move around an isolated wall",
                "position": [5, 3],
                "direction": 2,  # left
                "diagonal_direction": "right",
                "expected_success": True
            },
            {
                "description": "Diagonal move with both adjacent walls (should fail)",
                "position": [2, 2],
                "direction": 1,  # down
                "diagonal_direction": "right",
                "expected_success": False
            },
            {
                "description": "Clear diagonal path",
                "position": [5, 5],
                "direction": 0,  # right
                "diagonal_direction": "right",
                "expected_success": True
            }
        ]
        
        # Run the test cases
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"\nTest case {i+1}: {test_case['description']}")
            
            # Set the agent's position and direction
            self.set_agent_position_and_direction(
                test_case["position"], 
                test_case["direction"]
            )
            
            # Visualize the current setup
            self._print_grid()
            
            # Perform the diagonal action
            result = self.perform_diagonal_action(test_case["diagonal_direction"])
            
            # Add expected success to the result
            result["test_case"] = i + 1
            result["description"] = test_case["description"]
            result["expected_success"] = test_case["expected_success"]
            result["success_match"] = result["success"] == test_case["expected_success"]
            
            results.append(result)
            
            # Display result
            print(f"Expected success: {test_case['expected_success']}, Actual success: {result['success']}")
            print(f"Result matches expectation: {result['success_match']}")
            
            # Show the updated grid if the action was successful
            if result["success"]:
                print("\nAfter move:")
                self._print_grid()
        
        # Calculate success rate
        successful_tests = sum(r["success_match"] for r in results)
        total_tests = len(results)
        success_rate = (successful_tests / total_tests) * 100
        
        print(f"\nWall blocking test complete: {successful_tests}/{total_tests} tests passed ({success_rate:.2f}%)")
        
        return results, wall_positions, success_rate
    
    def visualize_test_grid(self, results, wall_positions):
        """Create a visualization of the test grid with test case results"""
        grid_size = 8  # 8x8 grid
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a grid
        for i in range(grid_size + 1):
            ax.axhline(i, color='black', linewidth=0.5)
            ax.axvline(i, color='black', linewidth=0.5)
        
        # Mark walls
        for wall_pos in wall_positions:
            ax.add_patch(Rectangle((wall_pos[0], grid_size - 1 - wall_pos[1]), 1, 1, 
                                   facecolor='black', alpha=0.8))
        
        # Mark test cases
        for result in results:
            # Extract positions
            start_pos = result["original_position"]
            target_pos = result["diagonal_position"]
            
            # Determine color based on if the test passed
            color = 'green' if result["success_match"] else 'red'
            
            # Plot position with a number on it
            ax.text(start_pos[0] + 0.5, grid_size - 1 - start_pos[1] + 0.5, 
                    str(result["test_case"]), 
                    ha='center', va='center', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))
            
            # Draw an arrow to show the expected movement
            ax.arrow(start_pos[0] + 0.5, grid_size - 1 - start_pos[1] + 0.5,
                     (target_pos[0] - start_pos[0]) * 0.8, 
                     (start_pos[1] - target_pos[1]) * 0.8,  # Invert Y for plotting
                     head_width=0.2, head_length=0.2, fc=color, ec=color)
        
        # Set labels and title
        ax.set_xticks(np.arange(0.5, grid_size + 0.5))
        ax.set_xticklabels(range(grid_size))
        ax.set_yticks(np.arange(0.5, grid_size + 0.5))
        ax.set_yticklabels(range(grid_size - 1, -1, -1))  # Invert Y labels
        
        ax.set_title("Diagonal Movement Test Results")
        
        # Create a custom legend
        import matplotlib.patches as mpatches
        wall_patch = mpatches.Patch(color='black', label='Wall')
        success_patch = mpatches.Patch(color='green', label='Test Passed')
        fail_patch = mpatches.Patch(color='red', label='Test Failed')
        
        ax.legend(handles=[wall_patch, success_patch, fail_patch], loc='upper right')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig("diagonal_test_results.png")
        plt.close()

def run_tests():
    """Run all tests and save results"""
    tester = DiagonalTester()
    
    # Test basic movement
    basic_results, basic_success_rate = tester.test_basic_diagonal_moves()
    
    # Test wall blocking
    wall_results, wall_positions, wall_success_rate = tester.test_diagonal_wall_blocking()
    
    # Create visualization
    tester.visualize_test_grid(wall_results, wall_positions)
    
    # Prepare results
    all_results = {
        "environment": tester.env_id,
        "basic_movement": {
            "results": basic_results,
            "success_rate": basic_success_rate
        },
        "wall_interaction": {
            "wall_positions": wall_positions,
            "results": wall_results,
            "success_rate": wall_success_rate
        }
    }
    
    # Save results to file
    with open("diagonal_fix_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n===== TEST SUMMARY =====")
    print(f"Basic movement test: {basic_success_rate:.2f}% success rate")
    print(f"Wall interaction test: {wall_success_rate:.2f}% success rate")
    print(f"Overall success: {(basic_success_rate + wall_success_rate) / 2:.2f}%")
    print("Results saved to diagonal_fix_results.json")
    print("Visualization saved to diagonal_test_results.png")
    
    return basic_success_rate, wall_success_rate

if __name__ == "__main__":
    basic_rate, wall_rate = run_tests()
    
    # Exit with success if both tests have high success rates
    import sys
    if basic_rate >= 95 and wall_rate >= 95:
        print("\nAll tests PASSED!")
        sys.exit(0)
    else:
        print("\nSome tests FAILED!")
        sys.exit(1) 