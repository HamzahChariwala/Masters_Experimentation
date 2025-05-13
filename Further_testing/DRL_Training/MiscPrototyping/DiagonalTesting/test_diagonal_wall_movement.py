#!/usr/bin/env python3
"""
Test script to verify that diagonal moves are properly blocked when walls are present.
"""

import os
import sys
import numpy as np
import random
import json

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

def is_walkable(env, pos):
    """
    Determine if a position is walkable using MiniGrid's logic.
    A position is walkable if it's within bounds and either has no object or the object can be overlapped.
    """
    grid = env.grid
    
    # Check bounds
    if not (0 <= pos[0] < grid.width and 0 <= pos[1] < grid.height):
        return False
    
    # Get cell at position
    cell = grid.get(pos[0], pos[1])
    
    # Cell is walkable if it's None or can be overlapped
    return cell is None or cell.can_overlap()

class DiagonalWallTester:
    """Class to test diagonal movement around walls"""
    
    def __init__(self, env_id='MiniGrid-Empty-8x8-v0', max_episode_steps=150):
        """Initialize the tester with the specified environment"""
        self.env_id = env_id
        
        # Create environment with same parameters as in main.py
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
            diagonal_success_reward=0.01,
            diagonal_failure_penalty=0
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
                else:
                    env.agent_pos = tuple(position_array)
            
            if hasattr(env, 'agent_dir'):
                env.agent_dir = direction
                
            env = env.env
    
    def place_wall(self, x, y):
        """Place a wall at the specified position"""
        self.base_env.grid.set(x, y, Wall())
        self.wall_positions.append((x, y))
    
    def calculate_diagonal_move_positions(self, agent_pos, agent_dir, diag_direction):
        """Calculate the positions that would be involved in a diagonal move"""
        # Define direction vectors for the four cardinal directions
        direction_vecs = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1])   # up
        }
        
        agent_pos = ensure_array(agent_pos)
        forward_vec = direction_vecs[agent_dir]
        
        # Calculate left and right vectors relative to the agent's direction
        left_vec = np.array([-forward_vec[1], forward_vec[0]])
        right_vec = np.array([forward_vec[1], -forward_vec[0]])
        
        # Define diagonal movement based on direction
        if diag_direction == "left":
            # Forward + left diagonal
            diag_vec = forward_vec + left_vec
            lateral_vec = left_vec
        else:  # Right diagonal
            # Forward + right diagonal
            diag_vec = forward_vec + right_vec
            lateral_vec = right_vec
        
        # Calculate all positions
        forward_pos = agent_pos + forward_vec
        lateral_pos = agent_pos + lateral_vec
        diagonal_pos = agent_pos + diag_vec
        
        # Check walkability
        forward_walkable = is_walkable(self.base_env, forward_pos)
        lateral_walkable = is_walkable(self.base_env, lateral_pos)
        diagonal_walkable = is_walkable(self.base_env, diagonal_pos)
        
        return {
            "agent_pos": agent_pos,
            "forward_pos": forward_pos,
            "lateral_pos": lateral_pos,
            "diagonal_pos": diagonal_pos,
            "forward_walkable": forward_walkable,
            "lateral_walkable": lateral_walkable,
            "diagonal_walkable": diagonal_walkable,
            "valid_diagonal": forward_walkable and lateral_walkable and diagonal_walkable
        }
    
    def create_wall_configuration(self):
        """Set up a wall configuration for testing diagonal movement."""
        # Reset the environment first
        self.env.reset()
        self.wall_positions = []
        
        # Create wall configuration:
        # - A wall corner at (3,3)-(3,4)-(4,3)
        # - A wall at (3,6)
        # - A wall at (4,1)
        # - A wall at (6,3)
        
        # Corner walls
        self.place_wall(3, 3)  # Corner wall
        self.place_wall(3, 4)  # Wall below the corner
        self.place_wall(4, 3)  # Wall to the right of the corner
        
        # Additional walls
        self.place_wall(3, 6)  # Wall at bottom left
        self.place_wall(4, 1)  # Wall at top right
        self.place_wall(6, 3)  # Wall at middle right
    
    def test_diagonal_wall_blocking(self):
        """Test that diagonal moves are properly blocked by walls"""
        # Set up wall configuration
        self.create_wall_configuration()
        
        self._print_grid()
        print("Wall positions:", self.wall_positions)
        
        # Test cases: each is a tuple of (agent_pos, agent_dir, action, expected_success)
        test_cases = [
            # TEST CASE 1: Diagonal move through a corner (should fail)
            # Agent at (2,2) facing right (0), diagonal right (moving into the corner from top left)
            ((2, 2), 0, "right", False, "Moving diagonally into a corner from top left (blocked by walls)"),
            
            # TEST CASE 2: Diagonal move through a corner (should fail)
            # Agent at (5,5) facing left (2), diagonal left (moving into the corner from bottom right)
            ((5, 5), 2, "left", False, "Moving diagonally into a corner from bottom right (blocked by walls)"),
            
            # TEST CASE 3: Diagonal blocked by a wall in forward direction (should fail)
            # Agent at (2,3) facing right (0), diagonal right (forward path blocked by wall)
            ((2, 3), 0, "right", False, "Diagonal move blocked by wall in forward direction"),
            
            # TEST CASE 4: Diagonal blocked by a wall in lateral direction (should fail)
            # Agent at (3,2) facing down (1), diagonal left (lateral path blocked by wall)
            ((3, 2), 1, "left", False, "Diagonal move blocked by wall in lateral direction"),
            
            # TEST CASE 5: Adjacent to a wall, but diagonal is clear (should succeed)
            # Agent at (2,4) facing right (0), diagonal right (diagonal path is clear)
            ((2, 4), 0, "right", True, "Diagonal path is clear despite adjacent walls"),
            
            # TEST CASE 6: Same as TEST CASE 5 but opposite side (should succeed)
            # Agent at (4,2) facing down (1), diagonal right (diagonal path is clear)
            ((4, 2), 1, "right", True, "Diagonal path is clear despite adjacent walls (opposite side)"),
            
            # TEST CASE 7: Diagonal move into an isolated wall (should fail)
            # Agent at (2,6) facing right (0), diagonal left (diagonal path is blocked by wall)
            ((2, 6), 0, "left", False, "Diagonal move into an isolated wall"),
            
            # TEST CASE 8: Diagonal move around an isolated wall (should succeed)
            # Agent at (5,3) facing left (2), diagonal right (diagonal path is clear around isolated wall)
            ((5, 3), 2, "right", True, "Diagonal move around an isolated wall"),
            
            # TEST CASE 9: Entirely clear diagonal move (should succeed)
            # Agent at (1,1) facing right (0), diagonal right (all paths are clear)
            ((1, 1), 0, "right", True, "Entirely clear diagonal move"),
            
            # TEST CASE 10: Diagonal blocked by the target position being a wall (should fail)
            # Agent at (3,5) facing right (0), diagonal left (target position is a wall)
            ((3, 5), 0, "left", False, "Diagonal move where target position is a wall"),
        ]
        
        results = []
        
        # Run through all test cases
        for i, (pos, direction, diag_direction, expected_success, description) in enumerate(test_cases):
            # Set agent position and direction
            self.set_agent_position_and_direction(pos, direction)
            
            # Print test case information
            print(f"\n==== Test Case {i+1}: {description} ====")
            print(f"Agent at {pos}, facing {DIRECTION_NAMES[direction]}, diagonal {diag_direction}")
            print(f"Expected success: {expected_success}")
            
            # Calculate and print information about the diagonal move
            move_info = self.calculate_diagonal_move_positions(pos, direction, diag_direction)
            
            # Highlight the key positions on the grid
            highlight_positions = [
                (move_info["forward_pos"], "F"),
                (move_info["lateral_pos"], "L"),
                (move_info["diagonal_pos"], "D")
            ]
            
            # Display grid with highlighted positions
            self._print_grid(highlight_positions=highlight_positions)
            
            # Print detailed movement information
            print(f"From position {move_info['agent_pos']}:")
            print(f"  - Forward position: {move_info['forward_pos']} (walkable: {move_info['forward_walkable']})")
            print(f"  - Lateral position: {move_info['lateral_pos']} (walkable: {move_info['lateral_walkable']})")
            print(f"  - Diagonal position: {move_info['diagonal_pos']} (walkable: {move_info['diagonal_walkable']})")
            print(f"Valid diagonal move according to criteria: {move_info['valid_diagonal']}")
            
            # Try the diagonal move
            diagonal_result = self.action_wrapper._diagonal_move(diag_direction)
            actual_success = "failed" not in diagonal_result[4]
            
            # Check if result matches expectation
            success_match = actual_success == expected_success
            
            print(f"Actual success: {actual_success}")
            if not success_match:
                print(f"⚠️ MISMATCH: Expected {expected_success}, got {actual_success}")
            
            # Store the result
            results.append({
                "test_case": i + 1,
                "description": description,
                "position": to_list(pos),
                "direction": direction,
                "direction_name": DIRECTION_NAMES[direction],
                "diagonal_direction": diag_direction,
                "expected_success": expected_success,
                "actual_success": actual_success,
                "success_match": success_match,
                "forward_position": to_list(move_info["forward_pos"]),
                "forward_walkable": move_info["forward_walkable"],
                "lateral_position": to_list(move_info["lateral_pos"]),
                "lateral_walkable": move_info["lateral_walkable"],
                "diagonal_position": to_list(move_info["diagonal_pos"]),
                "diagonal_walkable": move_info["diagonal_walkable"],
                "valid_diagonal": move_info["valid_diagonal"]
            })
        
        # Save results to JSON
        output_file = os.path.join(os.path.dirname(__file__), "diagonal_wall_results.json")
        with open(output_file, 'w') as f:
            json.dump({
                "environment": self.env_id,
                "wall_positions": self.wall_positions,
                "results": results
            }, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
        # Calculate and print summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["success_match"])
        
        print(f"\n==== Test Summary ====")
        print(f"Total tests: {total_tests}")
        print(f"Passed tests: {passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        # Categorize the mismatches
        failed_cases = [r for r in results if not r["success_match"]]
        false_positives = [r for r in failed_cases if r["actual_success"] and not r["expected_success"]]
        false_negatives = [r for r in failed_cases if not r["actual_success"] and r["expected_success"]]
        
        if false_positives:
            print("\nFalse Positives (moves that succeeded when they should have failed):")
            for r in false_positives:
                print(f"  - Test Case {r['test_case']}: {r['description']}")
        
        if false_negatives:
            print("\nFalse Negatives (moves that failed when they should have succeeded):")
            for r in false_negatives:
                print(f"  - Test Case {r['test_case']}: {r['description']}")
        
        return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    
    tester = DiagonalWallTester()
    tester.test_diagonal_wall_blocking() 