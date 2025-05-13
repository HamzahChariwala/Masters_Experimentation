#!/usr/bin/env python3
"""
Test script to visualize and test diagonal movement behavior around walls and corners.
This script focuses on edge cases around walls and corners to understand 
how the _is_walkable method handles diagonal movements in these scenarios.
"""

import os
import sys
import numpy as np
import time
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from pprint import pprint
import json
from collections import defaultdict

# Add the root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the environment generation functions
from EnvironmentEdits.EnvironmentGeneration import make_env
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper

# Set seeds for reproducibility
SEED = 12345

# Direction names for better readability
DIRECTION_NAMES = {
    0: "right",
    1: "down",
    2: "left",
    3: "up"
}

def safe_copy(value):
    """Safely copy a value that could be a tuple or numpy array"""
    if isinstance(value, np.ndarray):
        return value.copy()
    elif isinstance(value, tuple):
        return tuple(value)  # Tuples are immutable, so this effectively creates a copy
    else:
        return value  # For other types, return as is

def ensure_array(value):
    """Convert a tuple or other iterable to a numpy array"""
    if isinstance(value, np.ndarray):
        return value
    else:
        return np.array(value)

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
    """Class to test diagonal movement around walls in MiniGrid environments"""
    
    def __init__(self, env_id='MiniGrid-Empty-16x16-v0', max_episode_steps=150):
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
        
        # Test positions
        self.test_positions = []
        self.grid = self.base_env.grid
        
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
    
    def _print_grid(self):
        """Print the current grid layout (simplified)"""
        print("\nGrid Layout:")
        grid = self.base_env.grid
        
        width, height = grid.width, grid.height
        output = []
        
        for j in range(height):
            row = []
            for i in range(width):
                cell = grid.get(i, j)
                if cell is None:
                    ch = "."  # Empty
                elif cell.type == 'wall':
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
                    
                row.append(ch)
            output.append("".join(row))
        
        print("\n".join(output))
        print(f"Agent position: {self.base_env.agent_pos}, direction: {self.base_env.agent_dir} ({DIRECTION_NAMES[self.base_env.agent_dir]})")
    
    def _print_direction_vectors(self):
        """Print the direction vectors for the current agent state"""
        # Get the direction vectors from the action wrapper
        direction_vecs = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }
        
        agent_dir = self.base_env.agent_dir
        forward_vec = direction_vecs[agent_dir]
        left_vec = np.array([-forward_vec[1], forward_vec[0]])
        right_vec = np.array([forward_vec[1], -forward_vec[0]])
        
        diag_left = forward_vec + left_vec
        diag_right = forward_vec + right_vec
        
        print(f"\nDirection vectors for agent facing {DIRECTION_NAMES[agent_dir]}:")
        print(f"  Forward: {forward_vec}")
        print(f"  Left: {left_vec}")
        print(f"  Right: {right_vec}")
        print(f"  Diagonal Left: {diag_left}")
        print(f"  Diagonal Right: {diag_right}")
    
    def create_custom_wall_configuration(self):
        """Create a custom wall configuration to test corner cases"""
        # Reset the environment
        self.env.reset()
        grid = self.base_env.grid
        
        # First clear any existing walls (except outer walls)
        width, height = grid.width, grid.height
        for i in range(1, width-1):
            for j in range(1, height-1):
                if grid.get(i, j) is not None and grid.get(i, j).type == 'wall':
                    grid.set(i, j, None)
        
        # Create some walls for testing corner cases
        self._add_test_wall_pattern(4, 4)
        self._add_test_wall_pattern(10, 4)
        self._add_test_wall_pattern(4, 10)
        
        # Place the agent in a good starting position
        self.base_env.agent_pos = np.array([2, 2])
        self.base_env.agent_dir = 0  # Facing right
        
        self._print_grid()
        return self.env.gen_obs()
    
    def _add_test_wall_pattern(self, x, y):
        """Add a test wall pattern at the specified position"""
        grid = self.base_env.grid
        
        # Add a corner
        grid.set(x, y, gym.minigrid.minigrid.Wall())
        grid.set(x+1, y, gym.minigrid.minigrid.Wall())
        grid.set(x, y+1, gym.minigrid.minigrid.Wall())
        
        # Add an L shape
        grid.set(x+3, y, gym.minigrid.minigrid.Wall())
        grid.set(x+3, y+1, gym.minigrid.minigrid.Wall())
        grid.set(x+3, y+2, gym.minigrid.minigrid.Wall())
        grid.set(x+4, y+2, gym.minigrid.minigrid.Wall())
        grid.set(x+5, y+2, gym.minigrid.minigrid.Wall())
        
        # Add a wall with a gap
        grid.set(x, y+4, gym.minigrid.minigrid.Wall())
        grid.set(x+1, y+4, gym.minigrid.minigrid.Wall())
        grid.set(x+3, y+4, gym.minigrid.minigrid.Wall())
        grid.set(x+4, y+4, gym.minigrid.minigrid.Wall())
        
    def _get_walkability_around_agent(self):
        """Get walkability of all cells adjacent to the agent"""
        pos = ensure_array(self.base_env.agent_pos)
        
        # Check all 8 surrounding positions
        results = {}
        
        # Cardinal directions
        for dx, dy, name in [
            (1, 0, "right"),
            (0, 1, "down"),
            (-1, 0, "left"),
            (0, -1, "up")
        ]:
            test_pos = pos + np.array([dx, dy])
            results[name] = {
                "position": test_pos.copy(),
                "walkable": is_walkable(self.base_env, test_pos)
            }
        
        # Diagonal directions
        for dx, dy, name in [
            (1, 1, "down-right"),
            (-1, 1, "down-left"),
            (-1, -1, "up-left"),
            (1, -1, "up-right")
        ]:
            test_pos = pos + np.array([dx, dy])
            results[name] = {
                "position": test_pos.copy(),
                "walkable": is_walkable(self.base_env, test_pos)
            }
            
        return results
    
    def _test_diagonal_move_success(self, direction):
        """Test if a diagonal move would succeed"""
        agent_pos = safe_copy(self.base_env.agent_pos)
        agent_dir = self.base_env.agent_dir
        
        # Try the diagonal move
        diag_result = self.action_wrapper._diagonal_move(direction)
        success = "failed" not in diag_result[4]
        
        # Restore agent position and direction
        if isinstance(agent_pos, tuple):
            self.base_env.agent_pos = agent_pos  # For tuples, direct assignment is safe
        else:
            self.base_env.agent_pos = agent_pos.copy()  # For arrays, need to copy
        self.base_env.agent_dir = agent_dir
        
        return {
            "success": success,
            "info": diag_result[4]
        }
    
    def test_wall_corner_behavior(self):
        """Test and visualize diagonal movement behavior at wall corners"""
        # Create our test environment with walls
        self.create_custom_wall_configuration()
        
        test_positions = [
            # Test positions around the first corner
            (3, 3, 0),  # right of corner, facing right
            (3, 3, 1),  # right of corner, facing down
            (3, 3, 2),  # right of corner, facing left
            (3, 3, 3),  # right of corner, facing up
            
            # Test positions for L shape
            (2, 2, 0),  # normal open space
            (3, 2, 1),  # near vertical part of L
            (4, 3, 0),  # near horizontal part of L
            
            # Test positions near the wall with a gap
            (2, 4, 3),  # below the wall with gap
            (2, 3, 0),  # near left wall
        ]
        
        results = []
        
        for x, y, dir in test_positions:
            print(f"\n==== Testing position ({x}, {y}), direction {dir} ({DIRECTION_NAMES[dir]}) ====")
            
            # Position agent and set direction
            self.base_env.agent_pos = np.array([x, y])
            self.base_env.agent_dir = dir
            
            self._print_grid()
            self._print_direction_vectors()
            
            # Get walkability of surrounding cells
            walkability = self._get_walkability_around_agent()
            
            print("\nWalkability of surrounding cells:")
            for direction, data in walkability.items():
                pos = data["position"]
                walkable = data["walkable"]
                print(f"  {direction} ({pos[0]}, {pos[1]}): {walkable}")
            
            # Test diagonal left/right moves
            diagonal_left = self._test_diagonal_move_success("left")
            diagonal_right = self._test_diagonal_move_success("right")
            
            print("\nDiagonal move results:")
            print(f"  Left diagonal: {diagonal_left['success']}")
            print(f"  Right diagonal: {diagonal_right['success']}")
            
            # Calculate the expected positions for diagonal moves
            direction_vecs = {
                0: np.array([1, 0]),
                1: np.array([0, 1]),
                2: np.array([-1, 0]),
                3: np.array([0, -1])
            }
            forward_vec = direction_vecs[dir]
            left_vec = np.array([-forward_vec[1], forward_vec[0]])
            right_vec = np.array([forward_vec[1], -forward_vec[0]])
            
            agent_pos = ensure_array(self.base_env.agent_pos)
            diag_left_pos = agent_pos + forward_vec + left_vec
            diag_right_pos = agent_pos + forward_vec + right_vec
            
            # Store result
            result = {
                "agent_position": agent_pos.tolist() if isinstance(agent_pos, np.ndarray) else list(agent_pos),
                "agent_direction": dir,
                "agent_direction_name": DIRECTION_NAMES[dir],
                "walkability": walkability,
                "diagonal_left": {
                    "expected_position": diag_left_pos.tolist() if isinstance(diag_left_pos, np.ndarray) else list(diag_left_pos),
                    "success": diagonal_left["success"],
                    "info": diagonal_left["info"]
                },
                "diagonal_right": {
                    "expected_position": diag_right_pos.tolist() if isinstance(diag_right_pos, np.ndarray) else list(diag_right_pos),
                    "success": diagonal_right["success"],
                    "info": diagonal_right["info"]
                }
            }
            
            results.append(result)
            
        # Save results
        output_file = os.path.join(os.path.dirname(__file__), "wall_corner_test_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
        return results
    
    def test_all_directions_from_position(self, position):
        """Test all agent directions from a specific position"""
        results = []
        
        for direction in range(4):
            # Set agent position and direction
            self.base_env.agent_pos = np.array(position)
            self.base_env.agent_dir = direction
            
            # Get walkability information
            walkability = self._get_walkability_around_agent()
            
            # Test diagonal moves
            diagonal_left = self._test_diagonal_move_success("left")
            diagonal_right = self._test_diagonal_move_success("right")
            
            # Calculate expected positions
            direction_vecs = {
                0: np.array([1, 0]),
                1: np.array([0, 1]),
                2: np.array([-1, 0]),
                3: np.array([0, -1])
            }
            forward_vec = direction_vecs[direction]
            left_vec = np.array([-forward_vec[1], forward_vec[0]])
            right_vec = np.array([forward_vec[1], -forward_vec[0]])
            
            agent_pos = ensure_array(self.base_env.agent_pos)
            diag_left_pos = agent_pos + forward_vec + left_vec
            diag_right_pos = agent_pos + forward_vec + right_vec
            
            # Store result for this direction
            result = {
                "agent_position": position,
                "agent_direction": direction,
                "agent_direction_name": DIRECTION_NAMES[direction],
                "diagonal_left": {
                    "expected_position": diag_left_pos.tolist() if isinstance(diag_left_pos, np.ndarray) else list(diag_left_pos),
                    "success": diagonal_left["success"],
                },
                "diagonal_right": {
                    "expected_position": diag_right_pos.tolist() if isinstance(diag_right_pos, np.ndarray) else list(diag_right_pos),
                    "success": diagonal_right["success"],
                }
            }
            
            results.append(result)
            
        return results


def scan_environment_for_inconsistencies():
    """Scan a larger environment to find inconsistencies in diagonal walking"""
    print("\n==== Scanning Environment for Inconsistencies ====")
    
    # Use a larger environment
    tester = DiagonalWallTester('MiniGrid-Empty-16x16-v0')
    
    # Create custom wall configuration for testing
    tester.create_custom_wall_configuration()
    
    # Scan the entire grid
    grid = tester.base_env.grid
    width, height = grid.width, grid.height
    
    all_results = []
    inconsistencies = []
    
    # Skip the outer wall
    for x in range(1, width-1):
        for y in range(1, height-1):
            # Skip positions that have objects
            if grid.get(x, y) is not None:
                continue
                
            print(f"Testing position ({x}, {y})...")
            position_results = tester.test_all_directions_from_position([x, y])
            all_results.extend(position_results)
            
            # Check for inconsistencies
            for result in position_results:
                left_pos = tuple(result["diagonal_left"]["expected_position"])
                right_pos = tuple(result["diagonal_right"]["expected_position"])
                
                # Check if expected positions are within bounds and don't contain walls
                left_walkable = is_walkable(tester.base_env, left_pos)
                right_walkable = is_walkable(tester.base_env, right_pos)
                
                if left_walkable != result["diagonal_left"]["success"]:
                    inconsistencies.append({
                        "position": result["agent_position"],
                        "direction": result["agent_direction"],
                        "diagonal": "left",
                        "expected_walkable": left_walkable,
                        "actual_success": result["diagonal_left"]["success"],
                        "target_position": left_pos
                    })
                
                if right_walkable != result["diagonal_right"]["success"]:
                    inconsistencies.append({
                        "position": result["agent_position"],
                        "direction": result["agent_direction"],
                        "diagonal": "right",
                        "expected_walkable": right_walkable,
                        "actual_success": result["diagonal_right"]["success"],
                        "target_position": right_pos
                    })
    
    print(f"\nFound {len(inconsistencies)} inconsistencies out of {len(all_results)} tests")
    
    if inconsistencies:
        print("\nInconsistency examples:")
        for i, inconsistency in enumerate(inconsistencies[:5]):  # Show first 5
            print(f"  {i+1}. Position: {inconsistency['position']}, " + 
                  f"Direction: {DIRECTION_NAMES[inconsistency['direction']]}, " +
                  f"Diagonal: {inconsistency['diagonal']}, " +
                  f"Expected walkable: {inconsistency['expected_walkable']}, " +
                  f"Actual: {inconsistency['actual_success']}")
    
    # Save detailed results
    output_file = os.path.join(os.path.dirname(__file__), "inconsistency_scan_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "inconsistencies": inconsistencies,
            "total_tests": len(all_results)
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return inconsistencies


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    
    import argparse
    parser = argparse.ArgumentParser(description='Test diagonal movement behavior around walls')
    parser.add_argument('--test', type=str, choices=['corners', 'scan'], default='corners',
                      help='Test to run (corners: test wall corners, scan: scan for inconsistencies)')
    
    args = parser.parse_args()
    
    if args.test == 'corners':
        tester = DiagonalWallTester()
        tester.test_wall_corner_behavior()
    else:
        scan_environment_for_inconsistencies() 