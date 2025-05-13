#!/usr/bin/env python3
"""
Test script to diagnose how diagonal moves are implemented and processed by the environment.
This script focuses on the actual implementation of diagonal moves and how the environment
updates after a successful diagonal move is determined.
"""

import os
import sys
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

class DiagonalMoveImplementationTester:
    """Class to test diagonal movement implementation"""
    
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
        
        # Define a fixed test position
        self.TEST_POSITION = np.array([3, 3])
    
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
    
    def _calculate_expected_position(self, diag_direction):
        """
        Calculate the expected position after a diagonal move based on the ActionSpace implementation.
        This should match exactly how the CustomActionWrapper._diagonal_move method calculates positions.
        """
        # Define direction vectors for the four cardinal directions
        direction_vecs = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1])   # up
        }
        
        # Get the agent's current direction vector
        agent_dir = self.base_env.agent_dir
        agent_pos = ensure_array(self.base_env.agent_pos)
        forward_vec = direction_vecs[agent_dir]
        
        # Calculate left and right vectors relative to the agent's direction
        left_vec = np.array([-forward_vec[1], forward_vec[0]])
        right_vec = np.array([forward_vec[1], -forward_vec[0]])
        
        # Define diagonal movement based on direction
        if diag_direction == "left":
            # Forward + left diagonal
            diag_vec = forward_vec + left_vec
        else:  # Right diagonal
            # Forward + right diagonal
            diag_vec = forward_vec + right_vec
        
        # Calculate new position
        return agent_pos + diag_vec
    
    def test_diagonal_move_implementation(self):
        """Test how diagonal moves are implemented and processed by the environment"""
        # Reset the environment
        self.env.reset()
        
        results = []
        
        # Test diagonal moves in all directions
        for agent_dir in range(4):
            # Set the agent in the middle of an open area with the specified direction
            self.set_agent_position_and_direction(self.TEST_POSITION, agent_dir)
            
            print(f"\n==== Testing agent direction: {agent_dir} ({DIRECTION_NAMES[agent_dir]}) ====")
            self._print_grid()
            
            # Calculate expected positions for diagonal moves
            left_expected_pos = self._calculate_expected_position("left")
            right_expected_pos = self._calculate_expected_position("right")
            
            print(f"Left diagonal expected position: {left_expected_pos}")
            print(f"Right diagonal expected position: {right_expected_pos}")
            
            # Test left diagonal move
            print("\nTesting left diagonal move:")
            left_result = self.action_wrapper._diagonal_move("left")
            left_success = "failed" not in left_result[4]
            left_actual_pos = safe_copy(self.base_env.agent_pos)
            
            print(f"Success: {left_success}")
            print(f"New position: {left_actual_pos}")
            
            # Check if left actual position matches expected position
            left_pos_match = np.array_equal(ensure_array(left_actual_pos), ensure_array(left_expected_pos)) if left_success else None
            
            if left_success and not left_pos_match:
                print(f"⚠️ Left diagonal position mismatch: expected {left_expected_pos}, got {left_actual_pos}")
            
            # Reset agent position and direction for right diagonal test
            self.set_agent_position_and_direction(self.TEST_POSITION, agent_dir)
            
            # Verify reset worked correctly
            print(f"\nReset for right diagonal test - Position: {self.base_env.agent_pos}, Direction: {self.base_env.agent_dir}")
            
            # Test right diagonal move
            print("\nTesting right diagonal move:")
            right_result = self.action_wrapper._diagonal_move("right")
            right_success = "failed" not in right_result[4]
            right_actual_pos = safe_copy(self.base_env.agent_pos)
            
            print(f"Success: {right_success}")
            print(f"New position: {right_actual_pos}")
            
            # Check if right actual position matches expected position
            right_pos_match = np.array_equal(ensure_array(right_actual_pos), ensure_array(right_expected_pos)) if right_success else None
            
            if right_success and not right_pos_match:
                print(f"⚠️ Right diagonal position mismatch: expected {right_expected_pos}, got {right_actual_pos}")
            
            # Store result
            results.append({
                "agent_direction": agent_dir,
                "agent_direction_name": DIRECTION_NAMES[agent_dir],
                "original_position": to_list(self.TEST_POSITION),
                "left_diagonal": {
                    "expected_position": to_list(left_expected_pos),
                    "actual_position": to_list(left_actual_pos),
                    "success": left_success,
                    "position_match": left_pos_match
                },
                "right_diagonal": {
                    "expected_position": to_list(right_expected_pos),
                    "actual_position": to_list(right_actual_pos),
                    "success": right_success,
                    "position_match": right_pos_match
                }
            })
        
        # Save results
        output_file = os.path.join(os.path.dirname(__file__), "diagonal_move_implementation_results.json")
        with open(output_file, 'w') as f:
            json.dump({
                "environment": self.env_id,
                "results": results
            }, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
        return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    
    tester = DiagonalMoveImplementationTester()
    tester.test_diagonal_move_implementation() 