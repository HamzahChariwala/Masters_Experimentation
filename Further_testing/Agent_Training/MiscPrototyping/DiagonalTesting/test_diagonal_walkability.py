#!/usr/bin/env python3
"""
Test script to verify _is_walkable method functionality with diagonal moves.
This script creates environments similar to those used in training and
tests diagonal movement functionality across different scenarios.
"""

import os
import sys
import numpy as np
import time
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from pprint import pprint

# Add the root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the environment generation functions
from EnvironmentEdits.EnvironmentGeneration import make_env
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper

# Define environment types to test
ENV_TYPES = [
    'MiniGrid-Empty-8x8-v0',
    'MiniGrid-FourRooms-v0',
    'MiniGrid-SimpleCrossingS9N1-v0',
    'MiniGrid-LavaCrossingS9N1-v0',
]

# Set seeds for reproducibility
SEED = 12345

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

class DiagonalWalkabilityTester:
    """Class to test diagonal movement walkability in MiniGrid environments"""
    
    def __init__(self, env_id, max_episode_steps=150, use_random_spawn=False):
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
            use_random_spawn=use_random_spawn,
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
        print(f"Agent position: {self.base_env.agent_pos}, direction: {self.base_env.agent_dir}")
    
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
        
        print(f"\nDirection vectors:")
        print(f"  Forward ({agent_dir}): {forward_vec}")
        print(f"  Left: {left_vec}")
        print(f"  Right: {right_vec}")
        print(f"  Diagonal Left: {diag_left}")
        print(f"  Diagonal Right: {diag_right}")
    
    def reset_and_randomize_agent(self, num_actions=10):
        """Reset environment and randomize agent direction and position by taking random actions"""
        obs, info = self.env.reset()
        
        # Take some random actions to randomize agent position and direction
        for _ in range(num_actions):
            action = random.randint(0, 2)  # Only use non-diagonal actions for setup
            self.env.step(action)
        
        return obs, info
    
    def test_walkability_from_position(self):
        """Test walkability for diagonals from current position"""
        # Get current agent position and direction
        agent_pos = ensure_array(self.base_env.agent_pos)
        agent_dir = self.base_env.agent_dir
        
        # Show environment state
        self._print_grid()
        self._print_direction_vectors()
        
        # Get direction vectors
        direction_vecs = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }
        
        forward_vec = direction_vecs[agent_dir]
        left_vec = np.array([-forward_vec[1], forward_vec[0]])
        right_vec = np.array([forward_vec[1], -forward_vec[0]])
        
        # Calculate diagonal positions
        diag_left_pos = agent_pos + forward_vec + left_vec
        diag_right_pos = agent_pos + forward_vec + right_vec
        
        # Test direct walkability with is_walkable
        walkable_forward = is_walkable(self.base_env, agent_pos + forward_vec)
        walkable_left_diag = is_walkable(self.base_env, diag_left_pos)
        walkable_right_diag = is_walkable(self.base_env, diag_right_pos)
        
        # Log information
        print("\nWalkability Test Results:")
        print(f"  Forward position {agent_pos + forward_vec}: {walkable_forward}")
        print(f"  Diagonal Left position {diag_left_pos}: {walkable_left_diag}")
        print(f"  Diagonal Right position {diag_right_pos}: {walkable_right_diag}")
        
        # Try diagonal moves through the action wrapper
        saved_pos = safe_copy(self.base_env.agent_pos)
        saved_dir = self.base_env.agent_dir
        
        # Try diagonal left move
        print("\nTesting diagonal left move through action wrapper:")
        diag_left_result = self.action_wrapper._diagonal_move("left")
        diag_left_success = "failed" not in diag_left_result[4]
        print(f"  Success: {diag_left_success}")
        print(f"  Reward: {diag_left_result[1]}")
        print(f"  Info: {diag_left_result[4]}")
        
        # Reset agent position and direction 
        if isinstance(saved_pos, tuple):
            self.base_env.agent_pos = saved_pos  # For tuples, direct assignment is safe
        else:
            self.base_env.agent_pos = saved_pos.copy()  # For arrays, need to copy
        self.base_env.agent_dir = saved_dir
        
        # Try diagonal right move
        print("\nTesting diagonal right move through action wrapper:")
        diag_right_result = self.action_wrapper._diagonal_move("right")
        diag_right_success = "failed" not in diag_right_result[4]
        print(f"  Success: {diag_right_success}")
        print(f"  Reward: {diag_right_result[1]}")
        print(f"  Info: {diag_right_result[4]}")
        
        # Report if there are inconsistencies
        if diag_left_success != walkable_left_diag or diag_right_success != walkable_right_diag:
            print("\n⚠️ INCONSISTENCY DETECTED:")
            if diag_left_success != walkable_left_diag:
                print(f"  Diagonal left: is_walkable={walkable_left_diag}, but action {'succeeded' if diag_left_success else 'failed'}")
            if diag_right_success != walkable_right_diag:
                print(f"  Diagonal right: is_walkable={walkable_right_diag}, but action {'succeeded' if diag_right_success else 'failed'}")
        
        # Return all test results
        return {
            "environment": self.env_id,
            "agent_pos": agent_pos.tolist() if isinstance(agent_pos, np.ndarray) else list(agent_pos),
            "agent_dir": agent_dir,
            "forward_walkable": walkable_forward,
            "left_diag_pos": diag_left_pos.tolist() if isinstance(diag_left_pos, np.ndarray) else list(diag_left_pos),
            "left_diag_walkable": walkable_left_diag,
            "left_diag_action_success": diag_left_success,
            "right_diag_pos": diag_right_pos.tolist() if isinstance(diag_right_pos, np.ndarray) else list(diag_right_pos),
            "right_diag_walkable": walkable_right_diag, 
            "right_diag_action_success": diag_right_success,
            "has_inconsistency": diag_left_success != walkable_left_diag or diag_right_success != walkable_right_diag
        }
    
    def test_multiple_positions(self, positions=20):
        """Test walkability across multiple positions"""
        all_results = []
        
        for i in range(positions):
            print(f"\n==== Test {i+1}/{positions} ====")
            self.reset_and_randomize_agent(random.randint(5, 15))
            result = self.test_walkability_from_position()
            all_results.append(result)
        
        # Count inconsistencies
        inconsistencies = sum(1 for r in all_results if r["has_inconsistency"])
        
        print(f"\n==== Summary ====")
        print(f"Environment: {self.env_id}")
        print(f"Tested {positions} positions")
        print(f"Found {inconsistencies} inconsistencies ({inconsistencies/positions*100:.1f}%)")
        
        return all_results


def test_environment(env_id, positions=20, save_results=True):
    """Test a specific environment"""
    print(f"\n======================================================")
    print(f"TESTING ENVIRONMENT: {env_id}")
    print(f"======================================================")
    
    tester = DiagonalWalkabilityTester(env_id)
    results = tester.test_multiple_positions(positions)
    
    if save_results:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(__file__), exist_ok=True)
        
        # Save results to file
        import json
        output_file = os.path.join(os.path.dirname(__file__), f"{env_id.replace('-', '_')}_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return results


def run_all_tests(positions_per_env=20):
    """Run tests on all environment types"""
    all_results = {}
    
    for env_id in ENV_TYPES:
        try:
            results = test_environment(env_id, positions=positions_per_env)
            all_results[env_id] = results
        except Exception as e:
            print(f"Error testing {env_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summarize overall results
    print("\n==== OVERALL SUMMARY ====")
    for env_id, results in all_results.items():
        inconsistencies = sum(1 for r in results if r["has_inconsistency"])
        print(f"{env_id}: {inconsistencies}/{len(results)} inconsistencies ({inconsistencies/len(results)*100:.1f}%)")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Test diagonal walkability in MiniGrid environments')
    parser.add_argument('--env', type=str, choices=ENV_TYPES + ['all'], default='all',
                        help='Environment to test (default: all)')
    parser.add_argument('--positions', type=int, default=20,
                        help='Number of positions to test per environment (default: 20)')
    
    args = parser.parse_args()
    
    if args.env == 'all':
        run_all_tests(positions_per_env=args.positions)
    else:
        test_environment(args.env, positions=args.positions) 