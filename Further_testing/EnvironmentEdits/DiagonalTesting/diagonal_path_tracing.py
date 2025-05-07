#!/usr/bin/env python3
"""
Test script to trace paths taken by agents using diagonal movements.
This script visualizes the agent's path when using diagonal movements
to help understand how it navigates the environment and to identify
any issues with diagonal movement implementation.
"""

import os
import sys
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow
import json
from collections import defaultdict

# Add the root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the environment generation functions
from EnvironmentEdits.EnvironmentGeneration import make_env
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper

# Set seeds for reproducibility
SEED = 12345

# Define environment types to test
ENV_TYPES = [
    'MiniGrid-Empty-8x8-v0',
    'MiniGrid-FourRooms-v0',
    'MiniGrid-SimpleCrossingS9N1-v0',
    'MiniGrid-LavaCrossingS9N1-v0',
]

# Direction names for better readability
DIRECTION_NAMES = {
    0: "right",
    1: "down",
    2: "left",
    3: "up"
}

# Direction vectors
DIRECTION_VECTORS = {
    0: np.array([1, 0]),   # right
    1: np.array([0, 1]),   # down
    2: np.array([-1, 0]),  # left
    3: np.array([0, -1])   # up
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

class DiagonalPathTracer:
    """Class to trace and visualize paths using diagonal movements"""
    
    def __init__(self, env_id, max_episode_steps=150, use_random_spawn=False):
        """Initialize the tracer with the specified environment"""
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
        
        # Reset the environment
        self.env.reset()
        
        # Initialize path data
        self.path_data = []
        self.current_step = 0
        self.path_positions = []
        self.path_directions = []
        self.failed_moves = []
    
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
    
    def reset_environment(self):
        """Reset the environment and path data"""
        self.env.reset()
        self.path_data = []
        self.current_step = 0
        self.path_positions = [ensure_array(self.base_env.agent_pos)]
        self.path_directions = [self.base_env.agent_dir]
        self.failed_moves = []
    
    def take_action(self, action):
        """Take an action and record the result"""
        # Record pre-action state
        pre_pos = safe_copy(self.base_env.agent_pos)
        pre_dir = self.base_env.agent_dir
        
        # Take the action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Record post-action state
        post_pos = safe_copy(self.base_env.agent_pos)
        post_dir = self.base_env.agent_dir
        
        # Determine if the action succeeded (position changed)
        pre_pos_array = ensure_array(pre_pos)
        post_pos_array = ensure_array(post_pos)
        success = not np.array_equal(pre_pos_array, post_pos_array) or pre_dir != post_dir
        
        # For diagonal moves, check the info
        if action in [3, 4]:  # Diagonal left or right
            if 'failed' in info:
                success = not info['failed']
        
        # Save the path data
        self.current_step += 1
        self.path_positions.append(ensure_array(post_pos))
        self.path_directions.append(post_dir)
        
        # Record failed moves
        if not success and action >= 2:  # Only record failed forward or diagonal moves
            self.failed_moves.append({
                'step': self.current_step,
                'position': to_list(pre_pos),
                'direction': pre_dir,
                'action': action,
                'action_name': ACTION_NAMES[action],
                'info': info
            })
        
        # Add to path data
        self.path_data.append({
            'step': self.current_step,
            'pre_position': to_list(pre_pos),
            'pre_direction': pre_dir,
            'action': action,
            'action_name': ACTION_NAMES[action],
            'post_position': to_list(post_pos),
            'post_direction': post_dir,
            'reward': reward,
            'success': success,
            'info': info
        })
        
        return obs, reward, terminated, truncated, info
    
    def take_random_actions(self, num_actions, include_diagonal=True):
        """Take a series of random actions and record the results"""
        self.reset_environment()
        
        for _ in range(num_actions):
            # Choose a random action
            if include_diagonal:
                action = random.randint(0, 4)  # All actions including diagonals
            else:
                action = random.randint(0, 2)  # Only non-diagonal actions
            
            obs, reward, term, trunc, info = self.take_action(action)
            
            if term or trunc:
                break
        
        # Record summary statistics
        action_counts = defaultdict(int)
        success_counts = defaultdict(int)
        
        for step in self.path_data:
            action = step['action']
            action_counts[action] += 1
            if step['success']:
                success_counts[action] += 1
        
        summary = {
            'total_steps': len(self.path_data),
            'action_counts': {ACTION_NAMES[a]: count for a, count in action_counts.items()},
            'success_rates': {ACTION_NAMES[a]: success_counts[a]/count if count > 0 else 0 
                             for a, count in action_counts.items()},
            'failed_moves': len(self.failed_moves)
        }
        
        return summary
    
    def visualize_path(self, save_path=None):
        """Visualize the path taken by the agent"""
        grid = self.base_env.grid
        width, height = grid.width, grid.height
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw the grid
        for i in range(width):
            for j in range(height):
                cell = grid.get(i, j)
                
                if cell is None:
                    # Empty cell
                    ax.add_patch(Rectangle((i, height-j-1), 1, 1, facecolor='white', edgecolor='black', linewidth=0.5))
                elif cell.type == 'wall':
                    # Wall
                    ax.add_patch(Rectangle((i, height-j-1), 1, 1, facecolor='gray', edgecolor='black'))
                elif cell.type == 'lava':
                    # Lava
                    ax.add_patch(Rectangle((i, height-j-1), 1, 1, facecolor='red', edgecolor='black'))
                elif cell.type == 'goal':
                    # Goal
                    ax.add_patch(Rectangle((i, height-j-1), 1, 1, facecolor='green', edgecolor='black'))
                else:
                    # Other object
                    ax.add_patch(Rectangle((i, height-j-1), 1, 1, facecolor='yellow', edgecolor='black'))
        
        # Draw the path
        path_x = [p[0] + 0.5 for p in self.path_positions]
        path_y = [height - p[1] - 0.5 for p in self.path_positions]
        
        # Draw path lines
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
        
        # Draw start and end points
        ax.plot(path_x[0], path_y[0], 'go', markersize=10)  # Start
        ax.plot(path_x[-1], path_y[-1], 'ro', markersize=10)  # End
        
        # Draw direction arrows at each step
        for i in range(len(self.path_positions)):
            x, y = self.path_positions[i]
            dir = self.path_directions[i]
            dx, dy = DIRECTION_VECTORS[dir]
            
            # Adjust for plot coordinates (y is inverted)
            plt_x = x + 0.5
            plt_y = height - y - 0.5
            plt_dx = dx * 0.3
            plt_dy = -dy * 0.3  # Invert y direction
            
            ax.arrow(plt_x, plt_y, plt_dx, plt_dy, 
                    head_width=0.2, head_length=0.2, fc='blue', ec='blue', alpha=0.5)
        
        # Mark failed moves
        for fail in self.failed_moves:
            x, y = fail['position']
            plt_x = x + 0.5
            plt_y = height - y - 0.5
            
            ax.plot(plt_x, plt_y, 'rx', markersize=8, mew=2)
        
        # Set plot limits and labels
        ax.set_xlim(-0.5, width + 0.5)
        ax.set_ylim(-0.5, height + 0.5)
        ax.set_xticks(np.arange(0, width + 1, 1))
        ax.set_yticks(np.arange(0, height + 1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, linestyle='-', alpha=0.7)
        
        # Set title
        ax.set_title(f"Path in {self.env_id} - {len(self.path_positions)} steps, {len(self.failed_moves)} failed moves")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Start'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='End'),
            Line2D([0], [0], marker='x', color='r', markersize=8, label='Failed Move'),
            Line2D([0], [0], color='b', lw=2, alpha=0.7, label='Path')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Save or show the figure
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Path visualization saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def run_wall_corner_test(self, steps=50, save_visualization=True):
        """Run a test focusing on wall corners by placing walls in the environment"""
        # Create a custom environment with walls
        self.reset_environment()
        
        # Add some walls to create corners
        grid = self.base_env.grid
        width, height = grid.width, grid.height
        
        # Clear any existing walls (except outer walls)
        for i in range(1, width-1):
            for j in range(1, height-1):
                if grid.get(i, j) is not None and grid.get(i, j).type == 'wall':
                    grid.set(i, j, None)
        
        # Add L-shaped walls
        for x, y in [(2, 2), (5, 5)]:
            # Add an L shape
            grid.set(x, y, gym.minigrid.minigrid.Wall())
            grid.set(x+1, y, gym.minigrid.minigrid.Wall())
            grid.set(x, y+1, gym.minigrid.minigrid.Wall())
        
        # Position the agent
        self.base_env.agent_pos = np.array([1, 1])
        self.base_env.agent_dir = 0  # Facing right
        
        # Record initial position and direction
        self.path_positions = [ensure_array(self.base_env.agent_pos)]
        self.path_directions = [self.base_env.agent_dir]
        
        # Take random actions with high probability of diagonal moves
        for _ in range(steps):
            # Bias towards diagonal moves
            r = random.random()
            if r < 0.4:
                action = 3  # Diagonal left
            elif r < 0.8:
                action = 4  # Diagonal right
            else:
                action = random.randint(0, 2)  # Other actions
            
            # Take the action
            obs, reward, term, trunc, info = self.take_action(action)
            
            if term or trunc:
                break
        
        # Save path data
        os.makedirs(os.path.dirname(__file__), exist_ok=True)
        output_file = os.path.join(os.path.dirname(__file__), f"corner_test_path_data_{self.env_id.replace('-', '_')}.json")
        with open(output_file, 'w') as f:
            json.dump({
                'environment': self.env_id,
                'steps': len(self.path_data),
                'path_data': self.path_data,
                'failed_moves': self.failed_moves
            }, f, indent=2)
        print(f"Path data saved to {output_file}")
        
        # Visualize the path
        if save_visualization:
            save_path = os.path.join(os.path.dirname(__file__), f"corner_test_path_{self.env_id.replace('-', '_')}.png")
            self.visualize_path(save_path=save_path)
        else:
            self.visualize_path()
        
        # Return summary statistics
        return {
            'total_steps': len(self.path_data),
            'diagonal_attempts': sum(1 for step in self.path_data if step['action'] in [3, 4]),
            'diagonal_success': sum(1 for step in self.path_data if step['action'] in [3, 4] and step['success']),
            'failed_moves': len(self.failed_moves)
        }

    def examine_is_walkable_implementation(self):
        """Examine the implementation of _is_walkable by testing various cell combinations"""
        print(f"\n==== Examining walkability implementation in {self.env_id} ====")
        self.reset_environment()
        
        # Create a custom grid with various cell types
        grid = self.base_env.grid
        width, height = grid.width, grid.height
        
        # Clear any existing objects (except outer walls)
        for i in range(1, width-1):
            for j in range(1, height-1):
                grid.set(i, j, None)
        
        # Print out the is_walkable implementation
        print("\nWalkability is determined by:")
        print("1. Position is within grid bounds")
        print("2. Cell is either None or can_overlap() returns True")
        
        # Test walkability for different cell types
        test_cases = []
        
        # Add some walls
        wall_positions = [(2, 2), (2, 3), (3, 2), (5, 5)]
        for x, y in wall_positions:
            grid.set(x, y, gym.minigrid.minigrid.Wall())
        
        # Add a goal
        grid.set(6, 6, gym.minigrid.minigrid.Goal())
        
        # Add lava if available
        try:
            grid.set(4, 4, gym.minigrid.minigrid.Lava())
        except:
            pass
        
        # Test positions
        test_positions = [
            (1, 1),  # Empty cell
            (2, 2),  # Wall
            (6, 6),  # Goal
            (4, 4),  # Lava (if available)
            # Positions adjacent to walls
            (2, 1),  # Above wall
            (1, 2),  # Left of wall
            (3, 3),  # Diagonal from wall corner
        ]
        
        # Test each position
        for x, y in test_positions:
            pos = np.array([x, y])
            cell_type = "None" if grid.get(x, y) is None else grid.get(x, y).type
            walkable = is_walkable(self.base_env, pos)
            
            test_cases.append({
                'position': [x, y],
                'cell_type': cell_type,
                'walkable': walkable
            })
            
            print(f"Position ({x}, {y}): Cell type = {cell_type}, Walkable = {walkable}")
        
        # Test diagonal movement through walls
        print("\nTesting diagonal movement relative to walls:")
        
        # Position the agent near a wall
        self.base_env.agent_pos = np.array([1, 1])
        
        # Test all directions
        for direction in range(4):
            self.base_env.agent_dir = direction
            
            # Get forward and diagonal vectors
            forward_vec = DIRECTION_VECTORS[direction]
            left_vec = np.array([-forward_vec[1], forward_vec[0]])
            right_vec = np.array([forward_vec[1], -forward_vec[0]])
            
            agent_pos = ensure_array(self.base_env.agent_pos)
            diag_left_pos = agent_pos + forward_vec + left_vec
            diag_right_pos = agent_pos + forward_vec + right_vec
            
            # Test with is_walkable
            walkable_left = is_walkable(self.base_env, diag_left_pos)
            walkable_right = is_walkable(self.base_env, diag_right_pos)
            
            # Test with action wrapper
            saved_pos = safe_copy(self.base_env.agent_pos)
            saved_dir = self.base_env.agent_dir
            
            diag_left_result = self.action_wrapper._diagonal_move("left")
            diag_left_success = "failed" not in diag_left_result[4]
            
            # Reset position and direction
            if isinstance(saved_pos, tuple):
                self.base_env.agent_pos = saved_pos  # For tuples, direct assignment is safe
            else:
                self.base_env.agent_pos = saved_pos.copy()  # For arrays, need to copy
            self.base_env.agent_dir = saved_dir
            
            diag_right_result = self.action_wrapper._diagonal_move("right")
            diag_right_success = "failed" not in diag_right_result[4]
            
            # Print results
            print(f"\nAgent at {saved_pos}, facing {DIRECTION_NAMES[direction]}")
            print(f"  Diagonal left to {diag_left_pos}: is_walkable={walkable_left}, action success={diag_left_success}")
            print(f"  Diagonal right to {diag_right_pos}: is_walkable={walkable_right}, action success={diag_right_success}")
            
            # Check for inconsistencies
            if walkable_left != diag_left_success or walkable_right != diag_right_success:
                print("  ⚠️ INCONSISTENCY DETECTED between is_walkable and action result!")
        
        return test_cases


def run_wall_corner_tests():
    """Run tests for all environment types focusing on wall corners"""
    results = {}
    
    for env_id in ENV_TYPES:
        try:
            print(f"\n======================================================")
            print(f"TESTING ENVIRONMENT: {env_id} - Wall Corner Test")
            print(f"======================================================")
            
            tracer = DiagonalPathTracer(env_id)
            env_results = tracer.run_wall_corner_test(steps=50, save_visualization=True)
            results[env_id] = env_results
            
            print(f"Results for {env_id}:")
            print(f"  Total steps: {env_results['total_steps']}")
            print(f"  Diagonal attempts: {env_results['diagonal_attempts']}")
            diagonal_success_rate = 0
            if env_results['diagonal_attempts'] > 0:
                diagonal_success_rate = (env_results['diagonal_success'] / env_results['diagonal_attempts']) * 100
            print(f"  Diagonal success rate: {diagonal_success_rate:.1f}%")
            print(f"  Failed moves: {env_results['failed_moves']}")
            
        except Exception as e:
            print(f"Error testing {env_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save overall results
    output_file = os.path.join(os.path.dirname(__file__), "wall_corner_tests_summary.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nOverall results saved to {output_file}")
    
    return results


def examine_is_walkable_implementations():
    """Examine _is_walkable implementations across environments"""
    for env_id in ENV_TYPES:
        try:
            tracer = DiagonalPathTracer(env_id)
            tracer.examine_is_walkable_implementation()
        except Exception as e:
            print(f"Error examining {env_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    
    import argparse
    parser = argparse.ArgumentParser(description='Test and visualize diagonal movements')
    parser.add_argument('--test', type=str, choices=['corners', 'walkable', 'all'], default='all',
                      help='Test to run (corners: test wall corners, walkable: examine _is_walkable, all: run all tests)')
    
    args = parser.parse_args()
    
    if args.test == 'corners' or args.test == 'all':
        run_wall_corner_tests()
        
    if args.test == 'walkable' or args.test == 'all':
        examine_is_walkable_implementations() 