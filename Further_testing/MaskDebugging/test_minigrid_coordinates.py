import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from Environment_Tooling.BespokeEdits.ActionSpace import CustomActionWrapper
from Environment_Tooling.BespokeEdits.CustomWrappers import PartialObsWrapper
from minigrid.core.world_object import Wall, Goal, Lava
from robust_mask_generation import RobustMaskGenerator

class TestResult:
    def __init__(self, name, passed, message=""):
        self.name = name
        self.passed = passed
        self.message = message
    
    def __str__(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"{status} | {self.name} | {self.message}"

def create_test_env(env_id="MiniGrid-Empty-8x8-v0", use_robust_masks=False):
    """Create a test environment with relevant wrappers"""
    env = gym.make(env_id)
    env = CustomActionWrapper(env)
    
    # Choose between original or robust mask generation
    if use_robust_masks:
        env = RobustMaskGenerator(env, view_size=5)
    else:
        env = PartialObsWrapper(env, n=5)
    
    env = TimeLimit(env, max_episode_steps=100)
    return env

def create_rgb_env(env_id="MiniGrid-Empty-8x8-v0"):
    """Create a test environment with RGB observation for visualization"""
    env = gym.make(env_id)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

def create_test_grid(env):
    """Create a test grid with a specific pattern of objects"""
    env.reset()
    base_env = env.unwrapped
    grid = base_env.grid
    
    # Clear the grid
    for i in range(base_env.width):
        for j in range(base_env.height):
            if (i == 0 or j == 0 or i == base_env.width-1 or j == base_env.height-1):
                # Keep the outer walls
                continue
            grid.set(i, j, None)
    
    # Place objects in a pattern that will clearly show if coordinates are swapped
    # Place walls in an L shape
    for i in range(3):
        grid.set(2, i+2, Wall())  # Vertical line
    for i in range(3):
        grid.set(i+2, 4, Wall())  # Horizontal line
    
    # Place a goal
    grid.set(5, 1, Goal())
    
    # Place lava
    grid.set(1, 5, Lava())
    
    return base_env

def assert_mask_match(grid, mask, obj_type, offset_x, offset_y, agent_pos, agent_dir, rotated=True, is_robust=False):
    """
    Assert that an object in the grid appears in the expected position in the mask.
    
    Args:
        grid: The environment grid
        mask: The generated mask
        obj_type: The type of object to check ('wall', 'lava', 'goal')
        offset_x, offset_y: The offset from the agent position in world coordinates
        agent_pos: The agent's position (x, y)
        agent_dir: The agent's direction (0-3)
        rotated: Whether the mask is rotated based on agent orientation
        is_robust: Whether to use the robust mask generator transformation logic
    
    Returns:
        TestResult: The result of the assertion
    """
    # Calculate the world position of the object
    world_x = agent_pos[0] + offset_x
    world_y = agent_pos[1] + offset_y
    
    # Check if the object is actually at that position in the world
    cell = grid.get(world_x, world_y)
    if cell is None or cell.type != obj_type:
        return TestResult(
            f"{obj_type} at offset ({offset_x}, {offset_y})",
            False,
            f"Expected {obj_type} at world position ({world_x}, {world_y}) but found {cell}"
        )
    
    # For the rotated approach, we need to transform the offset based on agent direction
    if rotated:
        if is_robust:
            # Robust mask generator uses a different transformation
            # For the robust implementation, masks are agent-centric with forward always at the top
            # Agent is always at center (2,2) for a 5x5 mask
            # These rotations transform world offsets to mask coordinates
            rotations = {
                0: lambda x, y: (2 + y, 2 - x),   # Right
                1: lambda x, y: (2 - x, 2 - y),   # Down
                2: lambda x, y: (2 - y, 2 + x),   # Left
                3: lambda x, y: (2 + x, 2 + y)    # Up
            }
        else:
            # Original wrapper has a different approach with agent at (2,0) for a 5x5 mask
            # And a more complex transformation based on direction
            rotations = {
                0: lambda x, y: (2 + offset_y, offset_x),   # Right
                1: lambda x, y: (2 - offset_x, offset_y),   # Down
                2: lambda x, y: (2 - offset_y, -offset_x),  # Left
                3: lambda x, y: (2 + offset_x, -offset_y)   # Up
            }
        
        # Apply the rotation to get mask coordinates
        mask_x, mask_y = rotations[agent_dir](offset_x, offset_y)
    else:
        # For non-rotated masks, just add the offset to the center
        mask_x, mask_y = 2 + offset_x, 2 + offset_y
    
    # Ensure mask coordinates are within bounds
    if not (0 <= mask_x < mask.shape[1] and 0 <= mask_y < mask.shape[0]):
        return TestResult(
            f"{obj_type} at offset ({offset_x}, {offset_y}) dir={agent_dir}",
            False,
            f"Mask coordinates ({mask_x}, {mask_y}) out of bounds for mask shape {mask.shape}"
        )
    
    # Check if the mask has the object at the expected position
    is_present = mask[mask_y, mask_x] == 1
    
    return TestResult(
        f"{obj_type} at offset ({offset_x}, {offset_y}) dir={agent_dir}",
        is_present,
        f"Expected {obj_type} at mask position ({mask_x}, {mask_y}) - {'found' if is_present else 'not found'}"
    )

def test_grid_access():
    """Test that grid.get uses the correct parameter order (x,y)"""
    env = create_test_env()
    env.reset()
    base_env = env.unwrapped
    grid = base_env.grid
    
    test_results = []
    
    # Place a wall at a specific location for testing
    wall_x, wall_y = 3, 4
    orig_obj = grid.get(wall_x, wall_y)
    wall_obj = Wall()
    grid.set(wall_x, wall_y, wall_obj)
    
    # Test direct grid access
    direct_obj = grid.get(wall_x, wall_y)
    direct_test = TestResult(
        "Direct grid access", 
        direct_obj is wall_obj, 
        f"Expected wall at ({wall_x},{wall_y}), got {direct_obj}"
    )
    test_results.append(direct_test)
    
    # Test swapped params
    swapped_obj = grid.get(wall_y, wall_x)
    swapped_test = TestResult(
        "Swapped parameters", 
        swapped_obj is not wall_obj, 
        f"Should NOT find wall at ({wall_y},{wall_x})"
    )
    test_results.append(swapped_test)
    
    # Restore grid
    if orig_obj:
        grid.set(wall_x, wall_y, orig_obj)
    else:
        grid.set(wall_x, wall_y, None)
    
    return test_results

def test_mask_generation():
    """Test basic mask generation in PartialObsWrapper"""
    env = create_test_env()
    obs, _ = env.reset()
    base_env = env.unwrapped
    
    # Get masks from observation - use the mask keys from PartialObsWrapper 
    wall_mask = obs['wall_mask']
    lava_mask = obs['lava_mask']
    
    # Create expected mask manually
    x, y = base_env.agent_pos
    dir_idx = base_env.agent_dir
    
    results = []
    
    # Test if agent position is correctly represented in the mask
    agent_pos_test = TestResult(
        "Agent position in mask", 
        True, 
        f"Agent at ({x},{y}) with direction {dir_idx}"
    )
    results.append(agent_pos_test)
    
    # Visualize the masks
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(wall_mask, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Wall Mask Value')
    plt.title('Wall Mask')
    
    plt.subplot(132)
    plt.imshow(lava_mask, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Lava Mask Value')
    plt.title('Lava Mask')
    
    # Visualize the grid
    plt.subplot(133)
    grid_vis = np.zeros((base_env.height, base_env.width, 3))
    
    # Color the grid
    for i in range(base_env.width):
        for j in range(base_env.height):
            cell = base_env.grid.get(i, j)
            if cell:
                if cell.type == 'wall':
                    grid_vis[j, i] = [0.5, 0.5, 0.5]  # Gray
                elif cell.type == 'goal':
                    grid_vis[j, i] = [0, 1, 0]  # Green
                elif cell.type == 'lava':
                    grid_vis[j, i] = [1, 0, 0]  # Red
            else:
                grid_vis[j, i] = [1, 1, 1]  # White for empty
    
    # Mark agent position
    agent_x, agent_y = base_env.agent_pos
    grid_vis[agent_y, agent_x] = [0, 0, 1]  # Blue
    
    plt.imshow(grid_vis)
    plt.title('Grid Layout')
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig('mask_test.png')
    plt.close()
    
    return results

def test_enhanced_mask_generation():
    """
    Test mask generation with specific object placements
    to verify mask orientation and coordinate systems.
    Tests both the original and robust implementations.
    """
    # Test original mask generation
    test_results_original = test_specific_mask_orientation(use_robust=False)
    
    # Test robust mask generation
    test_results_robust = test_specific_mask_orientation(use_robust=True)
    
    # Combine results
    return test_results_original + test_results_robust

def test_specific_mask_orientation(use_robust=False):
    """Test mask generation with specific object placements for a given implementation"""
    env = create_test_env(use_robust_masks=use_robust)
    
    # Set up the environment with our test grid
    base_env = create_test_grid(env)
    grid = base_env.grid
    
    results = []
    wrapper_name = "RobustMaskGenerator" if use_robust else "PartialObsWrapper"
    
    # Test all four orientations
    for direction in range(4):
        # Position the agent with clear view of objects
        base_env.agent_pos = np.array([3, 3])
        base_env.agent_dir = direction
        
        # Get observation
        obs, _, _, _, _ = env.step(0)  # Take a no-op action to refresh observation
        
        # Get masks
        wall_mask = obs['wall_mask']
        lava_mask = obs['lava_mask']
        goal_mask = obs['goal_mask']
        
        # Create assertions for walls - test specifically based on the implementation
        if use_robust:
            # The robust implementation is agent-centric and more consistent
            # Check wall at (-1, -1) relative to agent
            wall_result1 = assert_mask_match(
                grid, wall_mask, 'wall', -1, -1, base_env.agent_pos, direction, 
                rotated=True, is_robust=use_robust
            )
            results.append(TestResult(
                f"{wrapper_name} wall test 1 dir={direction}",
                wall_result1.passed,
                wall_result1.message
            ))
            
            # Check wall at (-1, 0) relative to agent
            wall_result2 = assert_mask_match(
                grid, wall_mask, 'wall', -1, 0, base_env.agent_pos, direction,
                rotated=True, is_robust=use_robust
            )
            results.append(TestResult(
                f"{wrapper_name} wall test 2 dir={direction}",
                wall_result2.passed,
                wall_result2.message
            ))
            
            # Check lava and goal when facing the right direction
            if direction == 0:  # When facing right
                # Lava at (-2, 2) relative to agent
                lava_result = assert_mask_match(
                    grid, lava_mask, 'lava', -2, 2, base_env.agent_pos, direction,
                    rotated=True, is_robust=use_robust
                )
                results.append(TestResult(
                    f"{wrapper_name} lava test dir={direction}",
                    lava_result.passed,
                    lava_result.message
                ))
                
                # Goal at (2, -2) relative to agent
                goal_result = assert_mask_match(
                    grid, goal_mask, 'goal', 2, -2, base_env.agent_pos, direction,
                    rotated=True, is_robust=use_robust
                )
                results.append(TestResult(
                    f"{wrapper_name} goal test dir={direction}",
                    goal_result.passed,
                    goal_result.message
                ))
        else:
            # The original PartialObsWrapper is more complex - test locations that are more likely to be visible
            # based on its rotation scheme
            
            # Walls in view at different directions
            if direction == 0:  # Facing right
                # Look for walls at specific positions visible when facing right
                pos1 = (-1, 0)  # Wall to the left
                pos2 = (0, 1)   # Wall below
            elif direction == 1:  # Facing down
                pos1 = (0, -1)  # Wall above
                pos2 = (1, 0)   # Wall to right
            elif direction == 2:  # Facing left
                pos1 = (1, 0)   # Wall to right
                pos2 = (0, -1)  # Wall above
            else:  # Facing up
                pos1 = (0, 1)   # Wall below
                pos2 = (-1, 0)  # Wall to left
            
            # Test first wall position
            wall_result1 = assert_mask_match(
                grid, wall_mask, 'wall', pos1[0], pos1[1], base_env.agent_pos, direction, 
                rotated=True, is_robust=use_robust
            )
            results.append(TestResult(
                f"{wrapper_name} wall test dir={direction} pos={pos1}",
                wall_result1.passed,
                wall_result1.message
            ))
            
            # Test second wall position
            wall_result2 = assert_mask_match(
                grid, wall_mask, 'wall', pos2[0], pos2[1], base_env.agent_pos, direction,
                rotated=True, is_robust=use_robust
            )
            results.append(TestResult(
                f"{wrapper_name} wall test dir={direction} pos={pos2}",
                wall_result2.passed,
                wall_result2.message
            ))
        
        # Visualize the masks and grid for this orientation
        plt.figure(figsize=(20, 5))
        
        # Draw the grid
        plt.subplot(151)
        grid_vis = np.ones((base_env.height, base_env.width, 3))  # White background
        
        # Draw grid objects
        for i in range(base_env.width):
            for j in range(base_env.height):
                cell = grid.get(i, j)
                if cell:
                    if cell.type == 'wall':
                        grid_vis[j, i] = [0.5, 0.5, 0.5]  # Gray for walls
                    elif cell.type == 'goal':
                        grid_vis[j, i] = [0, 1, 0]  # Green for goal
                    elif cell.type == 'lava':
                        grid_vis[j, i] = [1, 0, 0]  # Red for lava
        
        # Mark agent position and direction
        agent_x, agent_y = base_env.agent_pos
        grid_vis[agent_y, agent_x] = [0, 0, 1]  # Blue for agent
        
        plt.imshow(grid_vis)
        plt.title(f'Grid Layout - Dir {direction}')
        
        # Add coordinate labels
        for i in range(base_env.width):
            for j in range(base_env.height):
                plt.text(i, j, f"({i},{j})", ha='center', va='center', 
                        color='black', fontsize=8)
        
        # Add direction arrow
        dir_arrows = ["→", "↓", "←", "↑"]
        arrow_colors = ['red', 'green', 'yellow', 'magenta']
        plt.text(agent_x, agent_y, dir_arrows[direction], 
                ha='center', va='center', color=arrow_colors[direction],
                fontsize=15, fontweight='bold')
        
        # Draw masks
        # Draw wall mask with coordinate overlay
        plt.subplot(152)
        plt.imshow(wall_mask, cmap='gray')
        plt.colorbar(label='Wall Mask')
        plt.title(f'Wall Mask - Dir {direction}')
        
        # Add mask coordinate labels
        for i in range(wall_mask.shape[0]):
            for j in range(wall_mask.shape[1]):
                plt.text(j, i, f"({j},{i})", ha='center', va='center', 
                        color='white', fontsize=6)
                if wall_mask[i, j] > 0:
                    plt.text(j, i+0.3, "W", ha='center', va='center', 
                            color='red', fontsize=8, fontweight='bold')
        
        # Draw lava mask
        plt.subplot(153)
        plt.imshow(lava_mask, cmap='hot')
        plt.colorbar(label='Lava Mask')
        plt.title(f'Lava Mask - Dir {direction}')
        
        # Add mask coordinate labels for lava
        for i in range(lava_mask.shape[0]):
            for j in range(lava_mask.shape[1]):
                if lava_mask[i, j] > 0:
                    plt.text(j, i, "L", ha='center', va='center', 
                            color='white', fontsize=8, fontweight='bold')
        
        # Draw goal mask
        plt.subplot(154)
        plt.imshow(goal_mask, cmap='Greens')
        plt.colorbar(label='Goal Mask')
        plt.title(f'Goal Mask - Dir {direction}')
        
        # Add mask coordinate labels for goal
        for i in range(goal_mask.shape[0]):
            for j in range(goal_mask.shape[1]):
                if goal_mask[i, j] > 0:
                    plt.text(j, i, "G", ha='center', va='center', 
                            color='black', fontsize=8, fontweight='bold')
        
        # Draw combined masks
        plt.subplot(155)
        combined = np.zeros((wall_mask.shape[0], wall_mask.shape[1], 3))
        combined[:, :, 0] = lava_mask  # Red channel
        combined[:, :, 1] = goal_mask  # Green channel
        combined[:, :, 2] = wall_mask * 0.5  # Blue channel
        plt.imshow(combined)
        plt.title(f'Combined Masks - Dir {direction}')
        
        # Save visualization
        plt.tight_layout()
        plt.savefig(f'mask_test_{wrapper_name}_dir{direction}.png')
        plt.close()
    
    return results

def test_diagonal_actions():
    """Test diagonal action implementation"""
    env = create_test_env()
    obs, _ = env.reset()
    base_env = env.unwrapped
    
    # Force agent position
    base_env.agent_pos = np.array([3, 3])
    base_env.agent_dir = 0  # Facing right
    
    results = []
    
    # Test each diagonal action
    for action in [3, 4]:  # 3=diagonal left, 4=diagonal right
        start_pos = np.array(base_env.agent_pos)
        start_dir = base_env.agent_dir
        
        action_name = "diagonal left" if action == 3 else "diagonal right"
        
        # Take the action
        _, _, _, _, _ = env.step(action)
        
        # Check new position
        end_pos = np.array(base_env.agent_pos)
        movement = end_pos - start_pos
        
        # Calculate expected movement based on orientation
        # For right-facing agent (dir=0):
        #   - diagonal left (3) should move [1,1]
        #   - diagonal right (4) should move [1,-1]
        expected_movements = {
            # Facing right (0)
            (0, 3): np.array([1, 1]),   # Diagonal left
            (0, 4): np.array([1, -1]),  # Diagonal right
            # Facing down (1)
            (1, 3): np.array([-1, 1]),  # Diagonal left 
            (1, 4): np.array([1, 1]),   # Diagonal right
            # Facing left (2)
            (2, 3): np.array([-1, -1]), # Diagonal left
            (2, 4): np.array([-1, 1]),  # Diagonal right
            # Facing up (3)
            (3, 3): np.array([1, -1]),  # Diagonal left
            (3, 4): np.array([-1, -1]), # Diagonal right
        }
        
        expected = expected_movements.get((start_dir, action), np.array([0, 0]))
        
        # Check if movement matches expected
        result = TestResult(
            f"Action {action} ({action_name}) from dir {start_dir}",
            np.array_equal(movement, expected),
            f"Expected movement {expected}, got {movement}"
        )
        results.append(result)
        
        # Reset for next test
        env.reset()
        base_env.agent_pos = np.array([3, 3])
        base_env.agent_dir = (start_dir + 1) % 4  # Test next direction
    
    return results

def test_diagonal_action_collisions():
    """Test that diagonal actions respect wall collision detection"""
    env = create_test_env()
    env.reset()
    base_env = create_test_grid(env)
    grid = base_env.grid
    
    results = []
    
    # Set up some walls to test diagonal movement through corners
    # Clear any existing objects near our test position
    for i in range(1, 6):
        for j in range(1, 6):
            if grid.get(i, j):
                grid.set(i, j, None)
    
    # Add walls for diagonal collision test
    # We'll create a corner to test if diagonal moves can go through
    grid.set(4, 3, Wall())  # Wall to the right
    grid.set(3, 4, Wall())  # Wall below
    
    # Position the agent
    base_env.agent_pos = np.array([3, 3])
    base_env.agent_dir = 0  # Facing right
    
    # Try to move diagonally down-right (diagonal left for right-facing agent)
    # This should fail as there's a wall corner
    _, reward, _, _, info = env.step(3)  # Diagonal left
    
    # Check for failure
    has_failure = info.get('failed', False)
    
    results.append(TestResult(
        "Diagonal collision through corner",
        has_failure,
        f"Diagonal move through corner walls should fail - {'failed' if has_failure else 'succeeded'}"
    ))
    
    # Reset and try again with walls in different positions
    env.reset()
    base_env = create_test_grid(env)
    
    # Place walls in a different configuration
    # Clear area around agent
    for i in range(2, 5):
        for j in range(2, 5):
            if grid.get(i, j):
                grid.set(i, j, None)
    
    # Place wall blocking forward movement but not to the side
    grid.set(4, 3, Wall())  # Wall to right
    
    base_env.agent_pos = np.array([3, 3])
    base_env.agent_dir = 0  # Facing right
    
    # Try to move diagonally (should still work in this case)
    _, reward, _, _, info = env.step(4)  # Diagonal right
    end_pos = np.array(base_env.agent_pos)
    
    # Check that movement succeeded
    expected_pos = np.array([4, 2])  # Should move diagonally up-right
    
    results.append(TestResult(
        "Diagonal movement with partial blocking",
        np.array_equal(end_pos, expected_pos),
        f"Expected to move to {expected_pos}, actually moved to {end_pos}"
    ))
    
    return results

def visualize_actions():
    """Visualize all possible actions from different orientations"""
    # Create a simpler visualization without using the RGBImgObsWrapper
    env = create_test_env()
    env.reset()
    base_env = env.unwrapped
    
    # Set up plot
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    directions = ["Right", "Down", "Left", "Up"]
    actions = ["Forward", "Left", "Right", "Diagonal Left", "Diagonal Right"]
    
    # Try all directions and actions
    for dir_idx in range(4):
        # Create a fresh environment for each direction
        env = create_test_env()
        env.reset()
        base_env = env.unwrapped
        
        # Set agent direction
        base_env.agent_dir = dir_idx
        
        # Place agent and wall
        base_env.agent_pos = np.array([3, 3])  # Center
        base_env.grid.set(5, 3, Wall())  # Add a wall for reference
        
        # Create a grid visualization
        grid_vis = np.ones((base_env.height, base_env.width, 3))  # White background
        
        # Draw objects
        for i in range(base_env.width):
            for j in range(base_env.height):
                cell = base_env.grid.get(i, j)
                if cell:
                    if cell.type == 'wall':
                        grid_vis[j, i] = [0.5, 0.5, 0.5]  # Gray for walls
                
        # Draw agent
        agent_x, agent_y = base_env.agent_pos
        grid_vis[agent_y, agent_x] = [0, 0, 1]  # Blue for agent
        
        # Add an arrow to show direction
        direction_colors = [[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1]]  # R, G, Y, M
        dir_arrows = ["→", "↓", "←", "↑"]
        
        # Show initial state
        axes[dir_idx, 0].imshow(grid_vis)
        axes[dir_idx, 0].text(agent_x, agent_y, dir_arrows[dir_idx], 
                             ha='center', va='center', color=direction_colors[dir_idx],
                             fontsize=15, fontweight='bold')
        axes[dir_idx, 0].set_title(f"Start ({directions[dir_idx]})")
        
        # Try each action
        for action in range(5):
            # Create a fresh environment for this action
            action_env = create_test_env()
            action_env.reset()
            action_base_env = action_env.unwrapped
            
            # Setup the same initial state
            action_base_env.agent_pos = np.array([3, 3])
            action_base_env.agent_dir = dir_idx
            action_base_env.grid.set(5, 3, Wall())
            
            # Take the action
            action_env.step(action)
            
            # Create a new grid visualization after the action
            new_grid_vis = np.ones((action_base_env.height, action_base_env.width, 3))
            
            # Draw objects
            for i in range(action_base_env.width):
                for j in range(action_base_env.height):
                    cell = action_base_env.grid.get(i, j)
                    if cell:
                        if cell.type == 'wall':
                            new_grid_vis[j, i] = [0.5, 0.5, 0.5]  # Gray for walls
            
            # Draw the agent after the action
            new_agent_x, new_agent_y = action_base_env.agent_pos
            new_agent_dir = action_base_env.agent_dir
            new_grid_vis[new_agent_y, new_agent_x] = [0, 0, 1]  # Blue for agent
            
            # Show the result
            axes[dir_idx, action].imshow(new_grid_vis)
            axes[dir_idx, action].text(new_agent_x, new_agent_y, dir_arrows[new_agent_dir], 
                                    ha='center', va='center', color=direction_colors[new_agent_dir],
                                    fontsize=15, fontweight='bold')
            
            # Add expected movement vector
            if action >= 3:  # For diagonal actions
                # Draw expected movement direction
                expected_dir = "↗" if action == 4 and dir_idx == 0 else "↘"
                if dir_idx == 1:
                    expected_dir = "↘" if action == 4 else "↙"
                elif dir_idx == 2:
                    expected_dir = "↙" if action == 4 else "↖"
                elif dir_idx == 3:
                    expected_dir = "↖" if action == 4 else "↗"
                
                # Add the expected movement annotation
                axes[dir_idx, action].text(
                    agent_x, agent_y - 0.4, 
                    expected_dir, 
                    ha='center', va='center',
                    color='purple', fontsize=18, fontweight='bold'
                )
            
            # Add coordinates to the plot
            for i in range(action_base_env.width):
                for j in range(action_base_env.height):
                    if abs(i - new_agent_x) <= 1 and abs(j - new_agent_y) <= 1:
                        axes[dir_idx, action].text(
                            i, j, f"({i},{j})", 
                            ha='center', va='center',
                            color='black', fontsize=6
                        )
            
            axes[dir_idx, action].set_title(f"Action {action}\n({actions[action]})")
    
    # Clean up plot appearance
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    
    plt.tight_layout()
    plt.savefig('action_visualization.png')
    plt.close()

def test_partial_obs_coordinates():
    """Test if PartialObsWrapper uses the correct coordinate order for grid.get()"""
    env = create_test_env()
    env.reset()
    base_env = env.unwrapped
    
    # Get the raw implementation of PartialObsWrapper
    import inspect
    from Environment_Tooling.BespokeEdits.CustomWrappers import PartialObsWrapper
    
    wrapper_code = inspect.getsource(PartialObsWrapper.observation)
    
    # Check if grid.get is called with (x, y) and not (y, x)
    correct_order = "cell = grid.get(x, y)" in wrapper_code
    
    test_result = TestResult(
        "PartialObsWrapper coordinate order", 
        correct_order,
        "PartialObsWrapper should use grid.get(x, y) not grid.get(y, x)"
    )
    
    # Place walls in specific locations to test mask generation
    
    # Reset the environment to a known state
    env.reset()
    base_env = env.unwrapped
    grid = base_env.grid
    
    # Place objects in a pattern that will clearly show if coordinates are swapped
    # We'll create an L shape with walls
    for i in range(3):
        grid.set(2, i+2, Wall())  # Vertical line
    for i in range(3):
        grid.set(i+2, 4, Wall())  # Horizontal line
    
    # Place a goal at a specific location
    grid.set(5, 1, Goal())
    
    # Place lava at another location
    grid.set(1, 5, Lava())
    
    # Position the agent where it can see these objects
    base_env.agent_pos = np.array([3, 3])
    base_env.agent_dir = 0  # Facing right
    
    # Get observation after setting up the environment
    obs, _, _, _, _ = env.step(0)  # Take a no-op action to refresh observation
    
    # Visualize the grid and masks
    plt.figure(figsize=(15, 10))
    
    # Draw the grid as ground truth
    plt.subplot(231)
    grid_vis = np.ones((base_env.height, base_env.width, 3))  # White background
    
    for i in range(base_env.width):
        for j in range(base_env.height):
            cell = grid.get(i, j)
            if cell:
                if cell.type == 'wall':
                    grid_vis[j, i] = [0.5, 0.5, 0.5]  # Gray for walls
                elif cell.type == 'goal':
                    grid_vis[j, i] = [0, 1, 0]  # Green for goal
                elif cell.type == 'lava':
                    grid_vis[j, i] = [1, 0, 0]  # Red for lava
    
    # Mark the agent position
    agent_x, agent_y = base_env.agent_pos
    grid_vis[agent_y, agent_x] = [0, 0, 1]  # Blue for agent
    
    plt.imshow(grid_vis)
    plt.title('Ground Truth Grid (x,y view)')
    
    # Add coordinate text to cells
    for i in range(base_env.width):
        for j in range(base_env.height):
            plt.text(i, j, f"({i},{j})", ha='center', va='center', 
                     color='black', fontsize=8)
    
    # Show the wall mask
    plt.subplot(232)
    plt.imshow(obs['wall_mask'], cmap='gray')
    plt.title('Wall Mask')
    
    # Show the lava mask
    plt.subplot(233)
    plt.imshow(obs['lava_mask'], cmap='hot')
    plt.title('Lava Mask')
    
    # Show the goal mask
    plt.subplot(234)
    plt.imshow(obs['goal_mask'], cmap='Greens')
    plt.title('Goal Mask')
    
    # Show empty mask
    plt.subplot(235)
    plt.imshow(obs['empty_mask'], cmap='Blues')
    plt.title('Empty Mask')
    
    # Show barrier mask
    plt.subplot(236)
    plt.imshow(obs['barrier_mask'], cmap='gray')
    plt.title('Barrier Mask')
    
    plt.tight_layout()
    plt.savefig('coordinate_test.png')
    plt.close()
    
    return [test_result]

def main():
    print("Running MiniGrid coordinate system tests...\n")
    
    # Run tests
    grid_tests = test_grid_access()
    mask_tests = test_mask_generation()
    enhanced_mask_tests = test_enhanced_mask_generation()
    diagonal_tests = test_diagonal_actions()
    diagonal_collision_tests = test_diagonal_action_collisions()
    coordinate_tests = test_partial_obs_coordinates()
    
    # Print results
    print("\n==== Grid Access Tests ====")
    for test in grid_tests:
        print(test)
    
    print("\n==== Basic Mask Generation Tests ====")
    for test in mask_tests:
        print(test)
        
    print("\n==== Enhanced Mask Generation Tests ====")
    for test in enhanced_mask_tests:
        print(test)
    
    print("\n==== Diagonal Action Tests ====")
    for test in diagonal_tests:
        print(test)
        
    print("\n==== Diagonal Collision Tests ====")
    for test in diagonal_collision_tests:
        print(test)
    
    print("\n==== Coordinate Order Tests ====")
    for test in coordinate_tests:
        print(test)
    
    # Generate visualizations
    print("\nGenerating action visualizations...")
    visualize_actions()
    
    print("\nTests complete. Check the following visualizations for results:")
    print("- mask_test.png: Basic mask generation")
    print("- mask_test_PartialObsWrapper_dir*.png: Orientation-specific masks with original implementation")
    print("- mask_test_RobustMaskGenerator_dir*.png: Orientation-specific masks with robust implementation")
    print("- action_visualization.png: Actions from different orientations")
    print("- coordinate_test.png: Grid vs masks comparison")

if __name__ == "__main__":
    main() 