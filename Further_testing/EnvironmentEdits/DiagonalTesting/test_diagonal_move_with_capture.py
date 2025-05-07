import numpy as np
import gymnasium as gym
import time
import os
import matplotlib.pyplot as plt
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.BespokeEdits.CustomWrappers import DiagonalMoveMonitor
from minigrid.wrappers import FullyObsWrapper

def test_diagonal_movement_with_visualization():
    """
    Test diagonal movement with path visualization to verify the actual movement pattern.
    """
    # Create output directory for visualizations
    os.makedirs("diagonal_test_results", exist_ok=True)
    
    # Create environment
    env = gym.make('MiniGrid-Empty-8x8-v0', render_mode="human")
    env = FullyObsWrapper(env)
    env = CustomActionWrapper(env)
    env = DiagonalMoveMonitor(env)
    
    # Dictionary to map direction indices to names
    dir_names = {
        0: "Right",
        1: "Down",
        2: "Left",
        3: "Up"
    }
    
    # Action mappings
    actions = {
        "turn_left": 0,
        "turn_right": 1,
        "forward": 2,
        "diagonal_left": 3,
        "diagonal_right": 4
    }
    
    # Test each diagonal direction individually with path tracking
    for direction_name, direction in [("right", 0), ("down", 1), ("left", 2), ("up", 3)]:
        print(f"\n\n=== Testing diagonal moves when facing {direction_name} ===")
        
        # Dictionary to track paths
        paths = {
            "diagonal_left": [],
            "diagonal_right": []
        }
        
        for move_type in ["diagonal_left", "diagonal_right"]:
            # Reset environment
            obs, info = env.reset(seed=42)
            
            # Track positions
            positions = [np.array(env.agent_pos)]
            
            # Turn to face the correct direction
            while env.agent_dir != direction:
                print(f"Turning right to face {dir_names[direction]}")
                obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
                time.sleep(0.3)
            
            # Make a few steps forward first to clear the starting position
            for _ in range(2):
                obs, reward, terminated, truncated, info = env.step(actions["forward"])
                positions.append(np.array(env.agent_pos))
                time.sleep(0.3)
            
            # Now try diagonal moves multiple times
            print(f"\nTesting {move_type} movement when facing {dir_names[direction]}")
            for i in range(3):  # Try 3 diagonal moves
                old_pos = np.array(env.agent_pos)
                print(f"Attempt {i+1}: Moving {move_type}")
                obs, reward, terminated, truncated, info = env.step(actions[move_type])
                positions.append(np.array(env.agent_pos))
                
                print(f"  From {old_pos} to {env.agent_pos}, delta={np.array(env.agent_pos) - old_pos}")
                if "action" in info and info["action"] == "diagonal":
                    result = "FAILED" if info.get("failed", False) else "SUCCESS"
                    print(f"  Diagonal move {result}")
                
                time.sleep(0.5)
            
            # Store the full path
            paths[move_type] = positions
            
            # Visualize and save the path
            plt.figure(figsize=(8, 8))
            plt.title(f"Path when facing {direction_name} and using {move_type}")
            
            # Plot empty grid
            plt.grid(True)
            plt.xlim(-0.5, 8.5)
            plt.ylim(-0.5, 8.5)
            
            # Plot path
            pos_array = np.array(positions)
            plt.plot(pos_array[:, 0], pos_array[:, 1], 'b-', linewidth=2)
            plt.plot(pos_array[:, 0], pos_array[:, 1], 'ro', markersize=8)
            
            # Add labels for each point
            for i, pos in enumerate(positions):
                plt.text(pos[0]+0.1, pos[1]+0.1, str(i), fontsize=12)
            
            # Save figure
            filename = f"diagonal_test_results/{direction_name}_{move_type}.png"
            plt.savefig(filename)
            print(f"Saved path visualization to {filename}")
            plt.close()
        
        # Create combined visualization for this direction
        plt.figure(figsize=(10, 10))
        plt.title(f"Movement paths when facing {direction_name}")
        
        # Plot empty grid
        plt.grid(True)
        plt.xlim(-0.5, 8.5)
        plt.ylim(-0.5, 8.5)
        
        # Plot both paths
        colors = {"diagonal_left": "blue", "diagonal_right": "green"}
        markers = {"diagonal_left": "o", "diagonal_right": "s"}
        
        for move_type, positions in paths.items():
            pos_array = np.array(positions)
            plt.plot(pos_array[:, 0], pos_array[:, 1], f'-', color=colors[move_type], linewidth=2, 
                    label=f"{move_type} path")
            plt.plot(pos_array[:, 0], pos_array[:, 1], marker=markers[move_type], color=colors[move_type], 
                    markersize=8)
            
            # Add labels for each point
            for i, pos in enumerate(positions):
                plt.text(pos[0]+0.1, pos[1]+0.1, f"{i}", fontsize=10, color=colors[move_type])
        
        plt.legend()
        # Save combined figure
        combined_filename = f"diagonal_test_results/{direction_name}_combined.png"
        plt.savefig(combined_filename)
        print(f"Saved combined visualization to {combined_filename}")
        plt.close()
    
    print("\nTest complete! All visualizations saved to diagonal_test_results/")
    env.close()

if __name__ == "__main__":
    test_diagonal_movement_with_visualization() 