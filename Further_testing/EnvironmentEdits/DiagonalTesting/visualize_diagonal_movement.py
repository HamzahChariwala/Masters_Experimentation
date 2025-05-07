import numpy as np
import gymnasium as gym
import os
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
from PIL import Image

# Add the parent directory to the system path for imports
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.BespokeEdits.CustomWrappers import DiagonalMoveMonitor
from minigrid.wrappers import FullyObsWrapper

def visualize_diagonal_movement():
    """
    Visual test of diagonal movement that captures frames
    to see how diagonal moves are actually rendered.
    """
    print("Creating environment for diagonal movement visualization...")
    
    # Create output directory
    output_dir = "diagonal_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Action mappings
    actions = {
        "turn_left": 0,
        "turn_right": 1,
        "forward": 2,
        "diagonal_left": 3,
        "diagonal_right": 4
    }
    
    # Direction names
    dir_names = {
        0: "Right",
        1: "Down",
        2: "Left",
        3: "Up"
    }
    
    # Enable rendering to capture frames
    env = gym.make('MiniGrid-Empty-16x16-v0', render_mode="rgb_array")
    env = CustomActionWrapper(env)
    env = DiagonalMoveMonitor(env)
    env = FullyObsWrapper(env)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    # Functions to capture and save frames
    def save_frame(frame, step, action):
        plt.figure(figsize=(8, 8))
        plt.imshow(frame)
        plt.title(f"Step {step}: {action}\nPosition: {env.agent_pos}, Direction: {dir_names[env.agent_dir]}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/step_{step:02d}_{action}.png", dpi=100)
        plt.close()
        return frame
    
    # Capture the initial state
    frame = env.render()
    save_frame(frame, 0, "initial")
    frames = [frame]
    
    # Move agent to center for better visualization
    print("Moving agent to center...")
    move_sequence = []
    
    # First make sure we're facing right
    while env.agent_dir != 0:  # 0 = Right
        move_sequence.append({"step": len(move_sequence) + 1, "action": "turn_right"})
        obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
        frame = env.render()
        save_frame(frame, len(move_sequence), "turn_right")
        frames.append(frame)
        time.sleep(0.2)
    
    # Move forward to center
    for i in range(6):
        move_sequence.append({"step": len(move_sequence) + 1, "action": "forward"})
        obs, reward, terminated, truncated, info = env.step(actions["forward"])
        frame = env.render()
        save_frame(frame, len(move_sequence), "forward")
        frames.append(frame)
        time.sleep(0.2)
    
    # Test diagonal moves from all four directions
    test_directions = [
        {"name": "Right", "dir": 0},
        {"name": "Down", "dir": 1},
        {"name": "Left", "dir": 2},
        {"name": "Up", "dir": 3}
    ]
    
    # For each direction, test both diagonal moves
    for direction in test_directions:
        # Turn to face the direction
        while env.agent_dir != direction["dir"]:
            move_sequence.append({"step": len(move_sequence) + 1, "action": "turn_right"})
            obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
            frame = env.render()
            save_frame(frame, len(move_sequence), f"turn_to_{direction['name']}")
            frames.append(frame)
            time.sleep(0.2)
        
        print(f"\nTesting diagonal moves when facing {direction['name']}")
        current_pos = list(map(int, env.agent_pos))
        print(f"Current position: {current_pos}, Direction: {dir_names[env.agent_dir]}")
        
        # Test diagonal_left
        move_sequence.append({"step": len(move_sequence) + 1, "action": f"diagonal_left_{direction['name']}"})
        obs, reward, terminated, truncated, info = env.step(actions["diagonal_left"])
        
        # Capture frames every step during the diagonal move
        frame = env.render()
        save_frame(frame, len(move_sequence), f"diagonal_left_{direction['name']}")
        frames.append(frame)
        
        new_pos = list(map(int, env.agent_pos))
        print(f"After diagonal_left: {new_pos}, Delta: {[new_pos[0]-current_pos[0], new_pos[1]-current_pos[1]]}")
        
        time.sleep(0.5)
        
        # Reset position for diagonal_right test
        obs, info = env.reset(seed=42)
        
        # Return to previous position and direction
        while env.agent_dir != 0:  # Reset to face right
            obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
            time.sleep(0.1)
        
        for i in range(6):
            obs, reward, terminated, truncated, info = env.step(actions["forward"])
            time.sleep(0.1)
        
        # Turn to face the test direction again
        while env.agent_dir != direction["dir"]:
            obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
            time.sleep(0.1)
        
        current_pos = list(map(int, env.agent_pos))
        
        # Test diagonal_right
        move_sequence.append({"step": len(move_sequence) + 1, "action": f"diagonal_right_{direction['name']}"})
        obs, reward, terminated, truncated, info = env.step(actions["diagonal_right"])
        
        frame = env.render()
        save_frame(frame, len(move_sequence), f"diagonal_right_{direction['name']}")
        frames.append(frame)
        
        new_pos = list(map(int, env.agent_pos))
        print(f"After diagonal_right: {new_pos}, Delta: {[new_pos[0]-current_pos[0], new_pos[1]-current_pos[1]]}")
        
        time.sleep(0.5)
        
        # Reset for next direction test
        if direction != test_directions[-1]:
            obs, info = env.reset(seed=42)
            
            # Return to previous position
            while env.agent_dir != 0:
                obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
                time.sleep(0.1)
            
            for i in range(6):
                obs, reward, terminated, truncated, info = env.step(actions["forward"])
                time.sleep(0.1)
    
    # Create a comparison of standard movement vs diagonal
    print("\nCreating standard move vs diagonal move comparison...")
    
    # Reset for comparison
    obs, info = env.reset(seed=42)
    frames_comparison = []
    
    # Move to center
    while env.agent_dir != 0:
        obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
        time.sleep(0.1)
    
    for i in range(6):
        obs, reward, terminated, truncated, info = env.step(actions["forward"])
        time.sleep(0.1)
    
    # Start comparison: same starting position
    starting_pos = list(map(int, env.agent_pos))
    frame = env.render()
    save_frame(frame, 0, "comparison_start")
    frames_comparison.append(frame)
    
    # First case: Diagonal move
    print("Capturing diagonal movement...")
    obs, reward, terminated, truncated, info = env.step(actions["diagonal_right"])
    frame = env.render()
    save_frame(frame, 1, "comparison_diagonal")
    frames_comparison.append(frame)
    
    diagonal_pos = list(map(int, env.agent_pos))
    
    # Reset to starting position
    obs, info = env.reset(seed=42)
    while env.agent_dir != 0:
        obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
        time.sleep(0.1)
    
    for i in range(6):
        obs, reward, terminated, truncated, info = env.step(actions["forward"])
        time.sleep(0.1)
    
    # Second case: forward + turn + forward
    print("Capturing traditional movement sequence...")
    obs, reward, terminated, truncated, info = env.step(actions["forward"])
    frame = env.render()
    save_frame(frame, 2, "comparison_forward")
    frames_comparison.append(frame)
    
    obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
    frame = env.render()
    save_frame(frame, 3, "comparison_turn")
    frames_comparison.append(frame)
    
    obs, reward, terminated, truncated, info = env.step(actions["forward"])
    frame = env.render()
    save_frame(frame, 4, "comparison_forward2")
    frames_comparison.append(frame)
    
    traditional_pos = list(map(int, env.agent_pos))
    
    print(f"Starting position: {starting_pos}")
    print(f"After diagonal: {diagonal_pos}")
    print(f"After traditional sequence: {traditional_pos}")
    
    # Create GIFs
    print("Creating GIFs of the movements...")
    
    # Main diagonal tests GIF
    imageio.mimsave(f'{output_dir}/diagonal_movement_test.gif', 
                   [np.array(frame) for frame in frames],
                   fps=2)
    
    # Comparison GIF
    imageio.mimsave(f'{output_dir}/diagonal_vs_traditional.gif', 
                   [np.array(frame) for frame in frames_comparison],
                   fps=1)
    
    print(f"\nVisualization complete! Images and GIFs saved to {output_dir}/")
    env.close()

if __name__ == "__main__":
    visualize_diagonal_movement() 