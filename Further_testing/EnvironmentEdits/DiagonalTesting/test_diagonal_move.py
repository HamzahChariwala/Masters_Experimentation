import numpy as np
import gymnasium as gym
import time
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.BespokeEdits.CustomWrappers import DiagonalMoveMonitor
from minigrid.wrappers import FullyObsWrapper

def test_diagonal_movement():
    """
    Test that diagonal movement properly respects the agent's orientation
    by stepping through all possible directions and trying diagonal moves.
    """
    # Create environment
    env = gym.make('MiniGrid-Empty-8x8-v0', render_mode="human")
    env = FullyObsWrapper(env)
    env = CustomActionWrapper(env)
    env = DiagonalMoveMonitor(env)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
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
    
    # Wait for user to see initial state
    print(f"\nInitial position: {env.agent_pos}, direction: {dir_names[env.agent_dir]}")
    time.sleep(1)
    
    # For each direction, try both diagonal moves
    for i in range(4):  # Test all 4 directions
        # Turn to face the correct direction
        while env.agent_dir != i:
            print(f"Turning right to face {dir_names[i]}")
            obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
            time.sleep(0.5)
        
        print(f"\nTesting diagonal moves when facing {dir_names[env.agent_dir]}")
        
        # Try diagonal left
        print("Trying diagonal left")
        old_pos = np.array(env.agent_pos)  # Convert to numpy array
        obs, reward, terminated, truncated, info = env.step(actions["diagonal_left"])
        print(f"  Old position: {old_pos}, New position: {env.agent_pos}")
        print(f"  Moved by vector: {np.array(env.agent_pos) - old_pos}")
        if "action" in info and info["action"] == "diagonal":
            if "failed" in info and info["failed"]:
                print("  Diagonal move FAILED")
            else:
                print("  Diagonal move SUCCESS")
        time.sleep(1)
        
        # Reset position (simplest way is to reset the environment)
        obs, info = env.reset(seed=42)
        
        # Get back to the same orientation
        while env.agent_dir != i:
            obs, reward, terminated, truncated, info = env.step(actions["turn_right"])
            time.sleep(0.1)
            
        # Try diagonal right
        print("Trying diagonal right")
        old_pos = np.array(env.agent_pos)  # Convert to numpy array
        obs, reward, terminated, truncated, info = env.step(actions["diagonal_right"])
        print(f"  Old position: {old_pos}, New position: {env.agent_pos}")
        print(f"  Moved by vector: {np.array(env.agent_pos) - old_pos}")
        if "action" in info and info["action"] == "diagonal":
            if "failed" in info and info["failed"]:
                print("  Diagonal move FAILED")
            else:
                print("  Diagonal move SUCCESS")
        time.sleep(1)
        
        # Reset for next direction test
        obs, info = env.reset(seed=42)
        print("\n-------------------")
    
    print("\nTest complete!")
    env.close()

if __name__ == "__main__":
    test_diagonal_movement() 