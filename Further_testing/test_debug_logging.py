import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, NoDeath
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper

def test_debug_logging():
    """Test that the debug_logging flag controls detailed logging output."""
    print("\n=== Testing Debug Logging Flag ===")
    
    # Test with debug_logging=False
    print("\n# 1. With debug_logging=False:")
    env1 = gym.make('MiniGrid-Empty-8x8-v0')
    env1 = FullyObsWrapper(env1)
    env1 = LavaStepCounterWrapper(
        env1,
        lava_step_multiplier=5.0,
        verbose=True,
        debug_logging=False
    )
    
    # Reset and run a few steps
    obs, _ = env1.reset()
    print("\nRunning 3 steps...")
    for i in range(3):
        action = 1  # Turn right
        obs, reward, terminated, truncated, info = env1.step(action)
    
    print("\n# 2. With debug_logging=True:")
    env2 = gym.make('MiniGrid-Empty-8x8-v0')
    env2 = FullyObsWrapper(env2)
    env2 = LavaStepCounterWrapper(
        env2,
        lava_step_multiplier=5.0,
        verbose=True,
        debug_logging=True
    )
    
    # Reset and run a few steps
    obs, _ = env2.reset()
    print("\nRunning 3 steps...")
    for i in range(3):
        action = 1  # Turn right
        obs, reward, terminated, truncated, info = env2.step(action)
    
    # Clean up
    env1.close()
    env2.close()
    
    print("\n=== Test Complete ===")
    print("If the test worked correctly, you should see detailed step logging only for the second section.")

if __name__ == "__main__":
    test_debug_logging() 