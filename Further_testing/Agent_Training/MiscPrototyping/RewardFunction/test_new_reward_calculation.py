import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, NoDeath
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper
import numpy as np

def test_new_reward_calculation():
    """Test the reward calculation with the new lava step multiplier formula."""
    print("\n=== Testing Reward Calculation with New Lava Multiplier Formula ===")
    
    # Create a simple environment
    multiplier = 5.0
    print(f"\nCreating environment with lava_step_multiplier={multiplier}")
    
    env = gym.make('MiniGrid-Empty-8x8-v0')
    
    # First apply the lava step counter wrapper
    env = LavaStepCounterWrapper(
        env,
        lava_step_multiplier=multiplier,
        verbose=True,
        debug_logging=True
    )
    
    # Then apply the reward wrapper
    env = EpisodeCompletionRewardWrapper(
        env,
        reward_type='linear',
        x_intercept=100,
        y_intercept=1.0,
        transition_width=10,
        count_lava_steps=True,
        verbose=True
    )
    
    # Example scenario: 1 regular step + 2 lava steps with multiplier 5
    regular_steps = 1
    lava_steps = 2
    
    # Calculate expected effective steps using the new formula
    expected_effective_steps = regular_steps + (lava_steps * multiplier)
    
    print(f"\nExample scenario:")
    print(f"  Regular steps: {regular_steps}")
    print(f"  Lava steps: {lava_steps}")
    print(f"  Lava multiplier: {multiplier}")
    print(f"  Expected effective steps: {regular_steps} + ({lava_steps} * {multiplier}) = {expected_effective_steps}")
    
    # Calculate expected reward using the linear formula
    x_intercept = 100
    y_intercept = 1.0
    slope = y_intercept / x_intercept
    expected_reward = max(0, y_intercept - slope * expected_effective_steps)
    
    print(f"  Expected reward (linear formula): {expected_reward}")
    print(f"  Calculation: max(0, {y_intercept} - ({y_intercept}/{x_intercept} * {expected_effective_steps}))")
    
    # Now create a test with the actual wrapper
    # We'll need to simulate a successful episode with our example values
    
    # Create a mock info dictionary with our example values
    # Patch the _get_effective_steps method to return our expected value
    original_get_effective_steps = env._get_effective_steps
    env._get_effective_steps = lambda info: expected_effective_steps
    
    # Store the original step method
    original_step = env.env.step
    
    # Create a mock step function that returns success
    def mock_success_step(action):
        mock_info = {
            'lava_steps': lava_steps,
            'effective_steps': expected_effective_steps
        }
        return None, 1.0, True, False, mock_info
    
    # Replace the step function with our mock
    env.env.step = mock_success_step
    
    # Reset step counter (not necessary for the test but matches real usage)
    env.step_count = regular_steps
    
    # Call step to trigger reward calculation
    _, actual_reward, _, _, _ = env.step(0)
    
    # Print the results
    print(f"\nTest results:")
    print(f"  Actual reward from wrapper: {actual_reward}")
    
    # Compare expected and actual rewards
    if abs(actual_reward - expected_reward) < 0.0001:
        print(f"  ✓ Reward calculation is correct!")
    else:
        print(f"  ❌ Reward calculation mismatch!")
        print(f"    Expected: {expected_reward}")
        print(f"    Actual: {actual_reward}")
        print(f"    Difference: {abs(actual_reward - expected_reward)}")
    
    # Restore the original methods
    env._get_effective_steps = original_get_effective_steps
    env.env.step = original_step
    
    # Clean up
    env.close()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_new_reward_calculation() 