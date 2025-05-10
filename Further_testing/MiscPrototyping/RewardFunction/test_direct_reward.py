import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, NoDeath
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

def test_reward_directly():
    """Test the reward formula calculation directly with controlled inputs."""
    print("\n=== Testing Direct Reward Calculation ===")
    
    # Create a simple environment
    env = gym.make('MiniGrid-Empty-8x8-v0')
    
    # Apply the reward wrapper (we'll override its methods for testing)
    env = EpisodeCompletionRewardWrapper(
        env,
        reward_type='linear',
        x_intercept=100,
        y_intercept=1.0,
        transition_width=10,
        count_lava_steps=True,
        verbose=True
    )
    
    # Test scenario: regular steps + lava steps
    # 1 regular step + 2 lava steps with multiplier 5
    # Expected effective steps = 1 + (2 * (5-1)) = 9
    
    regular_steps = 1
    lava_steps = 2
    multiplier = 5
    expected_effective_steps = regular_steps + (lava_steps * (multiplier - 1))
    print(f"\nTest scenario:")
    print(f"  Regular steps: {regular_steps}")
    print(f"  Lava steps: {lava_steps}")
    print(f"  Lava multiplier: {multiplier}")
    print(f"  Expected effective steps: {expected_effective_steps}")
    
    # Create a direct test function to calculate reward with our parameters
    def calculate_reward(effective_steps, reward_type='linear', x_intercept=100, y_intercept=1.0, transition_width=10):
        if reward_type == 'linear':
            slope = y_intercept / x_intercept
            return max(0, y_intercept - slope * effective_steps)
        elif reward_type == 'exponential':
            slope = -np.log(0.01) / x_intercept
            return y_intercept / np.exp(slope * effective_steps)
        elif reward_type == 'sigmoid':
            slope = 4 / transition_width
            shift = x_intercept / 2
            return y_intercept / (1 + np.exp(slope * (effective_steps - shift)))
    
    # Calculate expected reward with our formula
    expected_reward = calculate_reward(expected_effective_steps)
    print(f"  Expected reward (linear): {expected_reward}")
    
    # Now let's override _get_effective_steps in the wrapper to return our expected value
    original_get_effective_steps = env._get_effective_steps
    
    # Create a mock info dictionary
    mock_info = {'effective_steps': expected_effective_steps, 'lava_steps': lava_steps}
    
    # Override the method
    env._get_effective_steps = lambda info: expected_effective_steps
    
    # Create a step simulation that returns a positive reward and terminated=True
    original_step = env.env.step
    
    # Override env.env.step to return success
    def mock_success_step(action):
        return None, 1.0, True, False, mock_info
    
    env.env.step = mock_success_step
    
    # Call step with our overrides
    _, actual_reward, _, _, _ = env.step(0)
    
    # Print the result
    print(f"  Actual reward from wrapper: {actual_reward}")
    
    # Check if they match
    if abs(actual_reward - expected_reward) < 0.0001:
        print(f"  ✓ Reward calculation is correct")
    else:
        print(f"  ❌ Reward calculation mismatch!")
        print(f"    Expected: {expected_reward}")
        print(f"    Actual: {actual_reward}")
        print(f"    Difference: {abs(actual_reward - expected_reward)}")
    
    # Restore original methods
    env._get_effective_steps = original_get_effective_steps
    env.env.step = original_step
    
    # Close the environment
    env.close()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_reward_directly() 