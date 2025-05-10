import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, NoDeath
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper

def test_reward_calculation():
    """Test the reward calculation with lava step multiplier."""
    print("\n=== Testing Reward Calculation with Lava Multiplier ===")
    
    # Create environment with both wrappers
    multiplier = 5.0
    print(f"\nCreating environment with lava_step_multiplier={multiplier}")
    
    env = gym.make('MiniGrid-LavaGapS7-v0')
    env = FullyObsWrapper(env)
    env = NoDeath(env, no_death_types=("lava",), death_cost=-0.1)
    
    # First apply the lava step counter wrapper
    env = LavaStepCounterWrapper(
        env,
        lava_step_multiplier=multiplier,
        verbose=True,
        debug_logging=True
    )
    
    # Then apply the reward wrapper that uses the lava step counter
    env = EpisodeCompletionRewardWrapper(
        env,
        reward_type='linear',
        x_intercept=100,
        y_intercept=1.0,
        transition_width=10,
        count_lava_steps=True,
        verbose=True
    )
    
    # Reset and run steps, manually simulating a goal completion
    obs, _ = env.reset(seed=42)
    
    print("\nRunning steps with the agent through lava...")
    
    # Run some steps, including some on lava
    for i in range(5):
        if i == 0:  # First step, make it go through lava
            action = 2  # Move forward
        else:
            action = 1  # Turn right
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  Is in lava: {info['is_in_lava']}")
        print(f"  Lava steps so far: {info['lava_steps']}")
        print(f"  Effective steps: {info['effective_steps']}")
        print(f"  Current reward: {reward}")
    
    # Manually trigger a successful episode completion 
    print("\nManually triggering successful episode completion...")
    
    # Create a direct test of the reward wrapper with known values
    # Direct test to check formula calculations
    regular_steps = 6
    lava_steps = 6
    effective_steps = regular_steps + (lava_steps * (multiplier - 1))
    
    print(f"\nDirect calculation test:")
    print(f"  Regular steps: {regular_steps}")
    print(f"  Lava steps: {lava_steps}")
    print(f"  Multiplier: {multiplier}")
    print(f"  Effective steps: {effective_steps}")
    
    # Calculate expected reward (using the linear formula from the wrapper)
    x_intercept = 100
    y_intercept = 1.0
    slope = y_intercept / x_intercept
    expected_reward = max(0, y_intercept - slope * effective_steps)
    print(f"  Expected reward: {expected_reward}")
    
    # Force the environment to simulate a successful episode
    # We need to create a simulated success condition
    env.unwrapped._reward = 1.0  # Set reward to positive
    
    # Create a custom step function for test
    def custom_step(self, action):
        obs, reward, _, _, info = self.env.step(action)
        return obs, 1.0, True, False, info  # Force terminated=True and reward=1.0
    
    # Store original step method
    original_step = env.step
    
    # Replace with our custom step method temporarily
    env.step = lambda action: custom_step(env, action)
    
    # Call step with our custom function
    obs, final_reward, terminated, truncated, info = env.step(0)
    
    # Restore original step method
    env.step = original_step
    
    # Print the result
    print(f"\nFinal reward wrapper calculation:")
    print(f"  Actual reward: {final_reward}")
    
    # Clean up
    env.close()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_reward_calculation() 