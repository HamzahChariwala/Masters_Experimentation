import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, NoDeath
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper

def debug_effective_steps():
    """Debug the calculation of effective steps for lava penalty."""
    print("\n=== Debugging Effective Steps Calculation ===")
    
    # Create a simple environment
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = FullyObsWrapper(env)
    
    # Apply lava counter wrapper with high verbosity
    lava_counter_env = LavaStepCounterWrapper(
        env,
        lava_step_multiplier=5.0,  # Same as config
        verbose=True
    )
    
    # Apply reward wrapper
    reward_env = EpisodeCompletionRewardWrapper(
        lava_counter_env,
        reward_type='linear',
        x_intercept=100,
        y_intercept=1.0,
        count_lava_steps=True,
        verbose=True
    )
    
    # Reset to initialize the environments
    obs, _ = reward_env.reset()
    
    # Now check step counts
    print(f"Step counts after reset:")
    print(f"  Base env: {env.step_count}")
    print(f"  Lava counter env.env: {lava_counter_env.env.step_count}")
    print(f"  Lava counter env own: {lava_counter_env.step_count if hasattr(lava_counter_env, 'step_count') else 'N/A'}")
    print(f"  Reward env: {reward_env.step_count if hasattr(reward_env, 'step_count') else 'N/A'}")
    
    # Manually set the lava steps to 2 for testing
    print(f"\n==== Manual debug mode ====")
    print(f"Setting lava steps to 2")
    lava_counter_env.lava_steps = 2
    print(f"Current step counts:")
    print(f"  Base env: {env.step_count}")
    print(f"  Lava counter env.env: {lava_counter_env.env.step_count}")
    
    # Calculate expected effective steps
    multiplier = lava_counter_env.lava_step_multiplier
    actual_steps = lava_counter_env.env.step_count
    lava_steps = lava_counter_env.lava_steps
    
    expected_effective = actual_steps + (lava_steps * (multiplier - 1))
    actual_effective = lava_counter_env.get_effective_steps()
    
    # Print diagnostic information
    print(f"\nDiagnostic Information:")
    print(f"  Actual steps: {actual_steps}")
    print(f"  Lava steps: {lava_steps}")
    print(f"  Lava step multiplier: {multiplier}")
    print(f"  Expected effective steps: {expected_effective}")
    print(f"  Actual effective steps from method: {actual_effective}")
    
    # Now take a step and check values
    print(f"\nTaking a step...")
    action = 1  # Turn right (arbitrary action)
    obs, reward, terminated, truncated, info = reward_env.step(action)
    
    # Check updated step counts
    print(f"Updated step counts:")
    print(f"  Base env: {env.step_count}")
    print(f"  Lava counter env.env: {lava_counter_env.env.step_count}")
    
    # Recalculate 
    actual_steps = lava_counter_env.env.step_count
    lava_steps = lava_counter_env.lava_steps
    expected_effective = actual_steps + (lava_steps * (multiplier - 1))
    actual_effective = info.get('effective_steps', -1)
    
    print(f"\nAfter taking a step:")
    print(f"  Step count: {actual_steps}")
    print(f"  Lava steps: {lava_steps}")
    print(f"  Expected effective steps: {expected_effective}")
    print(f"  Actual effective steps from info: {actual_effective}")
    
    # Force a successful episode termination
    print("\nSimulating episode completion (success):")
    # To simulate, we would make a copy of the step method but set appropriate values
    print("  Calculating expected reward for this situation:")
    
    # Calculate reward
    slope = reward_env.y_intercept / reward_env.x_intercept
    expected_reward = max(0, reward_env.y_intercept - slope * expected_effective)
    
    print(f"  Expected reward: {expected_reward}")
    
    # Let's also look at what happens if we intentionally set a high step count
    print(f"\n==== Testing with manually set high step count ====")
    # Create a new env
    env3 = gym.make('MiniGrid-Empty-8x8-v0')
    env3 = FullyObsWrapper(env3)
    
    lava_counter_env3 = LavaStepCounterWrapper(
        env3,
        lava_step_multiplier=5.0,
        verbose=True
    )
    
    reward_env3 = EpisodeCompletionRewardWrapper(
        lava_counter_env3,
        reward_type='linear',
        x_intercept=100,
        y_intercept=1.0,
        count_lava_steps=True,
        verbose=True
    )
    
    obs, _ = reward_env3.reset()
    
    # Let's set a high step count of 125
    print(f"Setting step count to 125")
    env3.step_count = 125
    
    # And set lava steps to 2 (same as in your output)
    print(f"Setting lava steps to 2")
    lava_counter_env3.lava_steps = 2
    
    # Calculate effective steps with these values
    multiplier = lava_counter_env3.lava_step_multiplier
    actual_steps = env3.step_count
    lava_steps = lava_counter_env3.lava_steps
    
    expected_effective = actual_steps + (lava_steps * (multiplier - 1))
    actual_effective = lava_counter_env3.get_effective_steps()
    
    print(f"\nWith high step count:")
    print(f"  Actual steps: {actual_steps}")
    print(f"  Lava steps: {lava_steps}")
    print(f"  Lava step multiplier: {multiplier}")
    print(f"  Expected effective steps: {expected_effective}")
    print(f"  Actual effective steps from method: {actual_effective}")
    
    # Now test with 112 regular steps and 113 lava steps
    print(f"\n==== Testing with 112 regular steps and 113 lava steps ====")
    # Create a new env
    env2 = gym.make('MiniGrid-Empty-8x8-v0')
    env2 = FullyObsWrapper(env2)
    
    lava_counter_env2 = LavaStepCounterWrapper(
        env2,
        lava_step_multiplier=5.0,
        verbose=True
    )
    
    reward_env2 = EpisodeCompletionRewardWrapper(
        lava_counter_env2,
        reward_type='linear',
        x_intercept=100,
        y_intercept=1.0,
        count_lava_steps=True,
        verbose=True
    )
    
    obs, _ = reward_env2.reset()
    
    # Manually set the values
    env2.step_count = 112
    lava_counter_env2.lava_steps = 113
    
    # Calculate expected effective steps
    multiplier = lava_counter_env2.lava_step_multiplier
    actual_steps = env2.step_count
    lava_steps = lava_counter_env2.lava_steps
    
    expected_effective = actual_steps + (lava_steps * (multiplier - 1))
    actual_effective = lava_counter_env2.get_effective_steps()
    
    # Print diagnostic information
    print(f"\nDiagnostic Information for test case 2:")
    print(f"  Actual steps: {actual_steps}")
    print(f"  Lava steps: {lava_steps}")
    print(f"  Lava step multiplier: {multiplier}")
    print(f"  Formula: {actual_steps} + ({lava_steps} * ({multiplier} - 1))")
    print(f"  Calculation: {actual_steps} + ({lava_steps} * {multiplier - 1})")
    print(f"  Calculation: {actual_steps} + {lava_steps * (multiplier - 1)}")
    print(f"  Expected effective steps: {expected_effective}")
    print(f"  Actual effective steps from method: {actual_effective}")
    
    # Clean up
    reward_env.close()
    reward_env2.close()
    reward_env3.close()
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    debug_effective_steps() 