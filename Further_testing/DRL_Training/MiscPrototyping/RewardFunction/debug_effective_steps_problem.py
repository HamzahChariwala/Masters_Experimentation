import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, NoDeath
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper

def inspect_wrapper_attributes(env):
    """Inspect attributes of wrapper and its wrapped environments"""
    print("\n=== Inspecting Environment Wrapper Attributes ===")
    
    # Start with the outer wrapper
    current_env = env
    level = 0
    
    while hasattr(current_env, 'env'):
        print(f"Level {level} - Type: {type(current_env).__name__}")
        
        # Look for step_count and other relevant attributes
        if hasattr(current_env, 'step_count'):
            print(f"  step_count: {current_env.step_count}")
        
        if hasattr(current_env, 'lava_steps'):
            print(f"  lava_steps: {current_env.lava_steps}")
            
        # Get wrapped env
        level += 1
        current_env = current_env.env
    
    # Print info about base environment
    print(f"Base Env - Type: {type(current_env).__name__}")
    if hasattr(current_env, 'step_count'):
        print(f"  step_count: {current_env.step_count}")
    
    print("=== End Inspection ===\n")

def debug_high_effective_steps():
    """Debug why effective steps is showing a very high value (504) despite only having 1 step and 2 lava steps."""
    print("\n=== Debugging High Effective Steps Issue ===")
    
    # Create a simple environment
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = FullyObsWrapper(env)
    
    # Apply lava counter wrapper
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
    
    # Reset the environment
    obs, _ = reward_env.reset()
    
    # Inspect wrapper attributes
    inspect_wrapper_attributes(reward_env)
    
    # Try artificially setting some values
    print("\n=== Setting various step count combinations ===")
    
    # Make a few combinations to identify the problematic one
    test_cases = [
        {"step_count": 1, "lava_steps": 2, "desc": "Basic case (should be 9 effective steps)"},
        {"step_count": 1, "lava_steps": 126, "desc": "High lava steps (should be 505 effective steps)"},
        {"step_count": 125, "lava_steps": 2, "desc": "High step count (should be 133 effective steps)"},
        {"step_count": 125, "lava_steps": 126, "desc": "Both high (should be 629 effective steps)"},
    ]
    
    # Try all combinations
    for i, case in enumerate(test_cases):
        # Reset state for next test
        env.reset()
        
        # Set values
        env.step_count = case["step_count"]
        lava_counter_env.lava_steps = case["lava_steps"]
        
        # Calculate effective steps
        effective_steps = lava_counter_env.get_effective_steps()
        
        # Print results
        print(f"\nTest {i+1}: {case['desc']}")
        print(f"  Regular steps: {env.step_count}")
        print(f"  Lava steps: {lava_counter_env.lava_steps}")
        print(f"  Expected formula: {env.step_count} + ({lava_counter_env.lava_steps} * (5.0 - 1))")
        print(f"  Expected calculation: {env.step_count} + ({lava_counter_env.lava_steps} * 4.0)")
        print(f"  Expected calculation: {env.step_count} + {lava_counter_env.lava_steps * 4.0}")
        print(f"  Expected result: {env.step_count + (lava_counter_env.lava_steps * 4.0)}")
        print(f"  Actual effective steps: {effective_steps}")
        
        if effective_steps == 504:
            print(f"  *** THIS MATCHES THE ISSUE! ***")
    
    # Let's try a specific value to replicate the 504
    print("\n=== Testing specific value to match 504 effective steps ===")
    env.reset()
    env.step_count = 1
    lava_counter_env.lava_steps = 125.75  # Trying a value that would give 504 effective steps
    effective_steps = lava_counter_env.get_effective_steps()
    print(f"  Regular steps: {env.step_count}")
    print(f"  Lava steps: {lava_counter_env.lava_steps}")
    print(f"  Formula result: {env.step_count + (lava_counter_env.lava_steps * 4.0)}")
    print(f"  Actual effective steps: {effective_steps}")
    
    # Check if the specific issue is in the reward function
    reward_slope = reward_env.y_intercept / reward_env.x_intercept
    expected_reward = max(0, reward_env.y_intercept - reward_slope * effective_steps)
    print(f"  Expected reward with {effective_steps} effective steps: {expected_reward}")
    
    # Clean up
    reward_env.close()
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    debug_high_effective_steps() 