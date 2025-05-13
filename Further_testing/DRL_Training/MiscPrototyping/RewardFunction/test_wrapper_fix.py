import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, NoDeath
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper

def test_wrapper_fix():
    """Test the fixed LavaStepCounterWrapper implementation."""
    print("\n=== Testing Fixed LavaStepCounterWrapper ===")
    
    # Create a test environment
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = FullyObsWrapper(env)
    env = LavaStepCounterWrapper(env, lava_step_multiplier=5.0, verbose=True)
    env = EpisodeCompletionRewardWrapper(
        env,
        reward_type='linear',
        x_intercept=100,
        y_intercept=1.0,
        count_lava_steps=True,
        verbose=True
    )
    
    # Reset the environment
    print("\nInitial reset...")
    obs, _ = env.reset()
    
    # Run a few steps
    print("\nRunning 5 steps...")
    for i in range(5):
        action = 1  # Turn right
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: lava_steps={info.get('lava_steps', 0)}, effective_steps={info.get('effective_steps', 0)}")
    
    # Check internal step counts
    print("\nAfter 5 steps, checking internal counters:")
    # Access the lava counter wrapper
    lava_wrapper = None
    current_env = env
    while hasattr(current_env, 'env'):
        if isinstance(current_env, LavaStepCounterWrapper):
            lava_wrapper = current_env
            break
        current_env = current_env.env
    
    if lava_wrapper:
        print(f"  Base env step_count: {lava_wrapper.env.step_count}")
        print(f"  Wrapper step_count: {lava_wrapper.wrapper_step_count}")
    
    # Now reset and check if counters reset properly
    print("\nResetting environment...")
    obs, _ = env.reset()
    
    # Check if counters were reset
    if lava_wrapper:
        print(f"  After reset - Base env step_count: {lava_wrapper.env.step_count}")
        print(f"  After reset - Wrapper step_count: {lava_wrapper.wrapper_step_count}")
    
    # Run a few more steps
    print("\nRunning 3 more steps after reset...")
    for i in range(3):
        action = 1  # Turn right
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: lava_steps={info.get('lava_steps', 0)}, effective_steps={info.get('effective_steps', 0)}")
    
    # Check counters again
    if lava_wrapper:
        print(f"\nAfter 3 steps post-reset:")
        print(f"  Base env step_count: {lava_wrapper.env.step_count}")
        print(f"  Wrapper step_count: {lava_wrapper.wrapper_step_count}")
    
    # Now simulate some cumulative base environment step counts
    print("\nSimulating high base environment step count...")
    
    # Manually set base env step count to be high
    # but our wrapper should still count correctly
    if lava_wrapper:
        lava_wrapper.env.step_count = 500
        print(f"  Manually set base env step_count to 500")
    
    # Take a step to see what happens
    action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check counters
    if lava_wrapper:
        print(f"\nAfter another step with high base count:")
        print(f"  Base env step_count: {lava_wrapper.env.step_count}")
        print(f"  Wrapper step_count: {lava_wrapper.wrapper_step_count}")
        print(f"  Reported effective_steps: {info.get('effective_steps', 0)}")
    
    # Clean up
    env.close()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_wrapper_fix() 