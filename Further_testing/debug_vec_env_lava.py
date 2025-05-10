import gymnasium as gym
import minigrid
import numpy as np
from minigrid.wrappers import FullyObsWrapper, NoDeath
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper

def debug_wrapped_lava_steps():
    """Debug if lava steps are being properly reset after episode completion."""
    print("\n=== Debugging Lava Steps Reset ===")
    
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
    print("\nRunning 3 steps...")
    for i in range(3):
        action = 1  # Turn right
        obs, reward, terminated, truncated, info = env.step(action)
        lava_steps = info.get('lava_steps', 0)
        effective_steps = info.get('effective_steps', i+1)
        print(f"Step {i+1}: lava_steps={lava_steps}, effective_steps={effective_steps}")
    
    # Manually add lava steps
    print("\nManually adding lava steps...")
    # Find the LavaStepCounterWrapper
    current_env = env
    lava_wrapper = None
    while hasattr(current_env, 'env'):
        if isinstance(current_env, LavaStepCounterWrapper):
            lava_wrapper = current_env
            break
        current_env = current_env.env
    
    if lava_wrapper:
        lava_wrapper.lava_steps = 125
        print(f"Set lava_steps to 125")
        # Take one more step
        action = 1
        obs, reward, terminated, truncated, info = env.step(action)
        lava_steps = info.get('lava_steps', 0)
        effective_steps = info.get('effective_steps', 4)
        print(f"After setting lava_steps: lava_steps={lava_steps}, effective_steps={effective_steps}")
    else:
        print("Could not find LavaStepCounterWrapper")
    
    # Reset the environment
    print("\nResetting after manually setting lava steps...")
    obs, _ = env.reset()
    
    # Check if lava steps were reset
    action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    lava_steps = info.get('lava_steps', 0)
    effective_steps = info.get('effective_steps', 1)
    print(f"After reset: lava_steps={lava_steps}, effective_steps={effective_steps}")
    
    # Now try with a truncated episode
    print("\nTesting with a truncated episode...")
    
    # Manually set step count high to cause truncation
    base_env = None
    current_env = env
    while hasattr(current_env, 'env'):
        if hasattr(current_env, 'max_steps') and hasattr(current_env, '_elapsed_steps'):
            base_env = current_env
            break
        current_env = current_env.env
    
    if base_env:
        # Force a truncation by setting elapsed steps near max
        if hasattr(base_env, '_max_episode_steps'):
            max_steps = base_env._max_episode_steps
            print(f"Setting _elapsed_steps to {max_steps-1} to force truncation")
            base_env._elapsed_steps = max_steps - 1
    
    # Try to simulate truncation
    # Also manually set lava steps again
    if lava_wrapper:
        lava_wrapper.lava_steps = 125
        print(f"Set lava_steps to 125 again")
    
    # Take a step which should cause truncation
    action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    lava_steps = info.get('lava_steps', 0)
    effective_steps = info.get('effective_steps', 2)
    print(f"After forcing truncation: lava_steps={lava_steps}, effective_steps={effective_steps}, truncated={truncated}")
    
    # Reset and check again
    print("\nResetting after truncation...")
    obs, _ = env.reset()
    
    # Check if lava steps were reset
    action = 1
    obs, reward, terminated, truncated, info = env.step(action)
    lava_steps = info.get('lava_steps', 0)
    effective_steps = info.get('effective_steps', 1)
    print(f"After reset following truncation: lava_steps={lava_steps}, effective_steps={effective_steps}")
    
    # Clean up
    env.close()
    
    print("\n=== Debug Complete ===")
    
if __name__ == "__main__":
    debug_wrapped_lava_steps() 