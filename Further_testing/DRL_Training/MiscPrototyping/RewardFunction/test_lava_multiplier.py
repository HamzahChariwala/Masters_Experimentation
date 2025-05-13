import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, NoDeath
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper

def test_lava_multiplier():
    """Test that the lava_step_multiplier is correctly applied in calculations."""
    print("\n=== Testing Lava Step Multiplier ===")
    
    # Create environment with a specific multiplier
    multiplier = 5.0
    print(f"\nCreating environment with lava_step_multiplier={multiplier}")
    
    env = gym.make('MiniGrid-LavaGapS7-v0')
    env = FullyObsWrapper(env)
    env = NoDeath(env, no_death_types=("lava",), death_cost=-0.1)
    env = LavaStepCounterWrapper(
        env,
        lava_step_multiplier=multiplier,
        verbose=True,
        debug_logging=True
    )
    
    # Reset and run steps, forcing the agent to go through lava
    obs, _ = env.reset(seed=42)
    
    print("\nRunning steps with the agent through lava...")
    
    # Sample steps to demonstrate the calculation
    regular_steps = 0
    lava_steps = 0
    
    # Simulate a few steps, with some on lava
    for i in range(10):
        action = 2  # Move forward (adjust as needed)
        obs, reward, terminated, truncated, info = env.step(action)
        
        regular_steps += 1
        if info['is_in_lava']:
            lava_steps += 1
        
        print(f"\nStep {regular_steps}:")
        print(f"  Is in lava: {info['is_in_lava']}")
        print(f"  Lava steps so far: {info['lava_steps']}")
        print(f"  Effective steps: {info['effective_steps']}")
        
        expected_effective = regular_steps + (lava_steps * (multiplier - 1))
        print(f"  Expected effective steps: {expected_effective}")
        
        if info['effective_steps'] != expected_effective:
            print(f"  ❌ MISMATCH: Effective steps calculation is incorrect!")
        else:
            print(f"  ✓ Calculation correct")
    
    # Clean up
    env.close()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_lava_multiplier() 