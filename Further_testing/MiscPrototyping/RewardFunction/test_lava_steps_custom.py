import gymnasium as gym
import minigrid
import numpy as np
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper

class MockLavaStepCounterWrapper(LavaStepCounterWrapper):
    """A modified wrapper that simulates lava steps by directly setting the counter."""
    
    def __init__(self, env, lava_step_multiplier=2.0, verbose=True):
        super().__init__(env, lava_step_multiplier, verbose)
        # Additional parameter to control when to simulate lava steps
        self.simulate_lava_at_step = 5
        self.simulate_exit_at_step = 10
        self.simulate_lava_again_at_step = 15
        self.simulate_exit_again_at_step = 20
    
    def step(self, action):
        """Override step to simulate lava interactions at specific steps."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Count the step
        curr_step = self.env.step_count
        
        # Simulate entering lava at specific steps
        if curr_step == self.simulate_lava_at_step:
            if self.verbose:
                print(f"Agent entered lava at step {curr_step} (simulated)")
            self.is_in_lava = True
            self.lava_steps += 1
        elif curr_step > self.simulate_lava_at_step and curr_step < self.simulate_exit_at_step:
            # Continue counting lava steps
            if self.is_in_lava:
                self.lava_steps += 1
        elif curr_step == self.simulate_exit_at_step:
            if self.is_in_lava and self.verbose:
                print(f"Agent exited lava at step {curr_step} (simulated)")
            self.is_in_lava = False
        
        # Simulate entering lava again
        elif curr_step == self.simulate_lava_again_at_step:
            if self.verbose:
                print(f"Agent entered lava again at step {curr_step} (simulated)")
            self.is_in_lava = True
            self.lava_steps += 1
        elif curr_step > self.simulate_lava_again_at_step and curr_step < self.simulate_exit_again_at_step:
            # Continue counting lava steps
            if self.is_in_lava:
                self.lava_steps += 1
        elif curr_step == self.simulate_exit_again_at_step:
            if self.is_in_lava and self.verbose:
                print(f"Agent exited lava at step {curr_step} (simulated)")
            self.is_in_lava = False
        
        # Update info with lava status
        info['lava_steps'] = self.lava_steps
        info['is_in_lava'] = self.is_in_lava
        info['effective_steps'] = self.get_effective_steps()
        
        return obs, reward, terminated, truncated, info


def test_simulated_lava_steps():
    """Test the lava step counter with simulated lava interactions."""
    print("\n=== Testing Simulated Lava Steps ===")
    
    # Create a simple empty environment
    env = gym.make('MiniGrid-Empty-8x8-v0')
    
    # Wrap with mock lava counter
    lava_env = MockLavaStepCounterWrapper(
        env,
        lava_step_multiplier=2.0,
        verbose=True
    )
    
    # Run an episode
    obs, _ = lava_env.reset()
    done = False
    truncated = False
    step_count = 0
    max_steps = 30
    
    print("\nRunning an episode with simulated lava steps...")
    
    while not (done or truncated) and step_count < max_steps:
        # Use random actions
        action = lava_env.action_space.sample()
        obs, reward, done, truncated, info = lava_env.step(action)
        step_count += 1
        
        is_in_lava = info.get('is_in_lava', False)
        lava_steps = info.get('lava_steps', 0)
        effective_steps = info.get('effective_steps', step_count)
        
        print(f"Step {step_count}: In lava: {is_in_lava}, Lava steps: {lava_steps}, " 
              f"Effective steps: {effective_steps:.1f}, Regular steps: {step_count}")
    
    lava_env.close()
    
    print("\n=== Test Complete ===")


def test_reward_with_simulated_lava():
    """Test the reward modification with simulated lava steps."""
    print("\n=== Testing Reward Modification with Simulated Lava ===")
    
    # Create a simple empty environment
    env = gym.make('MiniGrid-Empty-8x8-v0')
    
    # Apply mock lava counter
    lava_env = MockLavaStepCounterWrapper(
        env,
        lava_step_multiplier=3.0,  # Higher multiplier for more visible effect
        verbose=True
    )
    
    # Apply reward wrapper
    reward_env = EpisodeCompletionRewardWrapper(
        lava_env,
        reward_type='linear',
        x_intercept=50,
        y_intercept=10.0,
        count_lava_steps=True,
        verbose=True
    )
    
    # Run controlled episode: we'll complete it after a specific number of steps
    obs, _ = reward_env.reset()
    steps_to_complete = 25  # We'll terminate after this many steps
    complete_with_reward = 1.0  # Custom reward to simulate successful completion
    
    for step in range(1, steps_to_complete + 1):
        action = reward_env.action_space.sample()
        obs, reward, done, truncated, info = reward_env.step(action)
        
        is_in_lava = info.get('is_in_lava', False)
        lava_steps = info.get('lava_steps', 0)
        effective_steps = info.get('effective_steps', step)
        
        print(f"Step {step}: In lava: {is_in_lava}, Lava steps: {lava_steps}, " 
              f"Effective steps: {effective_steps:.1f}, Reward: {reward:.3f}")
        
        # Force episode completion at the specified step
        if step == steps_to_complete:
            # Simulate successful completion by calling reward wrapper's step method with
            # custom values for termination and reward
            print("\nForcing episode completion...")
            # We can't directly call this, so instead we'll print the expected values
            
            # Calculate expected reward
            regular_steps = step - lava_steps
            extra_steps = lava_steps * (lava_env.lava_step_multiplier - 1)
            total_effective_steps = step + extra_steps
            
            # Linear reward calculation
            slope = reward_env.y_intercept / reward_env.x_intercept
            expected_reward_with_lava = max(0, reward_env.y_intercept - slope * total_effective_steps)
            expected_reward_without_lava = max(0, reward_env.y_intercept - slope * step)
            
            print(f"\nExpected final values:")
            print(f"  Regular steps: {regular_steps}")
            print(f"  Lava steps: {lava_steps}")
            print(f"  Lava step multiplier: {lava_env.lava_step_multiplier}")
            print(f"  Additional effective steps due to lava: {extra_steps:.1f}")
            print(f"  Total effective steps: {total_effective_steps:.1f}")
            print(f"  Expected reward with lava counting: {expected_reward_with_lava:.3f}")
            print(f"  Expected reward without lava counting: {expected_reward_without_lava:.3f}")
            print(f"  Difference: {expected_reward_without_lava - expected_reward_with_lava:.3f}")
            break
    
    reward_env.close()
    
    print("\n=== Comparison Test ===")
    
    # Now run a comparison with and without counting lava steps
    print("\nRunning comparison between regular reward and lava-aware reward...")
    
    # Setup environments with identical parameters except for lava counting
    # First environment: without lava counting
    env1 = gym.make('MiniGrid-Empty-8x8-v0')
    env1 = MockLavaStepCounterWrapper(env1, lava_step_multiplier=3.0, verbose=False)
    env1 = EpisodeCompletionRewardWrapper(
        env1,
        reward_type='linear',
        x_intercept=50,
        y_intercept=10.0,
        count_lava_steps=False,  # No lava counting
        verbose=False
    )
    
    # Second environment: with lava counting
    env2 = gym.make('MiniGrid-Empty-8x8-v0')
    env2 = MockLavaStepCounterWrapper(env2, lava_step_multiplier=3.0, verbose=False)
    env2 = EpisodeCompletionRewardWrapper(
        env2,
        reward_type='linear',
        x_intercept=50,
        y_intercept=10.0,
        count_lava_steps=True,  # With lava counting
        verbose=False
    )
    
    # Function to run the controlled episode
    def run_controlled_episode(env, count_lava_in_name="Unknown"):
        obs, _ = env.reset()
        for step in range(1, steps_to_complete + 1):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if step % 5 == 0:  # Print every 5 steps
                is_in_lava = info.get('is_in_lava', False)
                lava_steps = info.get('lava_steps', 0)
                print(f"{count_lava_in_name} - Step {step}: In lava: {is_in_lava}, "
                      f"Lava steps: {lava_steps}, Reward: {reward:.3f}")
        
        # Calculate and return the final values
        lava_steps = info.get('lava_steps', 0)
        effective_steps = info.get('effective_steps', steps_to_complete)
        
        # Manually calculate what the reward would be with a successful completion
        slope = 10.0 / 50.0  # y_intercept / x_intercept
        if count_lava_in_name == "With lava counting":
            reward_value = max(0, 10.0 - slope * effective_steps)
        else:
            reward_value = max(0, 10.0 - slope * steps_to_complete)
        
        return {
            'steps': steps_to_complete,
            'lava_steps': lava_steps,
            'effective_steps': effective_steps,
            'calculated_reward': reward_value
        }
    
    # Run both environments with the same action sequence
    print("\nRunning environment WITHOUT lava counting:")
    result1 = run_controlled_episode(env1, "Without lava counting")
    
    print("\nRunning environment WITH lava counting:")
    result2 = run_controlled_episode(env2, "With lava counting")
    
    # Compare results
    print("\nFinal Comparison:")
    print(f"WITHOUT lava counting: Steps={result1['steps']}, Lava Steps={result1['lava_steps']}, "
          f"Effective Steps={result1['effective_steps']:.1f}, Calculated Reward={result1['calculated_reward']:.3f}")
    print(f"WITH lava counting: Steps={result2['steps']}, Lava Steps={result2['lava_steps']}, "
          f"Effective Steps={result2['effective_steps']:.1f}, Calculated Reward={result2['calculated_reward']:.3f}")
    
    reward_diff = result1['calculated_reward'] - result2['calculated_reward']
    print(f"Reward difference: {reward_diff:.3f}")
    
    env1.close()
    env2.close()
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    # Run both tests
    test_simulated_lava_steps()
    test_reward_with_simulated_lava() 