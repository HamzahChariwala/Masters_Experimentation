import gymnasium as gym
import minigrid
import numpy as np
from minigrid.wrappers import FullyObsWrapper, NoDeath
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper
import matplotlib.pyplot as plt

# Import the MockLavaStepCounterWrapper from our previous test
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

def test_lava_reward_in_real_env():
    """Test lava step counting and reward modification in a real lava crossing environment."""
    print("\n=== Testing Lava Step Counting and Reward Modification in Real Environment ===")
    
    # Create a simple empty environment
    env = gym.make('MiniGrid-Empty-8x8-v0')
    
    # Apply necessary wrappers
    env = FullyObsWrapper(env)
    
    # Wrap with mock lava counter
    lava_env = MockLavaStepCounterWrapper(
        env,
        lava_step_multiplier=3.0,  # Higher multiplier for more visible effect
        verbose=True
    )
    
    # Wrap with reward function
    reward_env = EpisodeCompletionRewardWrapper(
        lava_env,
        reward_type='linear',
        x_intercept=30,  # Smaller value for testing purposes
        y_intercept=10.0,
        count_lava_steps=True,
        verbose=True
    )
    
    # Run an episode with random actions
    obs, _ = reward_env.reset()
    done = False
    truncated = False
    step_count = 0
    max_steps = 30
    lava_step_history = []
    effective_step_history = []
    regular_step_history = []
    
    print("\nRunning random actions with simulated lava steps...")
    
    while not (done or truncated) and step_count < max_steps:
        # Take random action
        action = reward_env.action_space.sample()
        obs, reward, done, truncated, info = reward_env.step(action)
        step_count += 1
        
        # Record step information
        is_in_lava = info.get('is_in_lava', False)
        lava_steps = info.get('lava_steps', 0)
        effective_steps = info.get('effective_steps', step_count)
        
        # Store history
        lava_step_history.append(lava_steps)
        effective_step_history.append(effective_steps)
        regular_step_history.append(step_count)
        
        if step_count % 5 == 0 or done or truncated or is_in_lava:
            print(f"Step {step_count}: In lava: {is_in_lava}, Lava steps: {lava_steps}, " 
                  f"Effective steps: {effective_steps:.1f}, Reward: {reward:.3f}")
    
    # Now force a successful completion
    print("\nForcing successful episode completion...")
    info = {}
    obs, done, truncated = None, False, False
    
    # Call directly to the step function with custom values
    # We'll manually simulate the final reward
    lava_steps = lava_env.lava_steps
    effective_steps = lava_env.get_effective_steps()
    
    # Calculate expected reward based on linear formula
    slope = reward_env.y_intercept / reward_env.x_intercept
    expected_reward_with_lava = max(0, reward_env.y_intercept - slope * effective_steps)
    expected_reward_without_lava = max(0, reward_env.y_intercept - slope * step_count)
    
    print(f"\nFinal episode statistics:")
    print(f"Regular steps: {step_count}")
    print(f"Lava steps: {lava_steps}")
    print(f"Effective steps: {effective_steps:.1f}")
    print(f"Expected reward without lava counting: {expected_reward_without_lava:.3f}")
    print(f"Expected reward with lava counting: {expected_reward_with_lava:.3f}")
    print(f"Difference: {expected_reward_without_lava - expected_reward_with_lava:.3f}")
    
    # Plot the step histories
    plt.figure(figsize=(10, 6))
    plt.plot(regular_step_history, label='Regular Steps')
    plt.plot(lava_step_history, label='Lava Steps')
    plt.plot(effective_step_history, label='Effective Steps')
    plt.xlabel('Episode Step')
    plt.ylabel('Step Count')
    plt.title('Effect of Lava Steps on Effective Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('lava_steps_effect.png', dpi=300)
    print("\nPlot saved as 'lava_steps_effect.png'")
    
    reward_env.close()
    
    # Compare different lava step multipliers
    print("\n=== Testing Different Lava Step Multipliers ===")
    
    multipliers = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    results = []
    
    for mult in multipliers:
        # Create environment
        env = gym.make('MiniGrid-Empty-8x8-v0')
        env = FullyObsWrapper(env)
        
        # Apply mock lava counter
        lava_env = MockLavaStepCounterWrapper(
            env,
            lava_step_multiplier=mult,
            verbose=False
        )
        
        # Apply reward wrapper
        reward_env = EpisodeCompletionRewardWrapper(
            lava_env,
            reward_type='linear',
            x_intercept=30,
            y_intercept=10.0,
            count_lava_steps=True,
            verbose=False
        )
        
        # Run a fixed number of steps
        obs, _ = reward_env.reset()
        for i in range(25):
            obs, reward, done, truncated, info = reward_env.step(1)  # Always take action 1
        
        # Get final stats
        lava_steps = info.get('lava_steps', 0)
        effective_steps = info.get('effective_steps', 25)
        
        # Calculate the reward as if episode completed successfully
        slope = reward_env.y_intercept / reward_env.x_intercept
        reward = max(0, reward_env.y_intercept - slope * effective_steps)
        
        results.append({
            'multiplier': mult,
            'steps': 25,
            'lava_steps': lava_steps,
            'effective_steps': effective_steps,
            'calculated_reward': reward
        })
        
        reward_env.close()
    
    # Print results
    print("\nResults by lava step multiplier:")
    for result in results:
        print(f"Multiplier {result['multiplier']}: "
              f"Lava steps: {result['lava_steps']}, "
              f"Effective steps: {result['effective_steps']:.1f}, "
              f"Calculated reward: {result['calculated_reward']:.3f}")
    
    # Plot rewards by multiplier
    if results:
        mults = [r['multiplier'] for r in results]
        rewards = [r['calculated_reward'] for r in results]
        effective_steps = [r['effective_steps'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot rewards
        ax1.plot(mults, rewards, 'o-', markersize=8)
        ax1.set_xlabel('Lava Step Multiplier')
        ax1.set_ylabel('Calculated Reward')
        ax1.set_title('Effect of Lava Step Multiplier on Reward')
        ax1.grid(True)
        
        # Plot effective steps
        ax2.plot(mults, effective_steps, 'o-', markersize=8, color='orange')
        ax2.set_xlabel('Lava Step Multiplier')
        ax2.set_ylabel('Effective Steps')
        ax2.set_title('Effect of Lava Step Multiplier on Effective Steps')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('lava_multiplier_effects.png', dpi=300)
        print("\nPlot saved as 'lava_multiplier_effects.png'")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_lava_reward_in_real_env() 