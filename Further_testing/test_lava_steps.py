import gymnasium as gym
import minigrid
import numpy as np
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper

def test_lava_step_counter():
    """
    Test the lava step counter wrapper with a simple lava crossing environment.
    """
    print("\n=== Testing Lava Step Counter ===")
    
    # Create a lava crossing environment
    env = gym.make('MiniGrid-LavaCrossingS9N1-v0', render_mode='human')
    
    # Apply the lava step counter wrapper
    lava_counter_env = LavaStepCounterWrapper(
        env,
        lava_step_multiplier=2.0,
        verbose=True
    )
    
    # Run a test episode
    obs, _ = lava_counter_env.reset()
    done = False
    truncated = False
    step_count = 0
    
    print("\nRunning a test episode with just the lava counter...")
    
    while not (done or truncated) and step_count < 50:
        action = lava_counter_env.action_space.sample()
        obs, reward, done, truncated, info = lava_counter_env.step(action)
        step_count += 1
        
        is_in_lava = info.get('is_in_lava', False)
        lava_steps = info.get('lava_steps', 0)
        effective_steps = info.get('effective_steps', step_count)
        
        print(f"Step {step_count}: In lava: {is_in_lava}, Lava steps: {lava_steps}, " 
              f"Effective steps: {effective_steps:.1f}, Reward: {reward:.3f}")
    
    lava_counter_env.close()
    
    print("\n=== Test Complete ===")


def test_lava_reward_integration():
    """
    Test the integration of lava step counter with reward functions.
    """
    print("\n=== Testing Lava Step Counter with Reward Functions ===")
    
    # Create a lava crossing environment
    env = gym.make('MiniGrid-LavaCrossingS9N1-v0', render_mode='human')
    
    # Apply the lava step counter wrapper
    lava_counter_env = LavaStepCounterWrapper(
        env,
        lava_step_multiplier=3.0,  # Higher multiplier for more visible effect
        verbose=True
    )
    
    # Apply the reward wrapper on top
    reward_env = EpisodeCompletionRewardWrapper(
        lava_counter_env,
        reward_type='linear',
        x_intercept=100,
        y_intercept=10.0,
        count_lava_steps=True,
        verbose=True
    )
    
    # Run a test episode
    obs, _ = reward_env.reset()
    done = False
    truncated = False
    step_count = 0
    
    print("\nRunning a test episode with reward function considering lava steps...")
    
    while not (done or truncated) and step_count < 50:
        action = reward_env.action_space.sample()
        obs, reward, done, truncated, info = reward_env.step(action)
        step_count += 1
        
        is_in_lava = info.get('is_in_lava', False)
        lava_steps = info.get('lava_steps', 0)
        effective_steps = info.get('effective_steps', step_count)
        
        # Only print every few steps to reduce output
        if step_count % 5 == 0 or done or truncated:
            print(f"Step {step_count}: In lava: {is_in_lava}, Lava steps: {lava_steps}, " 
                  f"Effective steps: {effective_steps:.1f}, Reward: {reward:.3f}")
    
    reward_env.close()
    
    print("\n=== Comparison Test ===")
    
    # Now run a comparison with and without counting lava steps
    print("\nRunning comparison between regular reward and lava-aware reward...")
    
    # Without lava counting
    env1 = gym.make('MiniGrid-LavaCrossingS9N1-v0')
    env1 = EpisodeCompletionRewardWrapper(
        env1,
        reward_type='linear',
        x_intercept=100,
        y_intercept=10.0,
        count_lava_steps=False,
        verbose=False
    )
    
    # With lava counting
    env2 = gym.make('MiniGrid-LavaCrossingS9N1-v0')
    env2 = LavaStepCounterWrapper(
        env2,
        lava_step_multiplier=3.0,
        verbose=False
    )
    env2 = EpisodeCompletionRewardWrapper(
        env2,
        reward_type='linear',
        x_intercept=100,
        y_intercept=10.0,
        count_lava_steps=True,
        verbose=False
    )
    
    # Run multiple episodes and compare rewards
    num_episodes = 10
    regular_rewards = []
    lava_aware_rewards = []
    
    for i in range(num_episodes):
        # Regular reward env
        obs, _ = env1.reset()
        done = False
        while not done:
            action = env1.action_space.sample()
            obs, reward, done, _, _ = env1.step(action)
            if done and reward > 0:
                regular_rewards.append(reward)
                
        # Lava-aware reward env
        obs, _ = env2.reset()
        done = False
        lava_steps = 0
        while not done:
            action = env2.action_space.sample()
            obs, reward, done, _, info = env2.step(action)
            if done and reward > 0:
                lava_aware_rewards.append(reward)
                lava_steps = info.get('lava_steps', 0)
                
        print(f"Episode {i+1}: Regular reward: {regular_rewards[-1] if i < len(regular_rewards) else 'failed'}, "
              f"Lava-aware reward: {lava_aware_rewards[-1] if i < len(lava_aware_rewards) else 'failed'}, "
              f"Lava steps: {lava_steps}")
    
    env1.close()
    env2.close()
    
    # Print summary statistics
    if regular_rewards and lava_aware_rewards:
        print("\nSummary Statistics:")
        print(f"Regular rewards: mean={np.mean(regular_rewards):.3f}, std={np.std(regular_rewards):.3f}")
        print(f"Lava-aware rewards: mean={np.mean(lava_aware_rewards):.3f}, std={np.std(lava_aware_rewards):.3f}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    # Run both tests
    test_lava_step_counter()
    test_lava_reward_integration() 