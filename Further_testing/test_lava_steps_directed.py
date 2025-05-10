import gymnasium as gym
import minigrid
import numpy as np
from gymnasium import spaces
from EnvironmentEdits.BespokeEdits.RewardModifications import LavaStepCounterWrapper, EpisodeCompletionRewardWrapper

class DirectedAgent:
    """A simple agent that follows a predefined path to test lava crossing."""
    
    def __init__(self, env):
        self.env = env
        
        # Define actions
        # MiniGrid actions: 0=turn left, 1=turn right, 2=move forward
        self.actions = {
            'left': 0,
            'right': 1,
            'forward': 2
        }
        
        # Simple path sequence to reach the goal: 
        # go forward until near lava, turn right, go through lava, turn left, go to goal
        self.action_sequence = (
            [self.actions['forward']] * 3 +  # Move forward
            [self.actions['right']] +         # Turn right
            [self.actions['forward']] * 3 +   # Move through lava
            [self.actions['left']] +          # Turn left
            [self.actions['forward']] * 3     # Reach goal
        )
        self.current_step = 0
    
    def act(self):
        """Return the next action in the sequence or a random action if out of sequence."""
        if self.current_step < len(self.action_sequence):
            action = self.action_sequence[self.current_step]
            self.current_step += 1
            return action
        else:
            # Random action if we've completed the sequence but haven't reached the goal
            return np.random.randint(0, 3)

def test_directed_lava_crossing():
    """
    Test lava step counting with a directed agent that purposely crosses lava.
    Uses a simplified environment for consistent testing.
    """
    print("\n=== Testing Directed Lava Crossing ===")
    
    # Create a known lava crossing environment
    env = gym.make('MiniGrid-SimpleCrossingS9N1-v0', render_mode='human')
    
    # Wrap with lava counter
    lava_env = LavaStepCounterWrapper(
        env,
        lava_step_multiplier=2.0,
        verbose=True
    )
    
    # Add reward wrapper
    reward_env = EpisodeCompletionRewardWrapper(
        lava_env,
        reward_type='linear',
        x_intercept=50,
        y_intercept=10.0,
        count_lava_steps=True,
        verbose=True
    )
    
    # Create directed agent
    agent = DirectedAgent(reward_env)
    
    # Run an episode
    obs, _ = reward_env.reset()
    done = False
    truncated = False
    step_count = 0
    
    print("\nRunning directed agent through lava to goal...")
    
    while not (done or truncated) and step_count < 50:
        action = agent.act()
        obs, reward, done, truncated, info = reward_env.step(action)
        step_count += 1
        
        is_in_lava = info.get('is_in_lava', False)
        lava_steps = info.get('lava_steps', 0)
        effective_steps = info.get('effective_steps', step_count)
        
        print(f"Step {step_count}: Action={action}, In lava: {is_in_lava}, "
              f"Lava steps: {lava_steps}, Effective steps: {effective_steps:.1f}, "
              f"Reward: {reward:.3f}")
        
        if done:
            print(f"\nEpisode completed in {step_count} steps")
            print(f"Regular steps: {step_count - lava_steps}")
            print(f"Lava steps: {lava_steps}")
            print(f"Effective steps: {effective_steps:.1f}")
            print(f"Final reward: {reward:.3f}")
    
    reward_env.close()
    
    # Comparison test: with and without lava counting
    print("\n=== Comparison Test ===")
    
    # First run with no lava penalty
    env1 = gym.make('MiniGrid-SimpleCrossingS9N1-v0')
    env1 = EpisodeCompletionRewardWrapper(
        env1,
        reward_type='linear',
        x_intercept=50,
        y_intercept=10.0,
        count_lava_steps=False,
        verbose=False
    )
    
    # Create agent for first env
    agent1 = DirectedAgent(env1)
    
    # Run episode
    obs, _ = env1.reset()
    done = False
    steps1 = 0
    reward1 = 0
    
    while not done and steps1 < 50:
        action = agent1.act()
        obs, reward, done, _, _ = env1.step(action)
        steps1 += 1
        if done:
            reward1 = reward
    
    env1.close()
    
    # Now with lava penalty
    env2 = gym.make('MiniGrid-SimpleCrossingS9N1-v0')
    env2 = LavaStepCounterWrapper(
        env2,
        lava_step_multiplier=2.0,
        verbose=False
    )
    env2 = EpisodeCompletionRewardWrapper(
        env2,
        reward_type='linear',
        x_intercept=50,
        y_intercept=10.0,
        count_lava_steps=True,
        verbose=False
    )
    
    # Create agent for second env
    agent2 = DirectedAgent(env2)
    
    # Run episode
    obs, _ = env2.reset()
    done = False
    steps2 = 0
    reward2 = 0
    lava_steps = 0
    
    while not done and steps2 < 50:
        action = agent2.act()
        obs, reward, done, _, info = env2.step(action)
        steps2 += 1
        if done:
            reward2 = reward
            lava_steps = info.get('lava_steps', 0)
    
    env2.close()
    
    # Compare results
    print(f"\nWithout lava counting: Steps={steps1}, Reward={reward1:.3f}")
    print(f"With lava counting: Steps={steps2}, Lava steps={lava_steps}, Reward={reward2:.3f}")
    print(f"Difference in reward: {reward2 - reward1:.3f}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    try:
        # Try with SimpleCrossing environment
        test_directed_lava_crossing()
    except gym.error.Error:
        # If the SimpleCrossing environment is not available, use LavaCrossing
        print("\nSimpleCrossing environment not found. Falling back to LavaCrossing environment.")
        
        # Update the script to work with LavaCrossing
        # This would require different action sequences depending on the layout
        # For now, we'll just inform the user
        print("This test requires a custom environment with a predictable lava crossing path.")
        print("Please create a test environment or modify the action sequence for your specific environment.") 