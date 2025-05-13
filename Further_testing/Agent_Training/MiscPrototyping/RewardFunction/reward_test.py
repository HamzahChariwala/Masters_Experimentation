import gymnasium as gym
import minigrid
from EnvironmentEdits.BespokeEdits.RewardModifications import EpisodeCompletionRewardWrapper
from EnvironmentEdits.EnvironmentGeneration import _make_env
import numpy as np

def test_direct_config():
    """
    Test creating an environment with reward wrapper configured directly.
    """
    print("\n=== Testing Direct Reward Wrapper Configuration ===")
    
    # Create environment directly with the reward wrapper
    env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='human')
    wrapped_env = EpisodeCompletionRewardWrapper(
        env,
        reward_type='linear',
        x_intercept=50,
        y_intercept=1.0,
        transition_width=10
    )
    
    # Run a test episode
    obs, _ = wrapped_env.reset()
    done = False
    truncated = False
    step_count = 0
    
    while not (done or truncated) and step_count < 100:
        action = wrapped_env.action_space.sample()
        obs, reward, done, truncated, info = wrapped_env.step(action)
        step_count += 1
        print(f"Step {step_count}: Reward = {reward:.3f}")
    
    wrapped_env.close()
    
    # Now try using the _make_env function
    print("\n=== Testing _make_env with Reward Wrapper ===")
    
    env2 = _make_env(
        env_id='MiniGrid-Empty-5x5-v0',
        render_mode='human',
        use_reward_function=True,
        reward_type='linear',
        reward_x_intercept=50,
        reward_y_intercept=1.0,
        reward_transition_width=10
    )
    
    # Run a test episode
    obs, _ = env2.reset()
    done = False
    truncated = False
    step_count = 0
    
    while not (done or truncated) and step_count < 100:
        action = env2.action_space.sample()
        obs, reward, done, truncated, info = env2.step(action)
        step_count += 1
        print(f"Step {step_count}: Reward = {reward:.3f}")
    
    env2.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_direct_config() 