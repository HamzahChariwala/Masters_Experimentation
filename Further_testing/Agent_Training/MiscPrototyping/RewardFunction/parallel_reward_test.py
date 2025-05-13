import gymnasium as gym
import minigrid
from EnvironmentEdits.BespokeEdits.RewardModifications import EpisodeCompletionRewardWrapper
from EnvironmentEdits.EnvironmentGeneration import _make_env, make_parallel_env
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

def test_parallel_reward_wrapper():
    """
    Test if print statements from reward wrapper are visible in SubprocVecEnv.
    """
    print("\n=== Testing Single Environment Wrapper ===")
    
    # Create a single environment with the reward wrapper
    env = _make_env(
        env_id='MiniGrid-Empty-5x5-v0',
        render_mode=None,
        use_reward_function=True,
        reward_type='linear',
        reward_x_intercept=50,
        reward_y_intercept=1.0,
        reward_transition_width=10
    )
    
    # Run a short episode
    obs, _ = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Single env - Step {i+1}: Reward = {reward:.3f}")
        if done or truncated:
            break
    
    env.close()
    
    print("\n=== Testing Parallel Environment Wrapper ===")
    
    # Define environment creation function for SubprocVecEnv
    def make_env_fn():
        def _init():
            return _make_env(
                env_id='MiniGrid-Empty-5x5-v0',
                render_mode=None,
                use_reward_function=True,
                reward_type='linear',
                reward_x_intercept=50,
                reward_y_intercept=1.0,
                reward_transition_width=10
            )
        return _init
    
    # Create parallel environment
    parallel_env = SubprocVecEnv([make_env_fn() for _ in range(2)])
    
    # Run a short episode in parallel
    obs = parallel_env.reset()
    for i in range(10):
        actions = [parallel_env.action_space.sample() for _ in range(2)]
        obs, rewards, dones, infos = parallel_env.step(actions)
        print(f"Parallel env - Step {i+1}: Rewards = {rewards}")
    
    parallel_env.close()
    
    print("\n=== Testing make_parallel_env Function ===")
    
    # Use the make_parallel_env function from EnvironmentGeneration
    parallel_env2 = make_parallel_env(
        env_id='MiniGrid-Empty-5x5-v0',
        num_envs=2,
        env_seed=42,
        use_reward_function=True,
        reward_type='linear',
        reward_x_intercept=50,
        reward_y_intercept=1.0,
        reward_transition_width=10
    )
    
    # Run a short episode in parallel
    obs = parallel_env2.reset()
    for i in range(10):
        actions = [parallel_env2.action_space.sample() for _ in range(2)]
        obs, rewards, dones, infos = parallel_env2.step(actions)
        print(f"make_parallel_env - Step {i+1}: Rewards = {rewards}")
    
    parallel_env2.close()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_parallel_reward_wrapper() 