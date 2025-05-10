import gymnasium as gym
import minigrid
from EnvironmentEdits.BespokeEdits.RewardModifications import EpisodeCompletionRewardWrapper
import numpy as np

def test_reward_wrapper():
    """
    Test the reward wrapper with a simple environment.
    """
    print("\n=== Testing Reward Wrapper ===")
    
    # Create a simple environment
    env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='human')
    
    # Wrap it with our reward wrapper
    wrapped_env = EpisodeCompletionRewardWrapper(
        env,
        reward_type='linear',
        x_intercept=50,  # Shorter x-intercept for testing
        y_intercept=1.0,
        transition_width=10
    )
    
    # Run a few episodes
    num_episodes = 3
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}")
        obs, _ = wrapped_env.reset()
        done = False
        truncated = False
        step_count = 0
        
        # Get goal position from observation
        goal_pos = np.where(obs['image'][:, :, 0] == 8)  # Goal is color 8
        if len(goal_pos[0]) > 0:
            goal_y, goal_x = goal_pos[0][0], goal_pos[1][0]
            print(f"Goal position: ({goal_x}, {goal_y})")
        else:
            goal_x, goal_y = None, None
        
        while not (done or truncated):
            action = wrapped_env.action_space.sample()  # Default to random action
            if goal_x is not None and goal_y is not None:
                # Get agent position
                agent_pos = np.where(obs['image'][:, :, 0] == 10)  # Agent is color 10
                if len(agent_pos[0]) > 0:
                    agent_y, agent_x = agent_pos[0][0], agent_pos[1][0]
                    # Simple movement logic
                    if agent_x < goal_x:
                        action = 1  # right
                    elif agent_x > goal_x:
                        action = 3  # left
                    elif agent_y < goal_y:
                        action = 2  # down
                    elif agent_y > goal_y:
                        action = 0  # up
                    else:
                        action = 5  # pickup
            # Take the action
            obs, reward, done, truncated, info = wrapped_env.step(action)
            step_count += 1
            # Print step info
            print(f"Step {step_count}: Reward = {reward:.3f}")
            # If episode ended, print final info
            if done or truncated:
                print(f"Episode ended after {step_count} steps")
                print(f"Final reward: {reward:.3f}")
                print(f"Terminated: {done}, Truncated: {truncated}")
                print(f"Info: {info}")
    
    wrapped_env.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_reward_wrapper() 