import os
import yaml
from import_vars import load_config, create_observation_params, print_training_info
import EnvironmentEdits.EnvironmentGeneration as Env

def test_reward_config():
    """
    Test loading config file and creating environment with reward wrapper.
    """
    print("\n=== Testing Config Loading and Environment Creation ===")
    
    # Load config
    config = load_config("config.yaml")
    
    # Set up spawn visualization directory
    spawn_vis_dir = "./distribution_vis"
    os.makedirs(spawn_vis_dir, exist_ok=True)
    
    # Create observation parameters
    observation_params = create_observation_params(config, spawn_vis_dir)
    
    # Print training information
    print_training_info(config, observation_params)
    
    # Create a single environment for testing
    print("\n=== Creating Test Environment ===")
    env = Env._make_env(
        env_id=config['environment']['id'],
        seed=config['seeds']['environment'],
        render_mode="human",
        **observation_params
    )
    
    # Run a few steps to see if the reward wrapper is applied
    print("\n=== Running Test Episode ===")
    obs, _ = env.reset()
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward = {reward:.3f}")
        if done or truncated:
            break
    
    env.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_reward_config() 