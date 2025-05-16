import os
import sys
import yaml
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Optional

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")

# Import from Environment_Tooling
from Environment_Tooling.EnvironmentGeneration import make_env
from Environment_Tooling.BespokeEdits.FeatureExtractor import CustomCombinedExtractor

# Import from stable_baselines3
from stable_baselines3 import DQN, PPO, A2C

# Import our grid extraction utilities
from Agent_Evaluation.extract_grid import extract_grid_from_env, visualize_env_tensor, print_env_tensor

# Default values for evaluation
DEFAULT_RANK = 0
DEFAULT_NUM_EPISODES = 1

def load_config(agent_folder: str) -> Dict[str, Any]:
    """
    Load configuration from the agent's config.yaml file.
    
    Args:
        agent_folder (str): Path to the agent folder relative to Agent_Storage
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Check if the path already includes Agent_Storage prefix
    if agent_folder.startswith("Agent_Storage/"):
        config_path = os.path.join(agent_folder, "config.yaml")
    else:
        # Construct path to the agent's config file in Agent_Storage
        config_path = os.path.join("Agent_Storage", agent_folder, "config.yaml")
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config


def extract_env_config(config: Dict[str, Any], override_env_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract only the environment-related settings needed to create the environment.
    Always override NoDeath settings to ensure it's applied with a cost of 0.
    
    Args:
        config (Dict[str, Any]): Full configuration dictionary
        override_env_id (Optional[str]): Override the environment ID if provided
        
    Returns:
        Dict[str, Any]: Environment configuration settings
    """
    # Extract environment settings
    env_config = config.get('environment', {})
    obs_config = config.get('observation', {})
    
    # Basic environment settings
    env_settings = {
        'env_id': override_env_id if override_env_id else env_config.get('id', 'MiniGrid-Empty-8x8-v0'),
        'window_size': obs_config.get('window_size', 7),
        'cnn_keys': obs_config.get('cnn_keys', []),
        'mlp_keys': obs_config.get('mlp_keys', []),
        'max_episode_steps': env_config.get('max_episode_steps', 150),
    }
    
    # Always disable random spawn and flexible spawn
    env_settings.update({
        'use_random_spawn': False,
        'use_flexible_spawn': False,
    })
    
    # Override NoDeath settings to always apply it with a cost of 0
    env_settings.update({
        'use_no_death': True,
        'no_death_types': ('lava',),  # Default to lava as the no-death type
        'death_cost': 0.0,  # Override to 0 cost
    })
    
    # Always disable diagonal movement monitoring
    env_settings.update({
        'monitor_diagonal_moves': False,
        'diagonal_success_reward': 0.0,
        'diagonal_failure_penalty': 0.0,
    })
    
    return env_settings


def create_evaluation_env(env_settings: Dict[str, Any], seed: int = 42, override_rank: int = 0) -> gym.Env:
    """
    Create an environment for evaluation using the make_env function.
    
    Args:
        env_settings (Dict[str, Any]): Environment settings
        seed (int, optional): Random seed. Defaults to 42.
        override_rank (int, optional): Override the environment rank. Defaults to 0.
        
    Returns:
        gym.Env: The configured environment
    """
    # Create environment function
    env_fn = make_env(
        env_id=env_settings['env_id'],
        rank=override_rank,
        env_seed=seed,
        window_size=env_settings['window_size'],
        cnn_keys=env_settings['cnn_keys'],
        mlp_keys=env_settings['mlp_keys'],
        max_episode_steps=env_settings['max_episode_steps'],
        use_random_spawn=False,  # Always off
        use_flexible_spawn=False,  # Always off
        spawn_distribution_type='uniform',  # Default value (won't be used)
        spawn_distribution_params={},  # Default value (won't be used)
        exclude_goal_adjacent=False,  # Default value (won't be used)
        use_no_death=env_settings['use_no_death'],
        no_death_types=env_settings['no_death_types'],
        death_cost=env_settings['death_cost'],
        monitor_diagonal_moves=False,  # Always off
        diagonal_success_reward=0.0,  # Always 0
        diagonal_failure_penalty=0.0,  # Always 0
    )
    
    # Create the environment
    env = env_fn()
    
    print(f"Created evaluation environment: {env_settings['env_id']}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    return env


def extract_and_visualize_env(env, env_id=None, save_path="visualizations/env_layout.png", generate_plot=True):
    """
    Extract grid layout from the environment and visualize it.
    
    Args:
        env: The MiniGrid environment
        env_id: Environment ID string (for reference)
        save_path: Path to save the visualization
        generate_plot: If False, no matplotlib plot will be generated (default: True)
        
    Returns:
        np.ndarray: The extracted environment tensor
    """
    # Extract grid from environment
    env_tensor = extract_grid_from_env(env)
    
    # Print the tensor
    print_env_tensor(env_tensor)
    
    # Visualize the environment
    visualize_env_tensor(env_tensor, save_path, generate_plot)
    
    return env_tensor


def load_agent(agent_folder: str, config: Dict[str, Any]) -> Any:
    """
    Load the trained agent from the specified folder.
    
    Args:
        agent_folder (str): Path to the agent folder relative to Agent_Storage
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Any: The loaded agent model
    """
    # Get agent path
    if agent_folder.startswith("Agent_Storage/"):
        agent_path = os.path.join(agent_folder, "agent.zip")
    else:
        agent_path = os.path.join("Agent_Storage", agent_folder, "agent.zip")
    
    # Check if agent exists
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agent not found at {agent_path}")
    
    # Determine model type from config
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'DQN').upper()
    
    # Custom objects for loading
    custom_objects = {"features_extractor_class": CustomCombinedExtractor}
    
    # Load the appropriate model type
    if model_type == 'DQN':
        model = DQN.load(agent_path, custom_objects=custom_objects)
    elif model_type == 'PPO':
        model = PPO.load(agent_path, custom_objects=custom_objects)
    elif model_type == 'A2C':
        model = A2C.load(agent_path, custom_objects=custom_objects)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Loaded {model_type} agent from {agent_path}")
    return model 