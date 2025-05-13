import os
import random
import numpy as np
import torch
import yaml
import time
import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Add the parent directory to sys.path to ensure proper imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.vector import AsyncVectorEnv
from gymnasium.spaces import MultiDiscrete

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper
from minigrid.wrappers import RGBImgObsWrapper, OneHotPartialObsWrapper, NoDeath, DirectionObsWrapper

from stable_baselines3 import PPO, DQN, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import local modules using relative imports for better compatibility
from Agent_Training.TrainingTooling.TerminalCondition import CustomTerminationCallback
from Agent_Training.TrainingTooling.Tooling import visualize_agent_behavior, evaluate_with_timeout

from Environment_Tooling.BespokeEdits.CustomWrappers import (GoalAngleDistanceWrapper, 
                                            PartialObsWrapper, 
                                            ExtractAbstractGrid, 
                                            PartialRGBObsWrapper, 
                                            PartialGrayObsWrapper, 
                                            ForceFloat32)
from Environment_Tooling.BespokeEdits.FeatureExtractor import CustomCombinedExtractor, SelectiveObservationWrapper
from Environment_Tooling.BespokeEdits.ActionSpace import CustomActionWrapper
from Environment_Tooling.BespokeEdits.GymCompatibility import OldGymCompatibility

# Import environment generation functions
import Environment_Tooling.EnvironmentGeneration as Env

# Import our flexible spawn distribution wrapper
from Environment_Tooling.BespokeEdits.SpawnDistribution import FlexibleSpawnWrapper, DistributionMap
from Environment_Tooling.BespokeEdits.SpawnDistribution import SpawnDistributionCallback, EnhancedSpawnDistributionCallback, generate_final_visualizations

import Agent_Training.TrainingTooling.SpawnTooling as Spawn


def load_config(config_path=None, agent_folder=None):
    """
    Load configuration from YAML file.
    
    Parameters:
    ----------
    config_path : str, optional
        Path to the config file. If None, will use agent_folder path.
    agent_folder : str, optional
        Path to the agent folder relative to Agent_Storage. Can include subdirectories.
        
    Returns:
    -------
    dict
        Configuration dictionary
    """
    if agent_folder is not None:
        # Construct path to the agent's config file in Agent_Storage
        # Preserve the full path structure
        agent_path = os.path.join("Agent_Storage", agent_folder)
        config_path = os.path.join(agent_path, "config.yaml")
    elif config_path is None:
        # Default fallback
        config_path = "Agent_Training/config.yaml"
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_environment_vars(config):
    """Extract environment-related variables from config."""
    env_config = config.get('environment', {})
    reward_config = config.get('reward_function', {})
    
    # Environment variables
    env_vars = {
        'env_id': env_config.get('id', 'MiniGrid-Empty-8x8-v0'),
        'num_envs': env_config.get('num_envs', 1),
        'max_episode_steps': env_config.get('max_episode_steps'),
        'use_different_envs': env_config.get('use_different_envs', False),
    }
    
    # Reward function variables
    reward_vars = {
        'use_reward_function': reward_config.get('enabled', False),
        'reward_type': reward_config.get('type', 'linear'),
        'reward_x_intercept': reward_config.get('x_intercept', 100),
        'reward_y_intercept': reward_config.get('y_intercept', 1.0),
        'reward_transition_width': reward_config.get('transition_width', 10),
        'reward_verbose': reward_config.get('verbose', True),
        'debug_logging': reward_config.get('debug_logging', False),
        'count_lava_steps': reward_config.get('count_lava_steps', False),
        'lava_step_multiplier': reward_config.get('lava_step_multiplier', 2.0),
    }
    
    return {**env_vars, **reward_vars}


def extract_model_vars(config):
    """Extract model-related variables from config."""
    model_config = config.get('model', {})
    return {
        'model_type': model_config.get('type', 'DQN'),
        'policy': model_config.get('policy', 'MultiInputPolicy'),
        'use_mps': model_config.get('use_mps', False),
        'buffer_size': model_config.get('buffer_size', 100000),
        'learning_starts': model_config.get('learning_starts', 10000),
        'batch_size': model_config.get('batch_size', 64),
        'exploration_fraction': model_config.get('exploration_fraction', 0.05),
        'exploration_final_eps': model_config.get('exploration_final_eps', 0.05),
        'gamma': model_config.get('gamma', 0.6),
        'learning_rate': model_config.get('learning_rate', 0.00025),
        'train_freq': model_config.get('train_freq', 8),
        'target_update_interval': model_config.get('target_update_interval', 1000),
        'verbose': model_config.get('verbose', 1),
    }


def extract_experiment_vars(config):
    """Extract experiment-related variables from config."""
    exp_config = config.get('experiment', {})
    return {
        'experiment_name': exp_config.get('name', 'minigrid_agent_training'),
        'description': exp_config.get('description', ''),
        'version': exp_config.get('version', '1.0'),
        'log_dir': exp_config.get('output', {}).get('log_dir', './logs'),
        'model_save_path': exp_config.get('output', {}).get('model_save_path', 'model'),
        'total_timesteps': exp_config.get('output', {}).get('total_timesteps', 1000000),
    }


def extract_seed_vars(config):
    """Extract seed-related variables from config."""
    seed_config = config.get('seeds', {})
    return {
        'model_seed': seed_config.get('model', 42),
        'env_seed': seed_config.get('environment', 42),
        'eval_seed': seed_config.get('evaluation', 42),
        'seed_increment': seed_config.get('seed_increment', 1),
    }


def extract_all_vars(config_path=None, agent_folder=None):
    """
    Extract all variables from config file.
    
    Parameters:
    ----------
    config_path : str, optional
        Path to the config file.
    agent_folder : str, optional
        Name of the agent folder in Agent_Storage.
        
    Returns:
    -------
    dict
        Dictionary containing all configuration variables
    """
    config = load_config(config_path, agent_folder)
    return {
        **extract_environment_vars(config),
        **extract_model_vars(config),
        **extract_experiment_vars(config),
        **extract_seed_vars(config),
    }


def setup_directories(config: Dict[str, Any], agent_folder: str = None) -> Tuple[str, str, str]:
    """
    Set up directories for logs, performance tracking, etc.
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary
    agent_folder : str, optional
        Path to the agent folder relative to Agent_Storage
        
    Returns:
    -------
    tuple
        (log_dir, performance_log_dir, spawn_vis_dir)
    """
    # Determine the log directory
    if agent_folder:
        # Store logs in the agent's folder path in Agent_Storage
        # Preserve the full path structure
        agent_path = os.path.join("Agent_Storage", agent_folder)
        log_dir = os.path.join(agent_path, "logs")
    else:
        # Use the log directory from config as fallback
        log_dir = config['experiment']['output']['log_dir']
    
    # Create main log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create directory for performance logging
    performance_log_dir = os.path.join(log_dir, "performance")
    os.makedirs(performance_log_dir, exist_ok=True)
    
    # Create a dummy spawn_vis_dir (we don't need it but it's used in function signatures)
    spawn_vis_dir = ""
    
    return log_dir, performance_log_dir, spawn_vis_dir


def create_observation_params(config: Dict[str, Any], spawn_vis_dir: str) -> Dict[str, Any]:
    """
    Create observation parameters dictionary from config
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary
    spawn_vis_dir : str
        Directory for spawn distribution visualizations
        
    Returns:
    -------
    dict
        Observation parameters
    """
    # Extract values from config
    spawn_config = config['spawn']
    observation_config = config['observation']
    environment_config = config['environment']
    no_death_config = config['no_death']
    diagonal_config = config['diagonal_moves']
    reward_config = config['reward_function']  # Extract reward function config
    
    # Check if stage training or continuous transition is enabled
    if spawn_config['stage_training']['enabled']:
        use_stage_training = True
        stage_training_config = {
            'num_stages': spawn_config['stage_training']['num_stages'],
            'distributions': spawn_config['stage_training']['distributions'],
            'curriculum_proportion': spawn_config['stage_training'].get('curriculum_proportion', 1.0),
            'smooth_transitions': spawn_config['stage_training'].get('smooth_transitions', {'enabled': False})
        }
        
        # Add smooth transition parameters if enabled
        if stage_training_config['smooth_transitions']['enabled']:
            stage_training_config['smooth_transitions']['transition_proportion'] = (
                spawn_config['stage_training']['smooth_transitions'].get('transition_proportion', 0.2)
            )
        
        # Process relative durations for stages
        if 'distributions' in stage_training_config:
            # Calculate total relative duration
            total_relative_duration = 0
            for dist in stage_training_config['distributions']:
                total_relative_duration += dist.get('relative_duration', 1.0)
            
            # Store this for calculations
            stage_training_config['total_relative_duration'] = total_relative_duration
    else:
        use_stage_training = False
        stage_training_config = None
    
    if spawn_config['continuous_transition']['enabled']:
        use_continuous_transition = True
        continuous_transition_config = {
            'target_type': spawn_config['continuous_transition']['target_type'],
            'rate': spawn_config['continuous_transition']['rate']
        }
    else:
        use_continuous_transition = False
        continuous_transition_config = None
    
    # Build observation parameters dictionary
    observation_params = {
        # We'll handle env_id separately to avoid duplication in function calls
        "window_size": observation_config['window_size'],
        "cnn_keys": observation_config['cnn_keys'],
        "mlp_keys": observation_config['mlp_keys'],
        "use_random_spawn": False,  # Disable default random spawn in favor of our flexible spawn
        "exclude_goal_adjacent": spawn_config['exclude_goal_adjacent'],
        "use_no_death": no_death_config['enabled'],
        "no_death_types": tuple(no_death_config['types']),
        "death_cost": no_death_config['cost'],
        "max_episode_steps": environment_config['max_episode_steps'],
        "monitor_diagonal_moves": diagonal_config['monitor'],
        "diagonal_success_reward": diagonal_config['success_reward'],
        "diagonal_failure_penalty": diagonal_config['failure_penalty'],
        "use_flexible_spawn": spawn_config['use_flexible_spawn'],
        "spawn_distribution_type": spawn_config['distribution_type'],
        "spawn_distribution_params": spawn_config['distribution_params'],
        "use_stage_training": use_stage_training,
        "stage_training_config": stage_training_config,
        "use_continuous_transition": use_continuous_transition,
        "continuous_transition_config": continuous_transition_config,
        # Add reward function parameters
        "use_reward_function": reward_config['enabled'],
        "reward_type": reward_config['type'],
        "reward_x_intercept": reward_config['x_intercept'],
        "reward_y_intercept": reward_config['y_intercept'],
        "reward_transition_width": reward_config['transition_width'],
        "reward_verbose": reward_config['verbose'],
        "debug_logging": reward_config.get('debug_logging', False),
        "count_lava_steps": reward_config.get('count_lava_steps', False),
        "lava_step_multiplier": reward_config.get('lava_step_multiplier', 2.0),
        # Keep the spawn visualization directory
        "spawn_vis_dir": spawn_vis_dir,
        "spawn_vis_frequency": 10000  # Default value if not specified elsewhere
    }
    
    return observation_params


def create_policy_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create policy keyword arguments from config
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary
        
    Returns:
    -------
    dict
        Policy keyword arguments
    """
    extractor_config = config['model']['features_extractor']
    
    policy_kwargs = {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {
            "features_dim": extractor_config['features_dim'],
            "cnn_num_layers": extractor_config['cnn']['num_layers'],
            "cnn_channels": extractor_config['cnn']['channels'],
            "cnn_kernels": extractor_config['cnn']['kernels'],
            "cnn_strides": extractor_config['cnn']['strides'],
            "cnn_paddings": extractor_config['cnn']['paddings'],
            "mlp_num_layers": extractor_config['mlp']['num_layers'],
            "mlp_hidden_sizes": extractor_config['mlp']['hidden_sizes'],
        }
    }
    
    return policy_kwargs


def set_random_seeds(config: Dict[str, Any]):
    """
    Set random seeds for reproducibility
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary
    """
    model_seed = config['seeds']['model']
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)


def print_training_info(config: Dict[str, Any], observation_params: Dict[str, Any]):
    """
    Print training information
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary
    observation_params : dict
        Observation parameters
    """
    env_config = config['environment']
    seed_config = config['seeds']
    eval_config = config['evaluation']['training']
    
    # Print training setup
    print("\n====== TRAINING SETUP ======")
    print(f"Environment: {env_config['id']}")
    print(f"Number of parallel environments: {env_config['num_envs']}")
    print(f"Model seed: {seed_config['model']}")
    print(f"Environment seed: {seed_config['environment']}")
    print(f"Evaluation seed: {seed_config['evaluation']}")
    print(f"Evaluation timeout: {eval_config['timeout']} seconds")
    print(f"Max episode steps: {observation_params['max_episode_steps']}")
    print(f"Diagonal success reward: {observation_params['diagonal_success_reward']}")
    print(f"Diagonal failure penalty: {observation_params['diagonal_failure_penalty']}")
    print(f"NoDeath wrapper: {observation_params['use_no_death']}")
    
    if observation_params['use_no_death']:
        print(f"  - Death types: {list(observation_params['no_death_types'])}")
        print(f"  - Death cost: {observation_params['death_cost']}")
    
    # Print reward function information
    print(f"Reward function wrapper: {observation_params['use_reward_function']}")
    if observation_params['use_reward_function']:
        print(f"  - Type: {observation_params['reward_type']}")
        print(f"  - X-intercept: {observation_params['reward_x_intercept']}")
        print(f"  - Y-intercept: {observation_params['reward_y_intercept']}")
        print(f"  - Transition width: {observation_params['reward_transition_width']}")
        print(f"  - Verbose: {observation_params['reward_verbose']}")
        print(f"  - Count lava steps: {observation_params['count_lava_steps']}")
        if observation_params['count_lava_steps']:
            print(f"  - Lava step multiplier: {observation_params['lava_step_multiplier']}")
    
    # Create a copy of observation parameters with env_id for spawn distribution info
    spawn_params = observation_params.copy()
    spawn_params['env_id'] = env_config['id']
    
    # Print spawn distribution info
    if observation_params['use_flexible_spawn']:
        Spawn.print_spawn_distribution_info(
            spawn_params, 
            config['experiment']['output']['total_timesteps']
        )


def create_model(config: Dict[str, Any], env, policy_kwargs: Dict[str, Any], log_dir: str) -> DQN:
    """
    Create and configure the model
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary
    env : gym.Env
        Training environment
    policy_kwargs : dict
        Policy keyword arguments
    log_dir : str
        Directory for logging
        
    Returns:
    -------
    model
        Initialized model
    """
    model_config = config['model']
    
    # Determine device
    use_mps = model_config['use_mps']
    device = torch.device("mps" if use_mps and torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Select model type
    model_type = model_config['type']
    model_class = DQN  # Default
    if model_type == "PPO":
        model_class = PPO
    elif model_type == "DDPG":
        model_class = DDPG
    
    # Create model
    model = model_class(
        model_config['policy'],
        env,
        policy_kwargs=policy_kwargs,
        buffer_size=model_config['buffer_size'],
        learning_starts=model_config['learning_starts'],
        batch_size=model_config['batch_size'],
        exploration_fraction=model_config['exploration_fraction'],
        exploration_final_eps=model_config['exploration_final_eps'],
        gamma=model_config['gamma'],
        learning_rate=model_config['learning_rate'],
        train_freq=model_config['train_freq'],
        target_update_interval=model_config['target_update_interval'],
        verbose=model_config['verbose'],
        tensorboard_log=log_dir,
        device=device
    )
    
    return model


def create_eval_environments(config: Dict[str, Any], observation_params: Dict[str, Any]) -> List[gym.Env]:
    """
    Create evaluation environments
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary
    observation_params : dict
        Observation parameters
        
    Returns:
    -------
    list
        List of evaluation environments
    """
    env_config = config['environment']
    eval_config = config['evaluation']['training']
    seed_config = config['seeds']
    
    # Get the seed increment (default to 1 if not specified)
    seed_increment = seed_config.get('seed_increment', 1)
    
    # Create modified observation parameters for evaluation
    eval_params = observation_params.copy()
    eval_params["use_flexible_spawn"] = False  # Disable flexible spawn for evaluation
    
    # Create environments
    num_eval_envs = eval_config['num_envs']
    eval_envs = []
    
    print("\n====== CREATING EVALUATION ENVIRONMENTS ======")
    print(f"Creating {num_eval_envs} evaluation environments with different seeds")
    print(f"Base evaluation seed: {seed_config['evaluation']}, increment: {seed_increment}")
    
    for i in range(num_eval_envs):
        # Use different seeds for each evaluation environment
        eval_seed = seed_config['evaluation'] + (i * seed_increment)
        eval_env = Env.make_eval_env(
            env_id=env_config['id'], 
            seed=eval_seed,
            **eval_params
        )
        eval_envs.append(eval_env)
    
    print(f"Created {len(eval_envs)} evaluation environments")
    print("=============================================\n")
    
    return eval_envs


def create_callbacks(config: Dict[str, Any], eval_envs: List[gym.Env], 
                     performance_log_dir: str, spawn_vis_dir: str):
    """
    Create training callbacks
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary
    eval_envs : list
        List of evaluation environments
    performance_log_dir : str
        Directory for performance logging
    spawn_vis_dir : str
        Directory for spawn distribution visualization
        
    Returns:
    -------
    list
        List of callbacks
    """
    eval_config = config['evaluation']['training']
    
    # Create termination callback
    termination_callback = CustomTerminationCallback(
        eval_envs=eval_envs,
        check_freq=eval_config['check_freq'],
        target_reward_threshold=eval_config['target_reward_threshold'],
        max_runtime=eval_config['max_runtime'],
        n_eval_episodes=eval_config['n_eval_episodes'],
        eval_timeout=eval_config['timeout'],
        log_dir=performance_log_dir,
        verbose=1,
        disable_early_stopping=eval_config.get('disable_early_stopping', False)
    )
    
    return [termination_callback]


def run_training(config: Dict[str, Any], model, callbacks, agent_folder=None):
    """
    Run model training
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary
    model : stable_baselines3 model
        Model to train
    callbacks : list
        List of callbacks
    agent_folder : str, optional
        Path to the agent folder relative to Agent_Storage
        
    Returns:
    -------
    model
        Trained model
    """
    total_timesteps = config['experiment']['output']['total_timesteps']
    eval_config = config['evaluation']['training']
    termination_callback = callbacks[0]
    
    # Determine where to save the model - prioritize agent_folder if provided
    if agent_folder:
        agent_path = os.path.join("Agent_Storage", agent_folder)
        model_save_path = os.path.join(agent_path, "agent")
    else:
        # Fallback to config path if no agent_folder
        model_save_path = config['experiment']['output']['model_save_path']
    
    # Print training start message
    print("\n====== STARTING TRAINING ======")
    print(f"Target timesteps: {total_timesteps}")
    print(f"Evaluation frequency: Every {termination_callback.check_freq} steps")
    print(f"Evaluation environments: {eval_config['num_envs']}")
    print(f"Evaluation episodes per environment: {eval_config['n_eval_episodes']}")
    print(f"Total evaluation episodes: {eval_config['num_envs'] * eval_config['n_eval_episodes']}")
    print(f"Evaluation timeout: {eval_config['timeout']} seconds")
    print(f"Model will be saved to: {model_save_path}")
    print("==============================\n")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=f"{config['model']['type']}_MiniGrid",
        callback=callbacks
    )
    
    # Save the model
    model.save(model_save_path)
    
    return model


def run_final_evaluation(config: Dict[str, Any], model, observation_params: Dict[str, Any]):
    """
    Run final evaluation
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary
    model : stable_baselines3 model
        Trained model
    observation_params : dict
        Observation parameters
        
    Returns:
    -------
    tuple
        (all_rewards, all_lengths)
    """
    env_config = config['environment']
    eval_config = config['evaluation']['final']
    seed_config = config['seeds']
    
    # Get the seed increment (default to 1 if not specified)
    seed_increment = seed_config.get('seed_increment', 1)
    
    # Create modified observation parameters for evaluation
    eval_params = observation_params.copy()
    eval_params["use_flexible_spawn"] = False  # Disable flexible spawn for evaluation
    
    # Print final evaluation message
    print("\n====== TRAINING COMPLETE ======")
    print("Running final evaluation...")
    print("==============================\n")
    
    # Final evaluation across multiple environments
    num_final_eval_envs = eval_config['num_envs']
    episodes_per_env = eval_config['episodes_per_env']
    final_eval_timeout = eval_config['timeout']
    
    print(f"Evaluating on {num_final_eval_envs} environments, {episodes_per_env} episodes each")
    print(f"Timeout per environment: {final_eval_timeout} seconds")
    print(f"Base evaluation seed: {seed_config['evaluation'] + 2000}, increment: {seed_increment}")
    
    all_rewards = []
    all_lengths = []
    
    for i in range(num_final_eval_envs):
        # Create a fresh evaluation environment with a different seed
        final_eval_seed = seed_config['evaluation'] + 2000 + (i * seed_increment)
        final_eval_env = Env.make_eval_env(
            env_id=env_config['id'], 
            seed=final_eval_seed,
            **eval_params
        )
        
        # Use our timeout evaluation function
        env_rewards, env_lengths, error = evaluate_with_timeout(
            model=model,
            env=final_eval_env,
            n_eval_episodes=episodes_per_env,
            timeout=final_eval_timeout,
            deterministic=True
        )
        
        # Handle evaluation result
        if error:
            print(f"Environment {i+1}/{num_final_eval_envs} (Seed: {final_eval_seed}): {error}")
            final_eval_env.close()
            continue
            
        # Store results
        all_rewards.extend(env_rewards)
        all_lengths.extend(env_lengths)
        
        # Print per-environment results
        print(f"Environment {i+1}/{num_final_eval_envs} (Seed: {final_eval_seed}):")
        print("  Rewards   : mean %.3f  std %.3f  min %.3f  max %.3f"
            % (np.mean(env_rewards),
                np.std(env_rewards),
                np.min(env_rewards),
                np.max(env_rewards)))
        print("  Lengths   : mean %.1f  std %.1f  min %d    max %d"
            % (np.mean(env_lengths),
                np.std(env_lengths),
                np.min(env_lengths),
                np.max(env_lengths)))
        
        # Close the environment
        final_eval_env.close()
    
    # Print overall results
    if len(all_rewards) > 0:
        print("\n====== FINAL RESULTS (All Environments) ======")
        print(f"Total episodes evaluated: {len(all_rewards)}")
        print("Rewards   : mean %.3f  std %.3f  min %.3f  max %.3f"
            % (np.mean(all_rewards),
                np.std(all_rewards),
                np.min(all_rewards),
                np.max(all_rewards)))
        print("Lengths   : mean %.1f  std %.1f  min %d    max %d"
            % (np.mean(all_lengths),
                np.std(all_lengths),
                np.min(all_lengths),
                np.max(all_lengths)))
    else:
        print("\n====== FINAL RESULTS ======")
        print("No valid evaluation results obtained")
    print("==========================\n")
    
    return all_rewards, all_lengths


def main(config_path=None, agent_folder=None):
    """
    Main function for running the training process from a config file.
    
    Parameters:
    ----------
    config_path : str, optional
        Path to the config file.
    agent_folder : str, optional
        Name of the agent folder in Agent_Storage.
    """
    # Parse command line arguments if not provided directly
    if config_path is None and agent_folder is None:
        parser = argparse.ArgumentParser(description='Run training with configuration')
        parser.add_argument('--config', type=str, help='Path to config file')
        parser.add_argument('--path', type=str, required=True, help='Agent folder name in Agent_Storage')
        args = parser.parse_args()
        
        config_path = args.config
        agent_folder = args.path
    
    # Ensure we have an agent folder
    if agent_folder is None:
        print("Error: No agent folder specified. Please use --path parameter.")
        sys.exit(1)
    
    # Load config
    config = load_config(config_path, agent_folder)
    
    # Setup directories - pass agent_folder to use for logs
    log_dir, performance_log_dir, spawn_vis_dir = setup_directories(config, agent_folder)
    
    # Create observation parameters
    observation_params = create_observation_params(config, spawn_vis_dir)
    
    # Create policy kwargs
    policy_kwargs = create_policy_kwargs(config)
    
    # Set random seeds
    set_random_seeds(config)
    
    # Print training information
    print_training_info(config, observation_params)
    
    # Create environment
    env = Env.make_parallel_env(
        env_id=config['environment']['id'],
        num_envs=config['environment']['num_envs'],
        env_seed=config['seeds']['environment'],
        use_different_envs=config['environment'].get('use_different_envs', False),
        **observation_params
    )
    
    # Create model
    model = create_model(config, env, policy_kwargs, log_dir)
    
    # Create evaluation environments
    eval_envs = create_eval_environments(config, observation_params)
    
    # Create callbacks
    callbacks = create_callbacks(config, eval_envs, performance_log_dir, spawn_vis_dir)
    
    # Run training
    model = run_training(config, model, callbacks, agent_folder)
    
    # Run final evaluation
    run_final_evaluation(config, model, observation_params)
    
    # Display final performance comparison if available
    termination_callback = callbacks[0]
    if hasattr(termination_callback, "performance_tracker"):
        print("\n====== TRAINING VS EVALUATION PERFORMANCE ======")
        print("Generating final performance comparison plots...")
        # Generate final plots with the show option set to False (we're in headless mode)
        termination_callback.performance_tracker.plot_performance(save=True, show=False)
        print(f"Performance plots saved to: {performance_log_dir}")
        print("===============================================\n")


if __name__ == "__main__":
    main() 