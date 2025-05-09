import os
import random
import numpy as np
import torch
import yaml
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List

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

from AgentTraining.TerminalCondition import CustomTerminationCallback
from AgentTraining.Tooling import visualize_agent_behavior, evaluate_with_timeout

from EnvironmentEdits.BespokeEdits.CustomWrappers import (GoalAngleDistanceWrapper, 
                                            PartialObsWrapper, 
                                            ExtractAbstractGrid, 
                                            PartialRGBObsWrapper, 
                                            PartialGrayObsWrapper, 
                                            ForceFloat32)
from EnvironmentEdits.BespokeEdits.FeatureExtractor import CustomCombinedExtractor, SelectiveObservationWrapper
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.BespokeEdits.GymCompatibility import OldGymCompatibility

# Import environment generation functions
import EnvironmentEdits.EnvironmentGeneration as Env

# Import our flexible spawn distribution wrapper
from SpawnDistributions.visualization import SpawnDistributionCallback, EnhancedSpawnDistributionCallback, generate_final_visualizations
from EnvironmentEdits.BespokeEdits.SpawnDistribution import FlexibleSpawnWrapper, DistributionMap

import AgentTraining.SpawnTooling as Spawn


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file
    
    Parameters:
    ----------
    config_path : str
        Path to the YAML configuration file
        
    Returns:
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Set up directories for logs, performance tracking, etc.
    
    Parameters:
    ----------
    config : dict
        Configuration dictionary
        
    Returns:
    -------
    tuple
        (log_dir, performance_log_dir, spawn_vis_dir)
    """
    # Create main log directory
    log_dir = config['experiment']['output']['log_dir']
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
    
    # Check if stage training or continuous transition is enabled
    if spawn_config['stage_training']['enabled']:
        use_stage_training = True
        stage_training_config = {
            'num_stages': spawn_config['stage_training']['num_stages'],
            'distributions': spawn_config['stage_training']['distributions']
        }
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
        "continuous_transition_config": continuous_transition_config
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
        verbose=1
    )
    
    return [termination_callback]


def run_training(config: Dict[str, Any], model, callbacks):
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
        
    Returns:
    -------
    model
        Trained model
    """
    total_timesteps = config['experiment']['output']['total_timesteps']
    model_save_path = config['experiment']['output']['model_save_path']
    eval_config = config['evaluation']['training']
    termination_callback = callbacks[0]
    
    # Print training start message
    print("\n====== STARTING TRAINING ======")
    print(f"Target timesteps: {total_timesteps}")
    print(f"Evaluation frequency: Every {termination_callback.check_freq} steps")
    print(f"Evaluation environments: {eval_config['num_envs']}")
    print(f"Evaluation episodes per environment: {eval_config['n_eval_episodes']}")
    print(f"Total evaluation episodes: {eval_config['num_envs'] * eval_config['n_eval_episodes']}")
    print(f"Evaluation timeout: {eval_config['timeout']} seconds")
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


def main(config_path='config.yaml'):
    """
    Main function
    
    Parameters:
    ----------
    config_path : str
        Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up directories
    log_dir, performance_log_dir, spawn_vis_dir = setup_directories(config)
    
    # Create observation parameters
    observation_params = create_observation_params(config, spawn_vis_dir)
    
    # Set random seeds
    set_random_seeds(config)
    
    # Print training information
    print_training_info(config, observation_params)
    
    # Create training environment
    env = Env.make_parallel_env(
        env_id=config['environment']['id'],
        num_envs=config['environment']['num_envs'],
        env_seed=config['seeds']['environment'],
        use_different_envs=config['environment']['use_different_envs'],
        **observation_params
    )
    
    # Create policy kwargs
    policy_kwargs = create_policy_kwargs(config)
    
    # Create model
    model = create_model(config, env, policy_kwargs, log_dir)
    
    # Create evaluation environments
    eval_envs = create_eval_environments(config, observation_params)
    
    # Create callbacks
    callbacks = create_callbacks(config, eval_envs, performance_log_dir, spawn_vis_dir)
    
    # Run training
    model = run_training(config, model, callbacks)
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Train an agent using a YAML configuration file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    main(args.config) 