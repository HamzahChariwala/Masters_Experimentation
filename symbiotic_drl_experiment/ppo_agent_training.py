import os
import random
import numpy as np
import torch
from typing import Tuple

import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgObsWrapper, OneHotPartialObsWrapper, NoDeath, DirectionObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from gymnasium.wrappers import RecordEpisodeStatistics

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def set_random_seeds(random_seed: int) -> None:
    """Sets random seeds for reproducibility in random, numpy, and torch."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def make_env(env_id: str, rank: int, env_seed: int, render_mode: str = None) -> callable:
    """
    Returns a function that creates an environment instance with the specified parameters.
    
    Args:
        env_id (str): Environment ID to be created.
        rank (int): The index of the environment instance. This helps in setting a unique seed.
        env_seed (int): The seed value; each environment is seeded with (env_seed + rank).
        render_mode (str, optional): The render mode to pass to gym.make (e.g., "rgb_array"). Defaults to None.
        
    Returns:
        A no-argument callable that creates the environment when called.
    """
    def _init():
        # Pass render_mode if specified, otherwise call gym.make without it.
        if render_mode is not None:
            env = gym.make(env_id, render_mode=render_mode)
        else:
            env = gym.make(env_id)
        env = RecordEpisodeStatistics(env)
        # Seed the environment; each instance gets a unique seed using env_seed + rank.
        env.reset(seed=env_seed + rank)
        # env = FullyObsWrapper(env)
        # env = NoDeath(env, no_death_types=("lava",), death_cost=-0.1)
        # env = OneHotPartialObsWrapper(env)
        # env = DirectionObsWrapper(env, type="slope")
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        return env
    return _init


def train_agent(env_id: str,
                model_dir: str,
                model_name: str,
                timesteps: int,
                random_seed: int,
                env_seed: int,
                n_envs: int,
                tensorboard_log: str) -> Tuple[PPO, DummyVecEnv]:
    """
    Trains an agent using PPO on a vectorized environment constructed from the provided parameters.
    
    Args:
        env_id (str): ID of the Gymnasium environment.
        model_dir (str): Directory to save the model.
        model_name (str): The filename for the saved model.
        timesteps (int): Number of timesteps for training.
        random_seed (int): Random seed for reproducibility.
        env_seed (int): The base seed for the environments.
        n_envs (int): Number of parallel environments.
        tensorboard_log (str): Directory to store TensorBoard log data.
        
    Returns:
        The trained PPO model and the vectorized environment.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    
    # Set seeds for reproducibility
    set_random_seeds(random_seed)
    
    # Create a list of environment creation functions, each with its own unique seed.
    env_fns = [make_env(env_id, rank, env_seed) for rank in range(n_envs)]
    env = DummyVecEnv(env_fns)
    
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log
    )
    
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, env


def load_agent(env_id: str,
               model_dir: str,
               model_name: str,
               random_seed: int) -> Tuple[PPO, DummyVecEnv]:
    """
    Loads a trained PPO agent from disk and associates it with a single instance environment.
    
    Args:
        env_id (str): ID of the Gymnasium environment.
        model_dir (str): Directory where the model is saved.
        model_name (str): The filename for the saved model.
        random_seed (int): Random seed used when training (to set up the environment).
        
    Returns:
        The loaded PPO model and the associated environment.
    """
    model_path = os.path.join(model_dir, model_name)
    
    # Set seeds for reproducibility
    set_random_seeds(random_seed)
    
    # Create a single environment instance for evaluation/loading.
    # (Note: When not visualizing, render_mode is not needed.)
    env = DummyVecEnv([make_env(env_id, 0, random_seed)])
    model = PPO.load(model_path, env=env)
    print("Model loaded successfully")
    
    return model, env


def visualize_agent(env_id: str,
                    model_dir: str,
                    model_name: str,
                    random_seed: int,
                    env_seed: int,
                    steps: int = 50,
                    interval: int = 1000,
                    test_seed: int = None) -> None:
    """
    Loads a trained agent and visualizes its behavior by generating an animation.
    
    The environment is re-created with the same wrappers as used during training,
    but with render_mode="rgb_array" so that frames can be captured.
    
    Args:
        env_id (str): ID of the Gymnasium environment.
        model_dir (str): Directory where the model is saved.
        model_name (str): The filename for the saved model.
        random_seed (int): Random seed for reproducibility.
        env_seed (int): The base seed for the environments during training.
        steps (int, optional): Number of steps to run for visualization. Defaults to 50.
        interval (int, optional): Delay between frames in the animation (milliseconds). Defaults to 1000.
        test_seed (int, optional): A separate seed for the test environment. If provided, it will be used instead of env_seed.
    
    Note:
        The animation shows the full state of the environment as it evolves with the agent's actions.
        If you wish to only display the agent's movement on a static background, you will need to create a 
        custom renderer that extracts and overlays the agent's position on a fixed background.
    """
    # Use the test_seed if provided; otherwise, fallback to env_seed.
    seed_to_use = test_seed if test_seed is not None else env_seed
    
    # Set random seeds for reproducibility.
    set_random_seeds(random_seed)
    
    # Create a single evaluation environment with render_mode enabled.
    vis_env = DummyVecEnv([make_env(env_id, 0, seed_to_use, render_mode="rgb_array")])
    
    # Load the trained model using the visualization environment.
    import os
    model_path = os.path.join(model_dir, model_name)
    model = PPO.load(model_path, env=vis_env)
    print("Agent loaded successfully for visualization.")
    
    # Reset the environment (only observation is returned).
    obs = vis_env.reset()
    frames = []
    
    # Run the agent for the specified number of steps.
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated = vis_env.step(action)
        # print(f"Step {step} observation:")
        # print(obs)
        # Get the rendered frame from the first (and only) environment.
        frame = vis_env.envs[0].render()
        frames.append(frame)
        if terminated[0] or truncated[0]:
            obs = vis_env.reset()
    
    vis_env.close()
    
    
    fig = plt.figure()
    im = plt.imshow(frames[0])
    
    def update(frame):
        im.set_array(frame)
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    # Example configuration parameters (these can be adjusted or passed from an external configuration)
    # ENV_ID = "MiniGrid-LavaCrossingS11N5-v0"
    # ENV_ID = "MiniGrid-Empty-8x8-v0"
    # ENV_ID = "MiniGrid-LavaGapS7-v0"
    ENV_ID = 'MiniGrid-LavaCrossingS9N2-v0'
    MODEL_DIR = "new_models"
    MODEL_NAME = "test_model_cnnpartial_crossingS9N2_5m"
    TIMESTEPS = 5_000_000
    RANDOM_SEED = 811
    ENV_SEED = 0
    N_ENVS = 8
    TENSORBOARD_LOG = "./ppo_tensorboard/"

    # make_env(ENV_ID, 0, 0)()
    
    # # Train the agent using the provided parameters.
    train_agent(ENV_ID, MODEL_DIR, MODEL_NAME, TIMESTEPS, RANDOM_SEED, ENV_SEED, N_ENVS, TENSORBOARD_LOG)
    
    # To visualize the trained agent, simply call the function:
    # visualize_agent(ENV_ID, MODEL_DIR, MODEL_NAME, RANDOM_SEED, ENV_SEED, steps=50, interval=1000, test_seed=1)


