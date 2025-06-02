import os
import sys
import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Additional import for hyperparameter optimization
import optuna

# Configuration
ENV_ID = "MiniGrid-DoorKey-8x8-v0"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_doorkey")
TUNING_MODEL_DIR = "tuned_models"  # Folder to save models from hyperparameter tuning
TUNING_RESULTS_FILE = "tuning_results.txt"  # File to store the tuning results
TIMESTEPS = 1_000_000  # Regular training timesteps
TUNING_TIMESTEPS = 500_000  # Use fewer timesteps for each tuning trial evaluation
RANDOM_SEED = 811

# Set the random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def make_env():
    # Create and seed the environment
    env = gym.make(ENV_ID, render_mode="rgb_array")
    # In gymnasium, you can reset with a seed:
    env.reset(seed=RANDOM_SEED)
    env = RGBImgPartialObsWrapper(env)  # Convert observation to a partial RGB image.
    env = ImgObsWrapper(env)  # Simplify the observation to just the image array.
    return env

def train_agent():
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Pass the RANDOM_SEED to the vectorized environment so that each instance is seeded.
    env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv, seed=RANDOM_SEED)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/"
    )

    model.learn(total_timesteps=TIMESTEPS)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_agent(path = MODEL_PATH):
    env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv, seed=RANDOM_SEED)
    model = PPO.load(path, env=env)
    print("Model loaded successfully")
    return model, env

#########################################
# Logging Function for Trial Results    #
#########################################

def log_trial_result(message: str, file_path: str = TUNING_RESULTS_FILE):
    """Append the provided message to the tuning results file."""
    with open(file_path, "a") as f:
        f.write(message + "\n")

########################################
# Hyperparameter Optimization Function #
########################################

def objective(trial):
    """
    Objective function for hyperparameter optimization. This function:
      - Samples hyperparameters using Optuna.
      - Creates the environment, initializes the PPO agent with these hyperparameters.
      - Trains the agent for TUNING_TIMESTEPS.
      - Evaluates the trained agent on a small number of evaluation episodes.
      - Saves the trial’s agent with a filename that encodes the hyperparameter values.
      - Logs the results to a text file.
    """
    # Suggest hyperparameters over wide ranges
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    vf_coef = trial.suggest_float("vf_coef", 0.2, 1.0)

    # Create the vectorized environment with the specified seed.
    env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv, seed=RANDOM_SEED)

    # Create the PPO agent with sampled hyperparameters
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        clip_range=clip_range,
        n_steps=n_steps,
        ent_coef=ent_coef,
        batch_size=batch_size,
        gae_lambda=gae_lambda,
        vf_coef=vf_coef,
        verbose=0  # Reduce logging during tuning
    )

    # Train the agent for a reduced number of timesteps for rapid evaluation
    model.learn(total_timesteps=TUNING_TIMESTEPS)

    # Evaluate the agent's performance over a few episodes
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, render=False)
    
    # Save the agent with a file name encoding the hyperparameters
    os.makedirs(TUNING_MODEL_DIR, exist_ok=True)
    model_filename = (
        f"ppo_lr{learning_rate:.1e}_gamma{gamma:.3f}_clip{clip_range:.3f}_nsteps{n_steps}"
        f"_ent{ent_coef:.1e}_bs{batch_size}_gae{gae_lambda:.3f}_vf{vf_coef:.3f}.zip"
    )
    save_path = os.path.join(TUNING_MODEL_DIR, model_filename)
    model.save(save_path)
    result_message = (
        f"Saved model to {save_path} with mean reward {mean_reward:.2f} and std reward {std_reward:.2f} | "
        f"hyperparameters: learning_rate={learning_rate:.1e}, gamma={gamma:.3f}, clip_range={clip_range:.3f}, "
        f"n_steps={n_steps}, ent_coef={ent_coef:.1e}, batch_size={batch_size}, gae_lambda={gae_lambda:.3f}, "
        f"vf_coef={vf_coef:.3f}"
    )
    
    # Print the result and log it to the file.
    print(result_message)
    log_trial_result(result_message)

    # Return the mean reward; Optuna will try to maximize this value.
    return mean_reward

def optimize_hyperparameters(num_trials=20):
    """
    Run Optuna hyperparameter optimization over the PPO parameters.
    You can adjust the number of trials. The best hyperparameters (by average reward)
    will be shown at the end.
    """
    # Clear previous tuning results (if any)
    if os.path.exists(TUNING_RESULTS_FILE):
        os.remove(TUNING_RESULTS_FILE)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)
    print("Hyperparameter optimization complete.")
    print("Best trial:")
    best_trial = study.best_trial
    best_message = (
        f"Best trial value: {best_trial.value:.2f}\n"
        + "\n".join([f"{key}: {value}" for key, value in best_trial.params.items()])
    )
    print(best_message)
    log_trial_result("Hyperparameter optimization complete.")
    log_trial_result("Best trial:")
    log_trial_result(best_message)


def test_all_agents(num_episodes=10):
    """
    Iterate through every trained agent in the 'tuned_models' folder.
    For each agent, run 10 evaluation episodes (using the same random seed for repeatability)
    and compute the reward from each episode. Append the agent's filename along with the list of rewards,
    as well as the mean and standard deviation of the rewards, to the tuning_results.txt file.
    """
    # Folder and file names (assumes these constants exist in the current script)
    tuned_models_dir = "tuned_models"
    results_file = "tuning_results.txt"

    # Iterate over all .zip files in the tuned_models folder
    for filename in os.listdir(tuned_models_dir):
        if not filename.endswith(".zip"):
            continue
        model_path = os.path.join(tuned_models_dir, filename)

        # Create the environment with the same RANDOM_SEED using the make_env function.
        # (This creates a vectorized environment with a single instance.)
        env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv, seed=RANDOM_SEED)

        # Load the agent
        model = PPO.load(model_path, env=env)

        # List to store per-episode rewards
        episode_rewards = []

        # Run evaluation for a fixed number of episodes
        for episode in range(num_episodes):
            # Reset the environment; this reset returns a tuple (obs, info)
            obs = env.reset()
            total_reward = 0.0
            done = [False]  # Because env is vectorized, done is a list/array
            truncated = [False]

            # Run the agent until the episode ends (either done or truncated)
            while not (done[0] or truncated[0]):
                action, _ = model.predict(obs, deterministic=True)
                # In gymnasium, step returns: (obs, reward, done, truncated, info)
                obs, reward, done, truncated = env.step(action)
                # Since we have one environment, reward is a list with one element
                total_reward += reward[0]
            episode_rewards.append(total_reward)

        # Compute mean and standard deviation of the rewards
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        # Create a result string with the model's filename, individual episode rewards, mean, and std.
        result_str = (
            f"Model: {filename} | Episode rewards: {episode_rewards} | "
            f"Mean reward: {mean_reward:.2f} | Std reward: {std_reward:.2f}\n"
        )

        # Append the result to the tuning_results.txt file.
        with open(results_file, "a") as f:
            f.write(result_str)

        # Optionally, print the result
        print(result_str)

        # Close the environment to free up resources
        env.close()

import matplotlib.pyplot as plt
import numpy as np

def get_agent_pos(env, seen=None):
    """
    Recursively search for an attribute named 'agent_pos' in the environment's wrappers.
    For vectorized environments (with an 'envs' attribute), automatically use the first environment.
    Uses a 'seen' set to avoid infinite recursion.
    
    Returns the agent position (as a tuple or array) if found; otherwise, returns None.
    """
    # For vectorized environments, grab the first one.
    if hasattr(env, "envs"):
        base_env = env.envs[0]
        if hasattr(base_env, "agent_pos"):
            return base_env.agent_pos
        if hasattr(base_env, "unwrapped") and hasattr(base_env.unwrapped, "agent_pos"):
            return base_env.unwrapped.agent_pos

    if seen is None:
        seen = set()
    if id(env) in seen:
        return None
    seen.add(id(env))
    if hasattr(env, "agent_pos"):
        return env.agent_pos
    if hasattr(env, "env"):
        candidate = get_agent_pos(env.env, seen)
        if candidate is not None:
            return candidate
    if hasattr(env, "unwrapped"):
        candidate = get_agent_pos(env.unwrapped, seen)
        if candidate is not None:
            return candidate
    return None

def get_grid_size(env, default=8, seen=None):
    """
    Recursively search for a 'width' attribute to determine the grid size.
    For vectorized environments (with an 'envs' attribute), use the first environment.
    Uses a 'seen' set to avoid infinite recursion.
    
    Returns the grid size if found; otherwise, returns the default.
    """
    if hasattr(env, "envs"):
        base_env = env.envs[0]
        if hasattr(base_env, "width"):
            return base_env.width
        if hasattr(base_env, "unwrapped") and hasattr(base_env.unwrapped, "width"):
            return base_env.unwrapped.width

    if seen is None:
        seen = set()
    if id(env) in seen:
        return default
    seen.add(id(env))
    if hasattr(env, "width"):
        return env.width
    if hasattr(env, "env"):
        candidate = get_grid_size(env.env, default, seen)
        if candidate != default:
            return candidate
    if hasattr(env, "unwrapped"):
        candidate = get_grid_size(env.unwrapped, default, seen)
        if candidate != default:
            return candidate
    return default

def visualize_agent_path(model, env, num_episodes=1):
    """
    Runs the agent for a given number of episodes, records its position at each step
    (if accessible via the base environment), and visualizes the path on a grid.

    Parameters:
        model: The trained agent (e.g., a PPO model)
        env: The vectorized environment created using your make_env() function.
        num_episodes: Number of episodes to visualize.
    """
    for episode in range(num_episodes):
        # Reset the environment (assuming the reset returns just the observation)
        obs = env.reset()
        path = []
        
        # Record the initial position.
        pos = get_agent_pos(env)
        if pos is not None:
            path.append(np.array(pos))
        else:
            print("Warning: Cannot find agent position attribute at the initial reset.")
        
        # Run the episode until termination.
        while True:
            action, _ = model.predict(obs, deterministic=True)
            result = env.step(action)
            # Handle both 5-element and 4-element returns.
            if len(result) == 5:
                obs, reward, done, truncated, info = result
            elif len(result) == 4:
                obs, reward, done, info = result
                truncated = False
            else:
                raise ValueError("Unexpected number of values returned from env.step()")
            
            # Convert done and truncated into booleans.
            done_val = done[0] if isinstance(done, (list, tuple)) else done
            truncated_val = truncated[0] if isinstance(truncated, (list, tuple)) else truncated
            
            # Record the agent's position.
            pos = get_agent_pos(env)
            if pos is not None:
                path.append(np.array(pos))
            
            # If the episode is finished, break.
            if done_val or truncated_val:
                break
        
        if len(path) == 0:
            print(f"No agent positions recorded in episode {episode+1}; cannot plot path.")
            continue
        
        # Convert the recorded positions to a numpy array.
        path = np.array(path)
        
        # Verify that the path data is in a 2D array.
        if path.ndim != 2 or path.shape[1] < 2:
            print(f"Path data is not in the expected shape: {path.shape}")
            continue
        
        grid_size = get_grid_size(env, default=8)

        print(f"Episode {episode+1} steps taken: {len(path)}")
        
        # Create the plot.
        plt.figure(figsize=(6, 6))
        plt.imshow(np.zeros((grid_size, grid_size)), cmap="gray", extent=[0, grid_size, grid_size, 0])
        # Assuming positions are given as (row, col)
        plt.plot(path[:, 1] + 0.5, path[:, 0] + 0.5, marker="o", color="red", linestyle="--")
        plt.title(f"Agent Path - Episode {episode+1}")
        plt.xlabel("Grid Column")
        plt.ylabel("Grid Row")
        plt.xlim(0, grid_size)
        plt.ylim(grid_size, 0)
        plt.grid(True)
        plt.show()


def debug_render_agent_path(model, env):
    """
    Runs an episode and renders the environment frame-by-frame for debugging.
    This uses the base environment's render() (via unwrapped.render()) so that
    you see the full visuals (walls, keys, doors, etc.) as intended.
    
    Assumptions:
      - The environment is created with render_mode="rgb_array".
      - The environment is vectorized with n_envs=1.
    
    Parameters:
      model: The trained agent (e.g., a PPO model).
      env: The vectorized environment created by your make_env() function.
    """
    # For vectorized environments, get the underlying base environment.
    base_env = env.envs[0] if hasattr(env, "envs") else env
    
    # Reset the environment via the vectorized wrapper.
    obs = env.reset()
    
    # Initialize termination flags.
    done = False
    truncated = False
    step_count = 0

    # Enable interactive mode in matplotlib.
    plt.ion()
    fig, ax = plt.subplots()
    im = None

    while not (done or truncated):
        # Render from the base (unwrapped) environment.
        frame = base_env.unwrapped.render()  # no mode argument here
        # Convert frame to a proper numeric numpy array.
        frame = np.array(frame, dtype=np.uint8)
        
        # On the first frame, create the image; thereafter update it.
        if im is None:
            im = ax.imshow(frame)
        else:
            im.set_data(frame)
        
        ax.set_title(f"Step {step_count}")
        plt.pause(0.001)  # Pause briefly for the frame to update

        # Have the agent predict an action.
        action, _ = model.predict(obs, deterministic=True)

        # Take a step and handle different return lengths.
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        elif len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            raise ValueError("Unexpected number of values returned from env.step()")
        
        # For vectorized environments, done or truncated might be a list;
        # get the first element if that is the case.
        done_val = done[0] if isinstance(done, (list, tuple)) else done
        truncated_val = truncated[0] if isinstance(truncated, (list, tuple)) else truncated
        
        if done_val or truncated_val:
            break
        
        step_count += 1

    # Turn off interactive mode and show the final frame.
    plt.ioff()
    plt.show()


# --- Helper functions to record positions ---

def record_agent_positions(model, env):
    """
    Run one episode using the model and record the agent positions at each step.
    Assumes the base environment exposes the agent's position in base_env.unwrapped.agent_pos.
    
    Returns:
        positions: A list of positions (each a NumPy array, e.g. [row, col]).
    """
    positions = []
    obs = env.reset()
    done = False
    truncated = False

    # Run until episode ends.
    while not (done or truncated):
        # Try to get the agent's position.
        base_env = env.envs[0] if hasattr(env, "envs") else env
        try:
            pos = base_env.unwrapped.agent_pos
        except AttributeError:
            pos = None
        if pos is not None:
            # Store a copy of the position as a NumPy array.
            positions.append(np.array(pos))
        
        # Let the model predict an action.
        action, _ = model.predict(obs, deterministic=True)
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        elif len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            raise ValueError("Unexpected number of values returned from env.step()")
    return positions

# --- Option 1: Animate the path over a still background ---

def animate_agent_path(env, positions, interval=50):
    """
    Create an animation that shows the path taken by the agent on top of a background image
    from the environment. This animation incrementally draws a line over the recorded positions.
    
    Parameters:
        env: The environment. The function uses env.envs[0].unwrapped.render() to obtain a background.
        positions: A list (or array) of recorded agent positions (each should be [row, col]).
        interval: Delay between frames in milliseconds.
    """
    # Attempt to get a background image from the environment.
    base_env = env.envs[0] if hasattr(env, "envs") else env
    try:
        bg_frame = base_env.unwrapped.render()
        bg_frame = np.array(bg_frame, dtype=np.uint8)
    except Exception as e:
        print("Could not get a background frame, using blank grid. Error:", e)
        # Use a blank image if no background is available.
        bg_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    
    fig, ax = plt.subplots()
    ax.imshow(bg_frame)
    # Create an empty line that will be updated.
    line, = ax.plot([], [], marker="o", color="red", linestyle="--")
    
    def init():
        line.set_data([], [])
        return line,
    
    def update(frame_num):
        # Update the line to show positions up to the current frame.
        cur_path = np.array(positions[:frame_num + 1])
        # Agent positions are assumed to be (row, col); we add 0.5 to center them in the grid cell.
        line.set_data(cur_path[:, 1] + 0.5, cur_path[:, 0] + 0.5)
        return line,
    
    # Create the animation.
    anim = animation.FuncAnimation(fig, update, frames=len(positions),
                                   init_func=init, interval=interval, blit=True, repeat=False)
    plt.title("Agent Path Animation")
    plt.xlabel("Grid Column")
    plt.ylabel("Grid Row")
    plt.show()
    
    # Optionally, you can save the animation as a GIF or MP4:
    # anim.save('agent_path.gif', writer='imagemagick', fps=1000/interval)

# --- Option 2: Plot the full path on a still background ---

def plot_full_path_on_still(env, positions):
    """
    Plots a final still image of the environment with the entire agent path overlaid.
    
    Parameters:
        env: The environment (should have a render mode that produces an image).
        positions: A list or array of recorded agent positions.
    """
    # Get a background frame from the environment.
    base_env = env.envs[0] if hasattr(env, "envs") else env
    try:
        bg_frame = base_env.unwrapped.render()
        bg_frame = np.array(bg_frame, dtype=np.uint8)
    except Exception as e:
        print("Could not retrieve background frame:", e)
        bg_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(bg_frame)
    pos_arr = np.array(positions)
    plt.plot(pos_arr[:, 1] + 0.5, pos_arr[:, 0] + 0.5, marker='o', color='red', linestyle='--')
    plt.title("Agent Path Over Full Episode")
    plt.xlabel("Grid Column")
    plt.ylabel("Grid Row")
    plt.show()

# --- Example Usage in Main ---

if __name__ == "__main__":
    # Load your agent and environment (your load_agent() should create the env with render_mode="rgb_array")
    model, env = load_agent()
    
    # Record the agent positions over one full episode.
    positions = record_agent_positions(model, env)
    print(f"Episode steps taken: {len(positions)}")
    
    # Choose one of the two visualization options:
    # Option 1: Animate the path.
    animate_agent_path(env, positions, interval=50)
    
    # Option 2: Overlay the path on a final still frame.
    plot_full_path_on_still(env, positions)
    
    env.close()



def make_env():
    # Ensure that when you create the environment, you set the render_mode to "rgb_array"
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env.reset(seed=RANDOM_SEED)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

def record_episode(model, env):
    """
    Run an episode with the given model and record both rendered frames and the agent positions.
    
    Returns:
        frames: A list of rendered frames (as NumPy arrays).
        positions: A list of agent positions (each a NumPy array, e.g., [row, col]).
    """
    frames = []
    positions = []
    
    # For vectorized environments, get the base environment.
    base_env = env.envs[0] if hasattr(env, "envs") else env
    
    obs = env.reset()
    done = False
    truncated = False

    # Run the episode until termination.
    while not (done or truncated):
        # Render from the unwrapped base environment (should give the full visual scene).
        frame = base_env.unwrapped.render()
        # Safety-check: if frame is None, break out.
        if frame is None:
            print("Render returned None; please check render_mode settings.")
            break
        # Convert the rendered frame to a numeric numpy array.
        frame = np.array(frame, dtype=np.uint8)
        frames.append(frame)
        
        # Record the current position, if available.
        try:
            pos = base_env.unwrapped.agent_pos
        except AttributeError:
            pos = None
        if pos is not None:
            positions.append(np.array(pos))
        else:
            # In case no position data is available, use a placeholder (e.g., [-1, -1]).
            positions.append(np.array([-1, -1]))
        
        # Have the model predict an action.
        action, _ = model.predict(obs, deterministic=True)
        result = env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        elif len(result) == 4:
            obs, reward, done, info = result
            truncated = False
        else:
            raise ValueError("Unexpected return from env.step()")
    
    return frames, positions

def animate_episode(frames, positions, interval=50):
    """
    Create and display an animation of the episode using the recorded frames and positions.
    
    Parameters:
        frames: List of NumPy arrays representing environment frames.
        positions: List of agent positions (each as [row, col]).
        interval: Time (in ms) between animation frames.
    """
    fig, ax = plt.subplots()
    
    # Display the first frame.
    im = ax.imshow(frames[0])
    # Create a line object for the path; we'll update its data.
    line, = ax.plot([], [], marker='o', color='red', linestyle='--')
    
    def init():
        im.set_data(frames[0])
        line.set_data([], [])
        return im, line
    
    def update(frame_idx):
        # Update the background image for the current step.
        im.set_data(frames[frame_idx])
        # Update the path line to show all positions up to this frame.
        pos_arr = np.array(positions[:frame_idx+1])
        if pos_arr.size > 0 and pos_arr.ndim == 2 and pos_arr.shape[1] >= 2:
            # Adjust the positions if needed (e.g., add 0.5 to center in the grid cell).
            line.set_data(pos_arr[:, 1] + 0.5, pos_arr[:, 0] + 0.5)
        return im, line
    
    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   init_func=init, interval=interval,
                                   blit=True, repeat=False)
    ax.set_title("Agent Path Animation")
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")
    plt.show()
    
    # Optionally, save the animation:
    # anim.save('agent_episode.gif', writer='imagemagick', fps=1000/interval)

# --- Example Usage ---

if __name__ == "__main__":
    # Load your trained agent and environment.
    model, env = load_agent()  # Ensure load_agent() sets up the environment with render_mode="rgb_array"
    
    # Record the full episode.
    frames, positions = record_episode(model, env)
    print(f"Recorded {len(frames)} frames and {len(positions)} positions.")
    
    # Animate the recorded episode.
    animate_episode(frames, positions, interval=50)
    
    # Optionally, you could also plot a still image with the full overlaid path:
    # (If desired, see previous solution for plotting a still overlay)
    
    env.close()
