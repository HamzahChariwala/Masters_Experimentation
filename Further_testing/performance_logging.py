import os
import sys
import json
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from EnvironmentEdits.ActionSpace import CustomActionWrapper
from EnvironmentEdits.FeatureExtractor import SelectiveObservationWrapper, CustomCombinedExtractor
from EnvironmentEdits.CustomWrappers import GoalAngleDistanceWrapper, PartialObsWrapper, ExtractAbstractGrid, PartialRGBObsWrapper, PartialGrayObsWrapper
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import FullyObsWrapper

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Ensure the EnvironmentEdits module is accessible during model loading

def convert_ndarray_to_list(obj):
    """Recursively convert numpy.ndarray objects to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    else:
        return obj

def run_agent_and_log(agent_path, env_id, output_json_path):
    # Load the trained agent with custom objects
    custom_objects = {"features_extractor_class": CustomCombinedExtractor}
    model = DQN.load(agent_path, custom_objects=custom_objects)

    # Create the environment
    env = gym.make(env_id)
    env = CustomActionWrapper(env)
    env = RecordEpisodeStatistics(env)
    env = FullyObsWrapper(env)
    env = GoalAngleDistanceWrapper(env)
    env = PartialObsWrapper(env, n=5)  # Example parameter, adjust as needed
    env = ExtractAbstractGrid(env)
    env = PartialRGBObsWrapper(env, n=5)  # Example parameter, adjust as needed
    env = PartialGrayObsWrapper(env, n=5)  # Example parameter, adjust as needed
    env = SelectiveObservationWrapper(
        env,
        cnn_keys=['grey_partial'],
        mlp_keys=["goal_distance", "goal_direction_vector"]
    )
    env = Monitor(env)
    obs, info = env.reset()

    # Data structure to store logs
    log_data = {}

    # Log the initial observation under 'action 0'
    log_data["action 0"] = {
        "action": None,
        "reward": 0.0,
        "info": convert_ndarray_to_list(info),
        "observation": convert_ndarray_to_list(obs)
    }

    done = False
    action_count = 1  # Counter for actions
    while not done:
        # Predict the action using the trained model
        action, _ = model.predict(obs, deterministic=True)

        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Log the info at each step under a key like 'action 1', 'action 2', etc.
        log_data[f"action {action_count}"] = {
            "action": int(action),
            "reward": float(reward),
            "info": convert_ndarray_to_list(info)  # Ensure info is JSON-serializable
        }

        # Increment the action counter
        action_count += 1

        # Check if the episode is done
        done = terminated or truncated

    # Save the log data to a JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(log_data, json_file, indent=4)

if __name__ == "__main__":
    # Example usage
    agent_path = "dqn_minigrid_agent_test.zip"
    env_id = "MiniGrid-Empty-5x5-v0"
    output_json_path = "AgentTesting/agent_run_log.json"

    run_agent_and_log(agent_path, env_id, output_json_path)
    print(f"Agent run log saved to {output_json_path}")