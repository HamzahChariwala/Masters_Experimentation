import os
import sys
import json
import gymnasium as gym
import numpy as np
import time
import signal
import platform
import argparse
from stable_baselines3 import DQN, PPO, A2C

# Import environment generation function
from EnvironmentEdits.EnvironmentGeneration import make_env
from EnvironmentEdits.BespokeEdits.FeatureExtractor import CustomCombinedExtractor

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Add a timeout handler to prevent infinite loops
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")

# Check if the platform supports SIGALRM
HAS_ALARM = hasattr(signal, 'SIGALRM') and platform.system() != 'Windows'

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

def print_env_structure(env, depth=0):
    """Print the structure of nested environment wrappers."""
    if depth == 0:
        print("\nEnvironment wrapper structure:")
    
    indent = "  " * depth
    env_name = env.__class__.__name__
    print(f"{indent}- {env_name}")
    
    if hasattr(env, "env"):
        print_env_structure(env.env, depth + 1)

def run_agent_and_log(
    agent_path, 
    env_id, 
    output_json_path, 
    agent_type="dqn",
    max_episode_steps=150,
    use_random_spawn=False,
    use_no_death=True,
    no_death_types=("lava",),
    death_cost=0,
    monitor_diagonal_moves=True,
    diagonal_success_reward=0.01,
    diagonal_failure_penalty=0,
    debug=False,
    timeout=30  # Timeout in seconds
):
    """
    Run a trained agent in the environment and log its performance.
    
    Parameters:
    ----------
    agent_path : str
        Path to the trained agent model file
    env_id : str
        Environment ID (e.g., MiniGrid-Empty-8x8-v0)
    output_json_path : str
        Path to save the output JSON log
    agent_type : str, optional
        Type of agent ("dqn", "ppo", or "a2c"), default="dqn"
    max_episode_steps : int, optional
        Maximum steps per episode, default=150
    use_random_spawn : bool, optional
        If True, agent will spawn at random empty locations instead of the default top-left position.
        If False, agent will spawn at the environment's default position (usually top-left).
        Default=False.
    use_no_death : bool, optional
        If True, apply the NoDeath wrapper, default=True
    no_death_types : tuple, optional
        Types of elements that don't cause death, default=("lava",)
    death_cost : float, optional
        Penalty for death, default=0
    monitor_diagonal_moves : bool, optional
        If True, monitor diagonal move usage, default=True
    diagonal_success_reward : float, optional
        Reward for successful diagonal moves, default=0.01
    diagonal_failure_penalty : float, optional
        Penalty for failed diagonal moves, default=0
    debug : bool, optional
        If True, print debug information, default=False
    timeout : int, optional
        Timeout in seconds, default=30
        
    Returns:
    -------
    bool
        True if execution completed successfully, False otherwise
    """
    print(f"DEBUG: Starting run_agent_and_log function")
    
    # For non-Unix platforms or if timeout is disabled
    start_time = time.time()
    
    # Setup timeout handler (only on platforms that support it)
    if timeout > 0 and HAS_ALARM:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        print(f"DEBUG: Set timeout to {timeout} seconds")
    else:
        print(f"DEBUG: Using manual timeout checking (timeout={timeout}s)")
    
    try:
        # Load the trained agent with custom objects
        print(f"DEBUG: Loading model from {agent_path}")
        custom_objects = {"features_extractor_class": CustomCombinedExtractor}
        
        # Load the appropriate model type
        if agent_type.lower() == "dqn":
            model = DQN.load(agent_path, custom_objects=custom_objects)
        elif agent_type.lower() == "ppo":
            model = PPO.load(agent_path, custom_objects=custom_objects)
        elif agent_type.lower() == "a2c":
            model = A2C.load(agent_path, custom_objects=custom_objects)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
            
        print(f"DEBUG: Model loaded successfully (type: {agent_type})")

        # Create the environment using the imported function
        print(f"DEBUG: Creating environment {env_id}")
        env_fn = make_env(
            env_id=env_id,
            rank=0,
            env_seed=12345,
            window_size=7,
            cnn_keys=[],
            mlp_keys=["four_way_goal_direction",
                      "four_way_angle_alignment",
                      "barrier_mask",
                      "lava_mask"],
            max_episode_steps=max_episode_steps,  # Maximum steps per episode
            use_random_spawn=use_random_spawn,    # Random spawn positions
            use_no_death=use_no_death,            # Use NoDeath wrapper
            no_death_types=no_death_types,        # Types that don't cause death
            death_cost=death_cost,                # Penalty for death
            monitor_diagonal_moves=monitor_diagonal_moves,  # Monitor diagonal moves
            diagonal_success_reward=diagonal_success_reward,  # Reward for successful diagonal moves
            diagonal_failure_penalty=diagonal_failure_penalty  # Penalty for failed diagonal moves
        )
        # Actually create the environment by calling the function
        env = env_fn()
        print(f"DEBUG: Environment created successfully")
        
        # Print the environment structure
        print_env_structure(env)
        
        # Reset the environment
        print(f"DEBUG: Resetting environment")
        obs, info = env.reset()
        print(f"DEBUG: Environment reset complete")
        print(f"DEBUG: Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
        print(f"DEBUG: Info keys: {list(info.keys())}")

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
        max_steps_safety = max_episode_steps * 2  # Safety counter to prevent infinite loops
        step_count = 0
        
        print(f"DEBUG: Starting step loop")
        while not done and step_count < max_steps_safety:
            # Check for manual timeout on platforms without SIGALRM
            if timeout > 0 and not HAS_ALARM and (time.time() - start_time > timeout):
                raise TimeoutException(f"Execution timed out after {timeout} seconds")
            
            step_start_time = time.time()
            print(f"DEBUG: Step {step_count+1} - Getting action from model")
            
            # Add specific exception handling for the prediction part
            try:
                # Predict the action using the trained model
                action, _ = model.predict(obs, deterministic=True)
                print(f"DEBUG: Model returned action {action}")
            except Exception as predict_error:
                print(f"ERROR in predict: {predict_error}")
                import traceback
                traceback.print_exc()
                raise
            
            # Convert action to int to avoid unhashable numpy.ndarray error in DiagonalMoveMonitor
            try:
                if isinstance(action, np.ndarray):
                    action_int = int(action.item())
                else:
                    action_int = int(action)
                print(f"DEBUG: Converted action to {action_int}")
            except Exception as convert_error:
                print(f"ERROR in conversion: {convert_error}")
                import traceback
                traceback.print_exc()
                raise

            # Take a step in the environment
            try:
                print(f"DEBUG: Taking step with action {action_int}")
                obs, reward, terminated, truncated, info = env.step(action_int)
                step_end_time = time.time()
                print(f"DEBUG: Step completed in {step_end_time - step_start_time:.3f}s. Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            except Exception as step_error:
                print(f"ERROR in step: {step_error}")
                import traceback
                traceback.print_exc()
                raise

            # Log the info at each step under a key like 'action 1', 'action 2', etc.
            log_data[f"action {action_count}"] = {
                "action": int(action_int),
                "reward": float(reward),
                "info": convert_ndarray_to_list(info)  # Ensure info is JSON-serializable
            }

            # Increment the action counter
            action_count += 1
            step_count += 1

            # Check if the episode is done
            done = terminated or truncated
            
            # Print any relevant info about the step
            if debug and 'eval_episode_return' in info:
                print(f"DEBUG: Current return: {info['eval_episode_return']}")

        print(f"DEBUG: Step loop completed after {step_count} steps. Done: {done}")
        
        if step_count >= max_steps_safety:
            print(f"WARNING: Reached safety limit of {max_steps_safety} steps without completing episode")
        
        # Save the log data to a JSON file
        print(f"DEBUG: Saving log data to {output_json_path}")
        with open(output_json_path, "w") as json_file:
            json.dump(log_data, json_file, indent=4)
        print(f"DEBUG: Log data saved successfully")
        
        # Disable the alarm if it was set
        if timeout > 0 and HAS_ALARM:
            signal.alarm(0)
            
        return True
        
    except TimeoutException:
        print(f"ERROR: Execution timed out after {timeout} seconds")
        return False
    finally:
        # Always disable the alarm even if an exception occurred
        if timeout > 0 and HAS_ALARM:
            signal.alarm(0)

# Commented out parse_args function for reference
"""
def parse_args():
    \"\"\"Parse command line arguments.\"\"\"
    parser = argparse.ArgumentParser(description='Run agent and log results')
    
    # Required arguments
    parser.add_argument('--agent', required=True, help='Path to the agent model file')
    parser.add_argument('--env', required=True, help='Environment ID (e.g., MiniGrid-Empty-8x8-v0)')
    parser.add_argument('--output', required=True, help='Path to save the output JSON log')
    
    # Optional arguments
    parser.add_argument('--agent-type', default='dqn', choices=['dqn', 'ppo', 'a2c'], help='Type of agent (default: dqn)')
    parser.add_argument('--max-steps', type=int, default=150, help='Maximum episode steps (default: 150)')
    parser.add_argument('--random-spawn', action='store_true', help='Use random spawn positions')
    parser.add_argument('--no-death', action='store_true', default=True, help='Use NoDeath wrapper')
    parser.add_argument('--no-death-types', default='lava', help='Types that don\'t cause death (comma-separated)')
    parser.add_argument('--death-cost', type=float, default=0, help='Penalty for death (default: 0)')
    parser.add_argument('--monitor-diag', action='store_true', default=True, help='Monitor diagonal moves')
    parser.add_argument('--diag-reward', type=float, default=0.01, help='Reward for successful diagonal moves (default: 0.01)')
    parser.add_argument('--diag-penalty', type=float, default=0, help='Penalty for failed diagonal moves (default: 0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds (default: 60, 0 to disable)')
    
    return parser.parse_args()
"""

if __name__ == "__main__":
    # Manual configuration (no command line arguments)
    agent_path = "dqn_minigrid_agent_empty_test_biglava_10m.zip"
    env_id = "MiniGrid-LavaCrossingS11N5-v0"
    output_json_path = "AgentTesting/agent_run_log_empty_biglava_10m.json"
    agent_type = "dqn"
    max_episode_steps = 150
    use_random_spawn = True
    use_no_death = True
    no_death_types = ("lava",)
    death_cost = 0
    monitor_diagonal_moves = True
    diagonal_success_reward = 0.01
    diagonal_failure_penalty = 0
    debug_mode = True
    timeout_seconds = 60

    # Print configuration
    print("\n====== AGENT EVALUATION SETTINGS ======")
    print(f"Agent path: {agent_path}")
    print(f"Agent type: {agent_type}")
    print(f"Environment: {env_id}")
    print(f"Output path: {output_json_path}")
    print(f"Max episode steps: {max_episode_steps}")
    print(f"Random agent spawning: {use_random_spawn}")
    print(f"NoDeath wrapper: {use_no_death}")
    if use_no_death:
        print(f"  - Death types: {no_death_types}")
        print(f"  - Death cost: {death_cost}")
    print(f"Monitor diagonal moves: {monitor_diagonal_moves}")
    print(f"Diagonal success reward: {diagonal_success_reward}")
    print(f"Diagonal failure penalty: {diagonal_failure_penalty}")
    print(f"Debug mode: {debug_mode}")
    print(f"Timeout: {timeout_seconds} seconds")
    print("=====================================\n")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)

    try:
        success = run_agent_and_log(
            agent_path, 
            env_id, 
            output_json_path,
            agent_type=agent_type,
            max_episode_steps=max_episode_steps,
            use_random_spawn=use_random_spawn,
            use_no_death=use_no_death,
            no_death_types=no_death_types,
            death_cost=death_cost,
            monitor_diagonal_moves=monitor_diagonal_moves,
            diagonal_success_reward=diagonal_success_reward,
            diagonal_failure_penalty=diagonal_failure_penalty,
            debug=debug_mode,
            timeout=timeout_seconds
        )
        
        if success:
            print(f"Agent run log saved to {output_json_path}")
        else:
            print(f"Failed to complete agent run")
            
    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()