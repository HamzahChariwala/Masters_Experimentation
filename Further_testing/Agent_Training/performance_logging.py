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
from pathlib import Path

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained agent and log performance metrics")
    parser.add_argument("--agent-path", type=str, help="Path to the trained agent file (agent.zip)")
    parser.add_argument("--env-id", type=str, default="MiniGrid-LavaCrossingS9N1-v0", help="Environment ID")
    parser.add_argument("--output-json", type=str, help="Path to save the output JSON log")
    parser.add_argument("--agent-type", type=str, default="dqn", choices=["dqn", "ppo", "a2c"], help="Type of agent")
    parser.add_argument("--max-steps", type=int, default=150, help="Maximum steps per episode")
    parser.add_argument("--random-spawn", action="store_true", help="Use random spawn positions")
    parser.add_argument("--disable-no-death", action="store_true", help="Disable the NoDeath wrapper")
    parser.add_argument("--death-cost", type=float, default=0, help="Penalty for death")
    parser.add_argument("--diagonal-success-reward", type=float, default=0.01, help="Reward for successful diagonal moves")
    parser.add_argument("--diagonal-failure-penalty", type=float, default=0, help="Penalty for failed diagonal moves")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument("--path", type=str, required=True, help="Agent folder name in Agent_Storage")
    args = parser.parse_args()
    
    # Require path parameter
    if not args.path:
        print("Error: No agent folder specified. Please use --path parameter.")
        parser.print_help()
        sys.exit(1)
    
    # Handle agent folder in Agent_Storage
    agent_dir = os.path.join("Agent_Storage", args.path)
    
    # Check if agent folder exists
    if not os.path.exists(agent_dir):
        print(f"Error: Agent folder '{args.path}' not found in Agent_Storage")
        sys.exit(1)
        
    # Look for agent.zip in the folder
    agent_path = os.path.join(agent_dir, "agent.zip")
    if not os.path.exists(agent_path):
        print(f"Error: agent.zip not found in Agent_Storage/{args.path}")
        sys.exit(1)
        
    # Determine output JSON path if not specified
    if not args.output_json:
        output_dir = os.path.join(agent_dir, "evaluations")
        os.makedirs(output_dir, exist_ok=True)
        args.output_json = os.path.join(output_dir, "performance_log.json")
        print(f"Output will be saved to: {args.output_json}")
    
    # Run the agent and log performance
    success = run_agent_and_log(
        agent_path=agent_path,
        env_id=args.env_id,
        output_json_path=args.output_json,
        agent_type=args.agent_type,
        max_episode_steps=args.max_steps,
        use_random_spawn=args.random_spawn,
        use_no_death=not args.disable_no_death,
        no_death_types=("lava",),  # Always use lava as no_death_type
        death_cost=args.death_cost,
        monitor_diagonal_moves=True,  # Always monitor diagonal moves
        diagonal_success_reward=args.diagonal_success_reward,
        diagonal_failure_penalty=args.diagonal_failure_penalty,
        debug=args.debug,
        timeout=args.timeout
    )
    
    if success:
        print(f"Successfully completed performance logging. Results saved to {args.output_json}")
        sys.exit(0)
    else:
        print("Failed to complete performance logging")
        sys.exit(1)