import os
import sys
import numpy as np

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")

import gymnasium as gym
from stable_baselines3 import DQN

# Now we can import from local modules
from Agent_Evaluation.EnvironmentTooling.import_vars import create_evaluation_env, extract_env_config, load_config
from Agent_Evaluation.EnvironmentTooling.position_override import make_custom_env, ForceStartState
from Agent_Evaluation.EnvironmentTooling.extract_grid import print_env_tensor

# And from Environment_Tooling
from Environment_Tooling.BespokeEdits.FeatureExtractor import CustomCombinedExtractor


def load_agent_from_path(folder_path):

    agent_path = folder_path + "/agent.zip"
    ModelClass = DQN

    model = ModelClass.load(
        agent_path,
        custom_objects={
            "features_extractor_class": CustomCombinedExtractor
        }
    )
    return model


def check_cell_type(env_tensor, pos):
    x, y, _ = pos
    return env_tensor[y, x]

def count_lava_steps(path, env_tensor):
    lava_steps = 0
    steps = path[1:]
    for pos in steps:
        if check_cell_type(env_tensor, pos) == 'lava':
            lava_steps += 1
    return lava_steps

def check_risky_diagonal(states, action_list, env_tensor, only_next=True):
    
    if only_next:
        path = states[:2]
        actions = action_list[:1]
    else:
        path = states
        actions = action_list

    # We need to check if diagonal moves are going through lava cells
    for i in range(len(actions)):
        # Only check diagonal moves
        if actions[i] == 3 or actions[i] == 4:  # 3 = diagonal left, 4 = diagonal right
            # We need the position BEFORE the move was made
            if i >= len(path) - 1:  # Safety check
                continue
                
            # Get the position before the move
            x, y, theta = path[i]
            
            # Set up rotation matrix based on agent orientation
            position = np.array([x, y])
            cw90 = np.array([[0, 1], [-1, 0]])
            rotation = np.linalg.matrix_power(cw90, theta)
            
            # Calculate the forward position (1 step forward in agent's orientation)
            forward_vec = (position + rotation @ np.array([1, 0])).astype(int).tolist()
            forward_pos = forward_vec.copy()
            forward_pos.append(theta)  # Add orientation for the check_cell_type function
            
            # Calculate the side position based on the diagonal direction
            side_vec = None
            if actions[i] == 4:  # diagonal left
                side_vec = (position + rotation @ np.array([0, -1])).astype(int).tolist()
            elif actions[i] == 3:  # diagonal right
                side_vec = (position + rotation @ np.array([0, 1])).astype(int).tolist()
                
            side_pos = side_vec.copy()
            side_pos.append(theta)  # Add orientation for the check_cell_type function
            
            # Check if either the forward or side cell is lava
            forward_cell = check_cell_type(env_tensor, forward_pos)
            side_cell = check_cell_type(env_tensor, side_pos)
            
            if forward_cell == 'lava' or side_cell == 'lava':
                return True
                
    return False
    

def evauate_agent_on_single_env(env, model, seed, env_tensor):

    results_dict = {}
    base = env.unwrapped
    
    # if not hasattr(env, 'force'):
    #     env = ForceStartState(env)
    #     print("Wrapped environment with ForceStartState to control agent position")
    
    for x in range(base.width):
        for y in range(base.height):
            for theta in range(4):
                if x*(base.width-(x+1))==0 or y*(base.height-(y+1))==0:
                    continue
                elif x==base.width-2 and y==base.height-2:
                    continue
                else:
                    pos = (x,y)
                    ori = theta
                    env.force(pos, ori)
                    obs, info = env.reset(seed=seed)
                    mlp_input = obs['MLP_input']
                    mlp_keys = info['log_data']

                    print(f"({x}, {y}, {theta}): {check_cell_type(env_tensor, (x,y,theta))}")

                    path = []
                    path.append((int(base.agent_pos[0]), int(base.agent_pos[1]), int(base.agent_dir)))

                    done = False
                    total_reward = 0
                    step_count = 0

                    action_list = []

                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        action_list.append(action)

                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated

                        path.append((int(base.agent_pos[0]), int(base.agent_pos[1]), int(base.agent_dir)))

                        total_reward += reward
                        step_count += 1
                        if step_count > env.spec.max_episode_steps:
                            break
                    
                    results_dict[f"{x},{y},{theta}"] = {
                        "path_taken": path,
                        "next_step": {
                            "action": int(action_list[0]) if len(action_list) > 0 else None,
                            "target_state": path[1] if len(path) > 1 else None,
                            "type": check_cell_type(env_tensor, path[1]) if len(path) > 1 else None,
                            "risky_diagonal": check_risky_diagonal(path, action_list, env_tensor, only_next=True) if len(action_list) > 0 else False
                        },
                        "summary_stats": {
                            "path_length": step_count,
                            "lava_steps": count_lava_steps(path, env_tensor),
                            "total_reward": float(total_reward),
                            "reachable": step_count < env.spec.max_episode_steps
                        },
                        'model_inputs': {'raw_input': mlp_input, 
                                         'lava_mask': mlp_keys['lava_mask']},
                    }

    return results_dict
