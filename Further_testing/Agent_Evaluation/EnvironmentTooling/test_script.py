import os

from stable_baselines3 import DQN

from import_vars import create_evaluation_env, extract_env_config, load_config
from new_env_construction import make_custom_env

from Environment_Tooling.BespokeEdits.FeatureExtractor import CustomCombinedExtractor


agent_path = "Agent_Storage/LavaTests/NoDeath/0.200_penalty/agent.zip"

ModelClass = DQN

model = ModelClass.load(
    agent_path,
    custom_objects={
        "features_extractor_class": CustomCombinedExtractor
    }
)

config = load_config("Agent_Storage/LavaTests/NoDeath/0.200_penalty")
env_settings = extract_env_config(config)             
env = create_evaluation_env(env_settings, seed=42, new_approach_bool=True)

# obs, info = env.reset(options={...})
# print("OBS at reset:", obs["MLP_input"])
# print("INFO at reset:", info)


# 3) A few test states: (x,y), orientation
tests = [
    ((1,1), 0),
    ((1,1), 2),
    ((3,5), 1),
    ((5,3), 3),
]

# for (pos, ori) in tests:
#     print(f"\n--- forcing start to pos={pos}, dir={ori} ---")
#     env.force(pos, ori)
#     obs, info = env.reset(seed=123)   # obs now comes from your very forced state
#     # Drill down to the raw MiniGrid env and read out what it thinks:
#     base = env.unwrapped
#     actual_pos = tuple(base.agent_pos.tolist())
#     actual_dir = int(base.agent_dir)
#     print(f" After reset → agent_pos={actual_pos}, agent_dir={actual_dir}")
#     # Optional: also sanity‐check your first observation
#     print(" Observation:")
#     if isinstance(obs, dict):
#         for k, v in obs.items():
#             print(f"   {k!r}: {v}")
#     else:
#         print("   (non‐dict obs):", obs)

#     # Print out the info dict
#     print(" Info:")
#     for k, v in info.items():
#         print(f"   {k!r}: {v}")

for (pos, ori) in tests:
    print(f"\n--- forcing start to pos={pos}, dir={ori} ---")
    env.force(pos, ori)
    obs, info = env.reset(seed=123)

    # grab the base MiniGrid env so we can inspect agent_pos / agent_dir
    base = env.unwrapped

    # record the trajectory of (x,y,dir)
    path = []
    # include the *initial* forced state
    path.append((int(base.agent_pos[0]), int(base.agent_pos[1]), int(base.agent_dir)))

    done = False
    total_reward = 0
    step_count = 0

    # roll out until termination
    while not done:
        # 1) get an action from your RL agent
        action, _ = model.predict(obs, deterministic=True)

        # 2) step the env
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 3) log the new state
        path.append((int(base.agent_pos[0]), int(base.agent_pos[1]), int(base.agent_dir)))

        total_reward += reward
        step_count += 1
        if step_count > env.spec.max_episode_steps:
            # safety
            break

    print(f"Path taken ({len(path)} steps): {path}")
    print(f"Total reward: {total_reward}, steps: {step_count}")



def dump_wrappers(env):
    w = env
    stack = []
    while True:
        stack.append(type(w).__name__)
        # Drill down one level if possible
        if hasattr(w, 'env'):
            w = w.env
        elif hasattr(w, 'unwrapped') and w is not w.unwrapped:
            w = w.unwrapped
        else:
            break
    return stack

env = create_evaluation_env(env_settings, seed=42)
print("WRAPPER STACK:", dump_wrappers(env))


env.close()