from import_vars import create_evaluation_env, extract_env_config, load_config
from new_env_construction import make_custom_env

config = load_config("Agent_Storage/LavaTests/NoDeath/0.200_penalty")
env_settings = extract_env_config(config)

print(env_settings['mlp_keys'])                

# 2) Build and wrap your eval‐env
env = create_evaluation_env(env_settings, seed=42, new_approach_bool=True)
# (create_evaluation_env already applies ForceStartState)

obs, info = env.reset(options={...})
print("OBS at reset:", obs["MLP_input"])
print("INFO at reset:", info)


# 3) A few test states: (x,y), orientation
tests = [
    ((1,1), 0),
    ((1,1), 2),
    ((3,5), 1),
    ((5,3), 3),
]

for (pos, ori) in tests:
    print(f"\n--- forcing start to pos={pos}, dir={ori} ---")
    env.force(pos, ori)
    obs, info = env.reset(seed=123)   # obs now comes from your very forced state
    # Drill down to the raw MiniGrid env and read out what it thinks:
    base = env.unwrapped
    actual_pos = tuple(base.agent_pos.tolist())
    actual_dir = int(base.agent_dir)
    print(f" After reset → agent_pos={actual_pos}, agent_dir={actual_dir}")
    # Optional: also sanity‐check your first observation
    print(" Observation:")
    if isinstance(obs, dict):
        for k, v in obs.items():
            print(f"   {k!r}: {v}")
    else:
        print("   (non‐dict obs):", obs)

    # Print out the info dict
    print(" Info:")
    for k, v in info.items():
        print(f"   {k!r}: {v}")

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