import gymnasium as gym
import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
import numpy as np

def visualize_lava_crossing(n_envs=4, base_seed=42):
    # Set up the matplotlib figure
    cols = int(np.ceil(np.sqrt(n_envs)))
    rows = int(np.ceil(n_envs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)  # Flatten in case of single row/column

    for i in range(n_envs):
        # Initialize the environment with the specified seed
        env = gym.make("MiniGrid-LavaCrossingS9N2-v0", render_mode="rgb_array")
        env = RGBImgObsWrapper(env)  # Get RGB image observations
        env = ImgObsWrapper(env)     # Remove the 'mission' field
        obs, _ = env.reset(seed=base_seed + i)

        # Render the environment
        img = env.render()

        # Display the image
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"Env {i} (Seed {base_seed + i})")
        ax.axis('off')

        env.close()

    # Hide any unused subplots
    for j in range(n_envs, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
visualize_lava_crossing(n_envs=8, base_seed=0)
