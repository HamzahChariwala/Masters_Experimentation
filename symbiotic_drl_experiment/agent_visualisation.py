import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
import time

# Create the environment with render_mode set to "rgb_array"
env = gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="rgb_array")

# Load the PPO agent from the "new_models/test_model" folder.
model = PPO.load("new_models/test_model", env=env)

# Reset the environment to start
obs, info = env.reset()

# List to store frames for the animation
frames = []

# Run the agent for a defined number of steps (here, 50)
for step in range(50):
    # Predict the action using the loaded agent (deterministic mode)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # Get the current frame (RGB array)
    frame = env.render()
    frames.append(frame)

    # If the episode is done, reset the environment
    if terminated or truncated:
        obs, info = env.reset()

# Clean up by closing the environment
env.close()

# Create an animation using matplotlib
fig = plt.figure()
im = plt.imshow(frames[0])

def update(frame):
    im.set_array(frame)
    return [im]

# Create and display the animation; the interval is in milliseconds.
ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, blit=True)
plt.axis('off')
plt.show()
