import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Create the environment
env = gym.make("MiniGrid-LavaGapS7-v0", render_mode="rgb_array")
obs, info = env.reset()

# Store frames for animation
frames = []

for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()
    frames.append(frame)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

# Create animation using matplotlib
fig = plt.figure()
im = plt.imshow(frames[0])

def update(frame):
    im.set_array(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, blit=True)
plt.axis('off')
plt.show()
