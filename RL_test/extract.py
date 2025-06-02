# collect_teacher_data.py

import numpy as np
import torch
from stable_baselines3 import PPO
import gymnasium as gym

# Load trained PPO model
model = PPO.load("ppo_cartpole")

# Create environment
env = gym.make("CartPole-v1")

# Number of episodes to roll out for data collection
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 500  # CartPole ends after 500 steps

# Storage lists
observations = []
action_probs = []
actions_taken = []

# Disable gradients for teacher model
model.policy.eval()
with torch.no_grad():
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            # Get action distribution from the policy
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.squeeze().cpu().numpy()
            action = np.random.choice(len(probs), p=probs)

            # Save data
            observations.append(obs)
            action_probs.append(probs)
            actions_taken.append(action)

            # Step in environment
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                break

# Convert to NumPy arrays
observations = np.array(observations)
action_probs = np.array(action_probs)
actions_taken = np.array(actions_taken)

# Save to file
np.savez("teacher_dataset_cartpole.npz",
         observations=observations,
         action_probs=action_probs,
         actions=actions_taken)

print(f"Saved dataset with {len(observations)} samples to 'teacher_dataset_cartpole.npz'")
