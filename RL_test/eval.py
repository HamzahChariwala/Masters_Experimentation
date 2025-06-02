import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# === Load PPO-trained agent ===
MODEL_PATH = "ppo_cartpole.zip"  # <- Update this if your model path differs

# Create environment
env = gym.make("CartPole-v1")
model = PPO.load(MODEL_PATH)

# === Interpretable policy: Rule-based probability output ===
def interpretable_policy(state):
    angle = state[2]  # Pole angle
    if angle < 0:
        return np.array([0.8, 0.2])  # Prefer action 0 (left)
    else:
        return np.array([0.2, 0.8])  # Prefer action 1 (right)

# === KL Divergence function ===
def kl_divergence(p, q):
    epsilon = 1e-8
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q))

# === Get action distribution from PPO ===
def get_ppo_action_probs(model, state):
    # PPO models support predict + predict_proba via policy's distribution
    obs_tensor = model.policy.obs_to_tensor(state)[0]
    dist = model.policy.get_distribution(obs_tensor)
    return dist.distribution.probs.detach().numpy()[0]

# === Main evaluation loop ===
def evaluate_kl(n_steps=100):
    obs, _ = env.reset()
    kl_values = []

    for _ in range(n_steps):
        p = get_ppo_action_probs(model, obs)
        q = interpretable_policy(obs)
        kl = kl_divergence(p, q)
        kl_values.append(kl)

        action = np.argmax(p)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    # Print summary stats
    print("\n=== KL Divergence Evaluation ===")
    print(f"Samples Evaluated: {len(kl_values)}")
    print(f"Mean KL Divergence: {np.mean(kl_values):.4f}")
    print(f"Std Dev KL:         {np.std(kl_values):.4f}")
    print(f"Min KL:             {np.min(kl_values):.4f}")
    print(f"Max KL:             {np.max(kl_values):.4f}")
    print("===============================\n")

if __name__ == "__main__":
    # Collect data
    obs, _ = env.reset()
    angles, kl_values, ang_vels = [], [], []

    for _ in range(100):
        p = get_ppo_action_probs(model, obs)
        q = interpretable_policy(obs)
        kl = kl_divergence(p, q)

        angles.append(obs[2])      # Pole angle
        ang_vels.append(obs[3])   # Angular velocity
        kl_values.append(kl)

        obs, _, terminated, truncated, _ = env.step(np.argmax(p))
        if terminated or truncated:
            obs, _ = env.reset()

    # Plot
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(angles, kl_values, c=ang_vels, cmap='magma', alpha=0.8)
    plt.colorbar(sc, label="Pole Angular Velocity")
    plt.xlabel("Pole Angle (radians)")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence vs. Pole Angle\nColoured by Angular Velocity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()