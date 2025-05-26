import numpy as np
import gym
from stable_baselines3 import PPO
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load PPO agent
model = PPO.load("ppo_cartpole.zip")
env = gym.make("CartPole-v1")

# Gather (state, action) data from PPO
X, y = [], []
obs, _ = env.reset()
for _ in range(1000):
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    dist = model.policy.get_distribution(obs_tensor)
    action = np.argmax(dist.distribution.probs.detach().numpy())

    X.append(obs)
    y.append(action)

    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

X = np.array(X)
y = np.array(y)

# Train decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X, y)
y_pred = tree.predict(X)

# Evaluate
acc = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

print(f"\nDecision Tree Surrogate Accuracy: {acc:.3f}\n")

# Confusion matrix plot
sns.heatmap(cm, annot=True, fmt='d', cmap='magma', xticklabels=['Left', 'Right'], yticklabels=['Left', 'Right'])
plt.xlabel("Predicted Action")
plt.ylabel("PPO Action (Ground Truth)")
plt.title("Confusion Matrix: PPO vs. Decision Tree")
plt.tight_layout()
plt.show()
