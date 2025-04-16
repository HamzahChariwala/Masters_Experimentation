# train_student_and_visualize.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial

# Load dataset
data = np.load("teacher_dataset_cartpole.npz")
observations = data["observations"]
action_probs = data["action_probs"]
actions = data["actions"]

# Convert to PyTorch tensors
X = torch.tensor(observations, dtype=torch.float32)
y = torch.tensor(actions, dtype=torch.long)

# Define a small student network
class StudentPolicy(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class BigStudentPolicy(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model, loss, optimizer
student = BigStudentPolicy()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student.parameters(), lr=1e-3)

# Training loop
EPOCHS = 30
BATCH_SIZE = 15

for epoch in range(EPOCHS):
    permutation = torch.randperm(X.size(0))
    epoch_loss = 0.0
    for i in range(0, X.size(0), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_x, batch_y = X[indices], y[indices]

        outputs = student(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# Evaluate accuracy
with torch.no_grad():
    logits = student(X)
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == y).numpy().astype(np.float32)

cart_velocity = observations[:, 1]
pole_angle = observations[:, 2]
pole_angular_velocity = observations[:, 3]

points = np.stack([cart_velocity, pole_angle, pole_angular_velocity], axis=1)

# Use KD-Tree for efficient neighborhood lookup
tree = scipy.spatial.cKDTree(points)

# Define radius for region-based averaging
RADIUS = 1

# Compute local average accuracy for each point
avg_accuracy = []
for i in range(len(points)):
    indices = tree.query_ball_point(points[i], r=RADIUS)
    if indices:
        mean_acc = np.mean(correct[indices])
    else:
        mean_acc = 0.0
    avg_accuracy.append(mean_acc)

avg_accuracy = np.array(avg_accuracy)

# Plot using averaged accuracy
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(cart_velocity, pole_angle, pole_angular_velocity,
               c=avg_accuracy, cmap='magma', alpha=0.85)

ax.set_xlabel('Cart Velocity')
ax.set_ylabel('Pole Angle')
ax.set_zlabel('Pole Angular Velocity')
ax.set_title(f'Local Accuracy in State Space (r = {RADIUS})')
fig.colorbar(p, label='Average Accuracy (Local Region)')

plt.show()
