#!/usr/bin/env python3
import os, json, numpy as np, matplotlib.pyplot as plt, yaml, sys
from matplotlib.cm import magma

project_root = os.path.dirname(os.getcwd())
sys.path.insert(0, project_root)

def extract_timesteps(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('experiment', {}).get('output', {}).get('total_timesteps')
    except:
        return None

agent_base_dir = os.path.join(project_root, 'Agent_Storage', 'LavaTests')
agent_types = ['Standard', 'Standard2', 'Standard3']

print('Finding agent files...')
all_metrics = []

for agent_type in agent_types:
    agent_dir = os.path.join(agent_base_dir, agent_type)
    if not os.path.exists(agent_dir):
        continue
    for item in os.listdir(agent_dir):
        version_dir = os.path.join(agent_dir, item)
        if os.path.isdir(version_dir) and item.startswith(f'{agent_type}-v'):
            perf_file = os.path.join(version_dir, 'evaluation_summary', 'performance_all_states.json')
            config_file = os.path.join(version_dir, 'config.yaml')
            if os.path.exists(perf_file) and os.path.exists(config_file):
                timesteps = extract_timesteps(config_file)
                try:
                    with open(perf_file, 'r') as f:
                        data = json.load(f)
                    summary = data.get('overall_summary', {})
                    goal_reached = summary.get('goal_reached_proportion')
                    next_cell_lava = summary.get('next_cell_lava_proportion')
                    if goal_reached is not None and next_cell_lava is not None:
                        all_metrics.append({
                            'timesteps': timesteps,
                            'goal_reached_proportion': goal_reached,
                            'y_value': 1 - next_cell_lava
                        })
                except:
                    continue

print(f'Found {len(all_metrics)} valid metrics')

# Group by timesteps
timestep_groups = {}
for m in all_metrics:
    t = m['timesteps']
    if t not in timestep_groups:
        timestep_groups[t] = {'x': [], 'y': []}
    timestep_groups[t]['x'].append(m['goal_reached_proportion'])
    timestep_groups[t]['y'].append(m['y_value'])

sorted_timesteps = sorted(timestep_groups.keys())
colors = [magma(0.2), magma(0.5), magma(0.8)]

# Create save directory
save_dir = 'standard'
os.makedirs(save_dir, exist_ok=True)

# Plot 1: Large dots
print('Creating large dots plot...')
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
for i, timesteps in enumerate(sorted_timesteps):
    data = timestep_groups[timesteps]
    label = f'{timesteps:,} timesteps'
    ax1.scatter(data['x'], data['y'], c=[colors[i]], label=label, alpha=0.7, s=150, edgecolors='black', linewidth=1.0)

ax1.set_xlabel('Goal Reached Proportion', fontsize=12, fontweight='bold')
ax1.set_ylabel('1 - Next Cell Lava Proportion', fontsize=12, fontweight='bold')
ax1.set_title('Agent Performance: Goal Reached vs Lava Avoidance (Large Dots)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc='best')

save_path1 = os.path.join(save_dir, 'standard_agents_scatter_large_dots.png')
plt.tight_layout()
plt.savefig(save_path1, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {save_path1}')

# Plot 2: With ellipses
print('Creating bubbles plot...')
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
for i, timesteps in enumerate(sorted_timesteps):
    data = timestep_groups[timesteps]
    label = f'{timesteps:,} timesteps'
    ax2.scatter(data['x'], data['y'], c=[colors[i]], label=label, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)

ax2.set_xlabel('Goal Reached Proportion', fontsize=12, fontweight='bold')
ax2.set_ylabel('1 - Next Cell Lava Proportion', fontsize=12, fontweight='bold')
ax2.set_title('Agent Performance: Goal Reached vs Lava Avoidance (With Confidence Ellipses)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11, loc='best')

save_path2 = os.path.join(save_dir, 'standard_agents_scatter_with_bubbles.png')
plt.tight_layout()
plt.savefig(save_path2, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {save_path2}')

print('Both plots created successfully!') 