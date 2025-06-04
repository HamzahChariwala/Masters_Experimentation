#!/usr/bin/env python3
"""
Generate penalty vs performance analysis plots for reward function types.
Two plots: Goal Reached and Next Cell Lava vs Penalty (1-10).
Each plot shows mean performance with ±1 standard deviation confidence bands for Linear, Exponential, and Sigmoid.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import magma


def parse_leaderboard_data(leaderboard_file):
    """Parse the leaderboard markdown file to extract reward function performance data."""
    reward_data = {
        'Linear': {},
        'Exponential': {},
        'Sigmoid': {}
    }
    
    with open(leaderboard_file, 'r') as f:
        content = f.read()
    
    # Extract Goal Reached Proportion section
    goal_section = re.search(r'## Goal Reached Proportion\n\n(.*?)\n## ', content, re.DOTALL)
    if goal_section:
        goal_lines = goal_section.group(1).strip().split('\n')
        
        for line in goal_lines:
            if '| LavaTests/EpisodeEnd/' in line and not line.startswith('|---'):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 5:  # Should have empty, rank, agent_path, mean, std, empty
                    agent_path = parts[2]
                    mean_val = float(parts[3])
                    std_val = float(parts[4])
                    
                    # Parse reward function type and penalty
                    match = re.search(r'LavaTests/EpisodeEnd/(\w+)/(\d+)_penalty', agent_path)
                    if match:
                        reward_type = match.group(1)
                        penalty = int(match.group(2))
                        
                        if reward_type in reward_data:
                            if 'goal_reached' not in reward_data[reward_type]:
                                reward_data[reward_type]['goal_reached'] = {}
                            reward_data[reward_type]['goal_reached'][penalty] = {'mean': mean_val, 'std': std_val}
    
    # Extract Next Cell Lava Proportion section
    lava_section = re.search(r'## Next Cell Lava Proportion\n\n(.*?)\n## ', content, re.DOTALL)
    if lava_section:
        lava_lines = lava_section.group(1).strip().split('\n')
        
        for line in lava_lines:
            if '| LavaTests/EpisodeEnd/' in line and not line.startswith('|---'):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 5:  # Should have empty, rank, agent_path, mean, std, empty
                    agent_path = parts[2]
                    mean_val = float(parts[3])
                    std_val = float(parts[4])
                    
                    # Parse reward function type and penalty
                    match = re.search(r'LavaTests/EpisodeEnd/(\w+)/(\d+)_penalty', agent_path)
                    if match:
                        reward_type = match.group(1)
                        penalty = int(match.group(2))
                        
                        if reward_type in reward_data:
                            if 'next_cell_lava' not in reward_data[reward_type]:
                                reward_data[reward_type]['next_cell_lava'] = {}
                            reward_data[reward_type]['next_cell_lava'][penalty] = {'mean': mean_val, 'std': std_val}
    
    return reward_data


def create_penalty_analysis_plots():
    """Create separate penalty vs performance analysis plots for each reward function type."""
    # Find the leaderboard file
    leaderboard_file = os.path.join(os.path.dirname(__file__), "..", "Leaderboard_Status", "new", "all_states_leaderboard.md")
    
    if not os.path.exists(leaderboard_file):
        print(f"Leaderboard file not found: {leaderboard_file}")
        return
    
    # Parse the leaderboard data
    print("Parsing leaderboard data...")
    reward_data = parse_leaderboard_data(leaderboard_file)
    
    # Verify we have data
    total_entries = sum(len(data.get('goal_reached', {})) for data in reward_data.values())
    if total_entries == 0:
        print("No reward function data found in leaderboard.")
        return
    
    print(f"Successfully parsed data for {len(reward_data)} reward function types")
    for reward_type, data in reward_data.items():
        goal_count = len(data.get('goal_reached', {}))
        lava_count = len(data.get('next_cell_lava', {}))
        print(f"  {reward_type}: {goal_count} goal entries, {lava_count} lava entries")
    
    # Colors matching the reward function plots
    reward_colors = {
        'Linear': magma(0.2),
        'Exponential': magma(0.5),
        'Sigmoid': magma(0.8)
    }
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), "custom_reward_2")
    os.makedirs(save_dir, exist_ok=True)
    
    # Penalty range
    penalties = list(range(1, 11))
    
    # Set fixed axis ranges matching previous plots (NoDeath agents)
    goal_ylim = (0.2, 1)  # Goal reached: 0.2-1
    lava_ylim = (0.25, 0.55)  # Lava proportion: 0.25-0.55
    
    save_paths = []
    
    # Create individual plots for each reward function
    for reward_type in ['Linear', 'Exponential', 'Sigmoid']:
        if reward_type not in reward_data:
            continue
        
        color = reward_colors[reward_type]
        
        # Goal Reached Plot
        if 'goal_reached' in reward_data[reward_type]:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            goal_data = reward_data[reward_type]['goal_reached']
            
            # Extract data for available penalties
            x_vals = []
            means = []
            stds = []
            
            for penalty in penalties:
                if penalty in goal_data:
                    x_vals.append(penalty)
                    means.append(goal_data[penalty]['mean'])
                    stds.append(goal_data[penalty]['std'])
            
            if len(x_vals) > 0:
                x_vals = np.array(x_vals)
                means = np.array(means)
                stds = np.array(stds)
                
                # Plot mean line
                ax.plot(x_vals, means, color=color, linewidth=3, 
                       marker='o', markersize=8)
                
                # Plot confidence band (mean ± 1 std)
                ax.fill_between(x_vals, means - stds, means + stds, 
                               color=color, alpha=0.3)
                
                print(f"{reward_type} Goal Reached: {len(x_vals)} data points")
                print(f"  Penalty range: {min(x_vals)} - {max(x_vals)}")
                print(f"  Mean range: {min(means):.3f} - {max(means):.3f}")
            
            ax.set_xlabel('Penalty', fontsize=14, fontweight='bold')
            ax.set_ylabel('Goal Reached Proportion', fontsize=14, fontweight='bold')
            ax.set_title(f'{reward_type} - Goal Reached Proportion vs Penalty', fontsize=16, fontweight='bold')
            ax.set_xlim(0.5, 10.5)
            ax.set_ylim(goal_ylim)
            ax.set_xticks(penalties)
            ax.grid(True, alpha=0.3)
            
            # Save goal reached plot
            goal_save_path = os.path.join(save_dir, f"{reward_type.lower()}_goal_reached_vs_penalty.png")
            plt.tight_layout()
            plt.savefig(goal_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            save_paths.append(goal_save_path)
            print(f"{reward_type} goal reached plot saved: {goal_save_path}")
        
        # Next Cell Lava Plot
        if 'next_cell_lava' in reward_data[reward_type]:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            lava_data = reward_data[reward_type]['next_cell_lava']
            
            # Extract data for available penalties
            x_vals = []
            means = []
            stds = []
            
            for penalty in penalties:
                if penalty in lava_data:
                    x_vals.append(penalty)
                    means.append(lava_data[penalty]['mean'])
                    stds.append(lava_data[penalty]['std'])
            
            if len(x_vals) > 0:
                x_vals = np.array(x_vals)
                means = np.array(means)
                stds = np.array(stds)
                
                # Plot mean line
                ax.plot(x_vals, means, color=color, linewidth=3, 
                       marker='o', markersize=8)
                
                # Plot confidence band (mean ± 1 std)
                ax.fill_between(x_vals, means - stds, means + stds, 
                               color=color, alpha=0.3)
                
                print(f"{reward_type} Next Cell Lava: {len(x_vals)} data points")
                print(f"  Penalty range: {min(x_vals)} - {max(x_vals)}")
                print(f"  Mean range: {min(means):.3f} - {max(means):.3f}")
            
            ax.set_xlabel('Penalty', fontsize=14, fontweight='bold')
            ax.set_ylabel('Next Cell Lava Proportion', fontsize=14, fontweight='bold')
            ax.set_title(f'{reward_type} - Next Cell Lava Proportion vs Penalty', fontsize=16, fontweight='bold')
            ax.set_xlim(0.5, 10.5)
            ax.set_ylim(lava_ylim)
            ax.set_xticks(penalties)
            ax.grid(True, alpha=0.3)
            
            # Save lava proportion plot
            lava_save_path = os.path.join(save_dir, f"{reward_type.lower()}_next_cell_lava_vs_penalty.png")
            plt.tight_layout()
            plt.savefig(lava_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            save_paths.append(lava_save_path)
            print(f"{reward_type} next cell lava plot saved: {lava_save_path}")
    
    print(f"\nAll penalty analysis plots saved to: {save_dir}")
    print(f"Goal reached y-axis range: {goal_ylim}")
    print(f"Next cell lava y-axis range: {lava_ylim}")
    return save_paths


if __name__ == "__main__":
    print("Generating reward function penalty analysis plots...")
    print("Creating separate plots for each reward function type:")
    print("- Linear Goal Reached vs Penalty")
    print("- Linear Next Cell Lava vs Penalty")
    print("- Exponential Goal Reached vs Penalty")
    print("- Exponential Next Cell Lava vs Penalty")
    print("- Sigmoid Goal Reached vs Penalty")
    print("- Sigmoid Next Cell Lava vs Penalty")
    print("Colors: Linear (magma 0.2), Exponential (magma 0.5), Sigmoid (magma 0.8)")
    print("Confidence bands: Mean ± 1 standard deviation")
    print("Consistent axis scaling within each plot type\n")
    
    save_paths = create_penalty_analysis_plots()
    
    if save_paths:
        print(f"\nReward function penalty analysis plots completed! Generated {len(save_paths)} plots.")
    else:
        print("Failed to generate penalty analysis plots.") 