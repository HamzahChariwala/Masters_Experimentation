#!/usr/bin/env python3
"""
Generate NoDeath penalty analysis plots from leaderboard data.
Creates separate plots for goal reached proportion and next cell lava proportion vs penalty.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import magma
import os


def create_nodeath_penalty_analysis_plots():
    """Create NoDeath penalty analysis plots with confidence intervals."""
    
    # NoDeath penalty data extracted from all_states_leaderboard.md
    # Penalty values: 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200
    penalties = [0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200]
    
    # Goal Reached Proportion data (from leaderboard)
    goal_reached_means = [0.8924, 0.8069, 0.8226, 0.8740, 0.7633, 0.7830, 0.7921, 0.8174]
    goal_reached_stds = [0.0338, 0.0848, 0.0700, 0.0575, 0.0730, 0.0407, 0.0438, 0.0661]
    
    # Next Cell Lava Proportion data (from leaderboard)
    lava_proportion_means = [0.4073, 0.4025, 0.3925, 0.3899, 0.3976, 0.3865, 0.3924, 0.3903]
    lava_proportion_stds = [0.0119, 0.0109, 0.0192, 0.0168, 0.0159, 0.0134, 0.0210, 0.0130]
    
    # Create output directory
    save_dir = os.path.join(os.path.dirname(__file__), "nodeath")
    os.makedirs(save_dir, exist_ok=True)
    
    # Colors from magma colormap
    goal_color = magma(0.3)
    lava_color = magma(0.6)
    
    # Plot 1: Goal Reached Proportion vs Penalty
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(penalties, goal_reached_means, 'o-', color=goal_color, 
            linewidth=2, markersize=8)
    ax.fill_between(penalties, 
                    np.array(goal_reached_means) - np.array(goal_reached_stds),
                    np.array(goal_reached_means) + np.array(goal_reached_stds),
                    alpha=0.3, color=goal_color)
    
    ax.set_xlabel('Penalty', fontsize=14, fontweight='bold')
    ax.set_ylabel('Goal Reached Proportion', fontsize=14, fontweight='bold')
    ax.set_title('Goal Reached Proportion vs Penalty (NoDeath Agents)', fontsize=16, fontweight='bold')
    ax.set_xticks(penalties)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot1_path = os.path.join(save_dir, "goal_reached_vs_penalty.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Next Cell Lava Proportion vs Penalty
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(penalties, lava_proportion_means, 's-', color=lava_color,
            linewidth=2, markersize=8)
    ax.fill_between(penalties,
                    np.array(lava_proportion_means) - np.array(lava_proportion_stds),
                    np.array(lava_proportion_means) + np.array(lava_proportion_stds),
                    alpha=0.3, color=lava_color)
    
    ax.set_xlabel('Penalty', fontsize=14, fontweight='bold')
    ax.set_ylabel('Next Cell Lava Proportion', fontsize=14, fontweight='bold')
    ax.set_title('Next Cell Lava Proportion vs Penalty (NoDeath Agents)', fontsize=16, fontweight='bold')
    ax.set_xticks(penalties)
    ax.set_ylim(0.25, 0.55)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot2_path = os.path.join(save_dir, "lava_proportion_vs_penalty.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("NoDeath penalty analysis plots created successfully!")
    print(f"Plot 1 (Goal vs Penalty): {plot1_path}")
    print(f"Plot 2 (Lava vs Penalty): {plot2_path}")
    print()
    print("Data Summary:")
    print("Penalties:", penalties)
    print("Goal Reached (mean ± std):")
    for i, penalty in enumerate(penalties):
        print(f"  Penalty {penalty}: {goal_reached_means[i]:.4f} ± {goal_reached_stds[i]:.4f}")
    print("Next Cell Lava Proportion (mean ± std):")
    for i, penalty in enumerate(penalties):
        print(f"  Penalty {penalty}: {lava_proportion_means[i]:.4f} ± {lava_proportion_stds[i]:.4f}")


if __name__ == "__main__":
    create_nodeath_penalty_analysis_plots() 