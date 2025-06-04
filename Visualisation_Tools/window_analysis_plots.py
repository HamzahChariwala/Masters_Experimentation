#!/usr/bin/env python3
"""
Generate window sizing analysis plots from leaderboard data.
Creates separate plots for goal reached proportion and next cell lava proportion.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import magma
import os


def create_window_analysis_plots():
    """Create window sizing analysis plots with confidence intervals."""
    
    # Window sizing data extracted from all_states_leaderboard.md
    # Window sizes: 3, 5, 7, 9, 11, 13, 15
    window_sizes = [3, 5, 7, 9, 11, 13, 15]
    total_cells = [size**2 for size in window_sizes]  # 9, 25, 49, 81, 121, 169, 225
    
    # Goal Reached Proportion data (from leaderboard)
    goal_reached_means = [0.8472, 0.8388, 0.8200, 0.7451, 0.7674, 0.7098, 0.7653]
    goal_reached_stds = [0.1102, 0.0766, 0.0367, 0.0877, 0.0563, 0.0552, 0.0910]
    
    # Next Cell Lava Proportion data (from leaderboard)
    lava_proportion_means = [0.3888, 0.3952, 0.4098, 0.4066, 0.4050, 0.4012, 0.3954]
    lava_proportion_stds = [0.0152, 0.0098, 0.0140, 0.0127, 0.0215, 0.0199, 0.0196]
    
    # Create output directory
    save_dir = os.path.join(os.path.dirname(__file__), "window")
    os.makedirs(save_dir, exist_ok=True)
    
    # Colors from magma colormap
    goal_color = magma(0.3)
    lava_color = magma(0.6)  # Darker (lower value in magma = darker)
    
    # Plot 1: Goal Reached Proportion vs Window Size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(window_sizes, goal_reached_means, 'o-', color=goal_color, 
            linewidth=2, markersize=8)
    ax.fill_between(window_sizes, 
                    np.array(goal_reached_means) - np.array(goal_reached_stds),
                    np.array(goal_reached_means) + np.array(goal_reached_stds),
                    alpha=0.3, color=goal_color)
    
    ax.set_xlabel('Window Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Goal Reached Proportion', fontsize=14, fontweight='bold')
    ax.set_title('Goal Reached Proportion vs Window Size', fontsize=16, fontweight='bold')
    ax.set_xticks(window_sizes)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot1_path = os.path.join(save_dir, "goal_reached_vs_window_size.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Next Cell Lava Proportion vs Window Size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(window_sizes, lava_proportion_means, 's-', color=lava_color,
            linewidth=2, markersize=8)
    ax.fill_between(window_sizes,
                    np.array(lava_proportion_means) - np.array(lava_proportion_stds),
                    np.array(lava_proportion_means) + np.array(lava_proportion_stds),
                    alpha=0.3, color=lava_color)
    
    ax.set_xlabel('Window Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Next Cell Lava Proportion', fontsize=14, fontweight='bold')
    ax.set_title('Next Cell Lava Proportion vs Window Size', fontsize=16, fontweight='bold')
    ax.set_xticks(window_sizes)
    ax.set_ylim(0.25, 0.55)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot2_path = os.path.join(save_dir, "lava_proportion_vs_window_size.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Goal Reached Proportion vs Total Cells
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(total_cells, goal_reached_means, 'o-', color=goal_color,
            linewidth=2, markersize=8)
    ax.fill_between(total_cells,
                    np.array(goal_reached_means) - np.array(goal_reached_stds),
                    np.array(goal_reached_means) + np.array(goal_reached_stds),
                    alpha=0.3, color=goal_color)
    
    ax.set_xlabel('Total Cells in Window', fontsize=14, fontweight='bold')
    ax.set_ylabel('Goal Reached Proportion', fontsize=14, fontweight='bold')
    ax.set_title('Goal Reached Proportion vs Total Cells', fontsize=16, fontweight='bold')
    ax.set_xticks(total_cells)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot3_path = os.path.join(save_dir, "goal_reached_vs_total_cells.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Next Cell Lava Proportion vs Total Cells
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(total_cells, lava_proportion_means, 's-', color=lava_color,
            linewidth=2, markersize=8)
    ax.fill_between(total_cells,
                    np.array(lava_proportion_means) - np.array(lava_proportion_stds),
                    np.array(lava_proportion_means) + np.array(lava_proportion_stds),
                    alpha=0.3, color=lava_color)
    
    ax.set_xlabel('Total Cells in Window', fontsize=14, fontweight='bold')
    ax.set_ylabel('Next Cell Lava Proportion', fontsize=14, fontweight='bold')
    ax.set_title('Next Cell Lava Proportion vs Total Cells', fontsize=16, fontweight='bold')
    ax.set_xticks(total_cells)
    ax.set_ylim(0.25, 0.55)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot4_path = os.path.join(save_dir, "lava_proportion_vs_total_cells.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Window sizing analysis plots created successfully!")
    print(f"Plot 1 (Goal vs Window Size): {plot1_path}")
    print(f"Plot 2 (Lava vs Window Size): {plot2_path}")
    print(f"Plot 3 (Goal vs Total Cells): {plot3_path}")
    print(f"Plot 4 (Lava vs Total Cells): {plot4_path}")
    print()
    print("Data Summary:")
    print("Window Sizes:", window_sizes)
    print("Total Cells:", total_cells)
    print("Goal Reached (mean ± std):")
    for i, size in enumerate(window_sizes):
        print(f"  Size {size}: {goal_reached_means[i]:.4f} ± {goal_reached_stds[i]:.4f}")
    print("Next Cell Lava Proportion (mean ± std):")
    for i, size in enumerate(window_sizes):
        print(f"  Size {size}: {lava_proportion_means[i]:.4f} ± {lava_proportion_stds[i]:.4f}")


if __name__ == "__main__":
    create_window_analysis_plots() 