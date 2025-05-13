#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_comparison_visualization():
    """Create a comparison visualization between actual environment and our distribution visualization."""
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # First subplot: Actual environment
    img_path = Path('MiniGrid_LavaCrossingS11N5_v0_seed_12345.png')
    if img_path.exists():
        img = plt.imread(str(img_path))
        ax1.imshow(img)
        ax1.set_title("Actual Environment: MiniGrid-LavaCrossingS11N5-v0", fontsize=14)
        ax1.axis('off')
    else:
        ax1.text(0.5, 0.5, "Image not found", ha='center', va='center')
        ax1.set_title("Missing Environment Image", fontsize=14)
    
    # Second subplot: Our visualization - text representation
    # Create a representation of our visualization
    our_vis_text = """
    Probability Distribution Array:
        0     1     2     3     4     5     6     7     8     9    10 
    ------------------------------------------------------------------
 0 | 5.55e-09 LAVA  LAVA  LAVA  LAVA  LAVA  2.24e-06 6.08e-06 1.65e-05 4.49e-05 1.65e-05
 1 | 1.51e-08 LAVA  LAVA  LAVA  LAVA  LAVA  6.08e-06 1.65e-05 4.49e-05 1.22e-04 4.49e-05
 2 | 4.10e-08 LAVA  LAVA  LAVA  LAVA  LAVA  1.65e-05 4.49e-05 1.22e-04 3.32e-04 1.22e-04
 3 | 1.11e-07 LAVA  LAVA  LAVA  LAVA  LAVA  4.49e-05 1.22e-04 3.32e-04 9.03e-04 3.32e-04
 4 | 3.03e-07 LAVA  LAVA  LAVA  LAVA  LAVA  1.22e-04 3.32e-04 9.03e-04 2.45e-03 9.03e-04
 5 | 8.23e-07 LAVA  LAVA  LAVA  LAVA  LAVA  3.32e-04 9.03e-04 2.45e-03 6.67e-03 2.45e-03
 6 | 2.24e-06 LAVA  LAVA  LAVA  LAVA  LAVA  9.03e-04 2.45e-03 6.67e-03 1.81e-02 6.67e-03
 7 | 6.08e-06 LAVA  LAVA  LAVA  LAVA  LAVA  2.45e-03 6.67e-03 1.81e-02 4.93e-02 1.81e-02
 8 | 1.65e-05 LAVA  LAVA  LAVA  LAVA  LAVA  6.67e-03 1.81e-02 4.93e-02 1.34e-01 4.93e-02
 9 | 4.49e-05 LAVA  LAVA  LAVA  LAVA  LAVA  1.81e-02 4.93e-02 1.34e-01 GOAL  1.34e-01
10 | 1.65e-05 LAVA  LAVA  LAVA  LAVA  LAVA  6.67e-03 1.81e-02 4.93e-02 1.34e-01 4.93e-02
    """
    
    # Create a visual representation of our distribution
    grid_width, grid_height = 11, 11
    grid = np.zeros((grid_height, grid_width, 4))  # RGBA
    
    # Define colors for different cell types
    colors = {
        'lava': [1.0, 0.0, 0.0, 1.0],       # Red (RGBA)
        'goal': [0.0, 1.0, 0.0, 1.0],       # Green
        'empty_high': [0.0, 0.0, 1.0, 1.0],  # Blue (high probability)
        'empty_med': [0.3, 0.3, 1.0, 1.0],   # Medium blue
        'empty_low': [0.7, 0.7, 1.0, 1.0],   # Light blue (low probability)
        'boundary': [0.5, 0.5, 0.5, 1.0]     # Gray for boundary
    }
    
    # Setup our visualization
    # This matches our original visualization from the previous result
    # Column 0 has very low probabilities
    # Columns 1-5 have lava
    # Column 6-10 have increasing probabilities
    # Goal at (9, 9)
    for y in range(grid_height):
        for x in range(grid_width):
            if x == 9 and y == 9:
                grid[y, x] = colors['goal']
            elif 1 <= x <= 5:
                grid[y, x] = colors['lava']
            elif x == 0:
                grid[y, x] = colors['empty_low']
            elif 6 <= x <= 8:
                grid[y, x] = colors['empty_med']
            elif x == 10:
                grid[y, x] = colors['empty_high']
            else:
                grid[y, x] = colors['empty_low']
    
    # Display our visualization
    ax2.imshow(grid)
    ax2.set_title("Our Distribution Visualization", fontsize=14)
    ax2.grid(True, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(np.arange(-.5, grid_width, 1), minor=True)
    ax2.set_yticks(np.arange(-.5, grid_height, 1), minor=True)
    
    # Add column/row indices
    for i in range(grid_width):
        ax2.text(i, -0.5, str(i), ha='center', va='center')
    for j in range(grid_height):
        ax2.text(-0.5, j, str(j), ha='center', va='center')
    
    # Add legend for our visualization
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=colors['lava'], label='LAVA'),
        plt.Rectangle((0, 0), 1, 1, color=colors['goal'], label='GOAL'),
        plt.Rectangle((0, 0), 1, 1, color=colors['empty_high'], label='High Probability'),
        plt.Rectangle((0, 0), 1, 1, color=colors['empty_med'], label='Medium Probability'),
        plt.Rectangle((0, 0), 1, 1, color=colors['empty_low'], label='Low Probability')
    ]
    ax2.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15))
    
    # Add a title to the whole figure
    fig.suptitle("Comparison: Actual Environment vs. Our Visualization", fontsize=16)
    
    # Add explanation text
    fig.text(0.5, 0.01, """
    Discrepancy: Our visualization shows lava in columns 1-5, but the actual environment has lava in diagonal patterns.
    Our goal position (9,9) is correct, but we don't properly represent the walls or lava pattern.
    """, ha='center', wrap=True, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('visualization_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison visualization to visualization_comparison.png")

if __name__ == "__main__":
    create_comparison_visualization() 