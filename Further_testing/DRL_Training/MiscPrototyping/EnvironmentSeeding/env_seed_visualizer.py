#!/usr/bin/env python3
"""
Environment Seed Visualizer

This script allows you to visualize a range of environments with incremental seeds.
It generates a grid of rendered environments to help understand how seed values
affect environment generation.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Visualize environments with different seeds")
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0",
                        help="Environment ID (default: MiniGrid-Empty-8x8-v0)")
    parser.add_argument("--base-seed", type=int, default=0,
                        help="Base seed to start from (default: 0)")
    parser.add_argument("--num-envs", type=int, default=9,
                        help="Number of environments to visualize (default: 9)")
    parser.add_argument("--increment", type=int, default=1,
                        help="Seed increment between environments (default: 1)")
    parser.add_argument("--rows", type=int, default=3,
                        help="Number of rows in the visualization grid (default: 3)")
    parser.add_argument("--cols", type=int, default=3,
                        help="Number of columns in the visualization grid (default: 3)")
    parser.add_argument("--tile-size", type=int, default=32,
                        help="Size of each grid cell in pixels (default: 32)")
    parser.add_argument("--output", type=str, default="env_seeds.png",
                        help="Output file path (default: env_seeds.png)")
    return parser.parse_args()

def render_env(env_id, seed, tile_size=32):
    """Render an environment with a specific seed"""
    env = gym.make(env_id, render_mode=None)
    env = RGBImgObsWrapper(env, tile_size=tile_size)
    obs, _ = env.reset(seed=seed)
    
    # Get the RGB image
    img = env.get_frame(highlight=True, tile_size=tile_size)
    env.close()
    
    return img

def visualize_environments(env_id, base_seed, num_envs, increment, rows, cols, tile_size, output_path):
    """Generate a grid of environment visualizations with different seeds"""
    # Calculate seeds
    seeds = [base_seed + i * increment for i in range(num_envs)]
    
    # Create figure
    figsize = (cols * 4, rows * 4)  # Adjust size as needed
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten axes for easier indexing if multiple rows/cols
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Render each environment
    for i, seed in enumerate(seeds):
        if i < len(axes):
            img = render_env(env_id, seed, tile_size)
            axes[i].imshow(img)
            axes[i].set_title(f"Seed: {seed}")
            axes[i].axis("off")
    
    # Hide any unused axes
    for i in range(num_envs, len(axes)):
        axes[i].axis("off")
    
    # Add overall title
    plt.suptitle(f"Environment: {env_id} (Increment: {increment})", fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")
    
    # Optionally show it if running interactively
    # plt.show()

def main():
    """Main function"""
    args = parse_args()
    
    # Adjust rows/cols if necessary
    if args.rows * args.cols < args.num_envs:
        print(f"Warning: Grid size ({args.rows}x{args.cols}) is smaller than number of environments ({args.num_envs})")
        print("Adjusting grid size...")
        
        # Calculate minimum grid size needed
        total_cells = args.rows * args.cols
        while total_cells < args.num_envs:
            if args.cols <= args.rows:
                args.cols += 1
            else:
                args.rows += 1
            total_cells = args.rows * args.cols
        
        print(f"New grid size: {args.rows}x{args.cols}")
    
    visualize_environments(
        args.env_id,
        args.base_seed,
        args.num_envs,
        args.increment,
        args.rows,
        args.cols,
        args.tile_size,
        args.output
    )

if __name__ == "__main__":
    main() 