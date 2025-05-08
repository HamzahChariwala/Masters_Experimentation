#!/usr/bin/env python

"""
Utility script to visualize curriculum learning evolution with spawn distributions.

This script demonstrates how spawn distributions change during the training process,
both for stage-based training and continuous temporal transitions.

Run this script to generate visualization images and animations of curriculum evolution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from .spawn_distributions import DistributionMap

def visualize_stage_based_curriculum(output_dir="curriculum_visualizations", 
                                     grid_size=(8, 8), 
                                     goal_pos=(7, 7),
                                     num_stages=4):
    """
    Visualize stage-based curriculum learning with spawn distributions.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save visualizations
    grid_size : tuple
        Size of the grid (height, width)
    goal_pos : tuple
        Position of the goal (y, x)
    num_stages : int
        Number of training stages to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the stage-based curriculum
    stages = [
        {"type": "poisson_goal", "params": {"lambda_param": 1.0, "favor_near": True}},
        {"type": "distance_goal", "params": {"favor_near": True, "power": 1}},
        {"type": "gaussian_goal", "params": {"sigma": 2.0, "favor_near": False}},
        {"type": "uniform", "params": {}}
    ]
    
    # Ensure we have the right number of stages
    stages = stages[:num_stages]
    while len(stages) < num_stages:
        stages.append({"type": "uniform", "params": {}})
    
    # Create a figure to hold all stage visualizations
    fig, axes = plt.subplots(1, num_stages, figsize=(5*num_stages, 5))
    fig.suptitle("Stage-based Curriculum Learning: Spawn Distributions", fontsize=16)
    
    # Generate and plot each stage distribution
    distributions = []
    
    for i, (stage_config, ax) in enumerate(zip(stages, axes)):
        print(f"Generating Stage {i+1} distribution...")
        dist_map = DistributionMap(grid_size)
        
        # Set distribution based on stage type
        if stage_config["type"] == "uniform":
            dist_map.uniform()
        elif stage_config["type"] == "poisson_goal":
            params = stage_config["params"]
            dist_map.poisson_from_point(
                goal_pos, 
                lambda_param=params.get("lambda_param", 1.0),
                favor_near=params.get("favor_near", True)
            )
        elif stage_config["type"] == "gaussian_goal":
            params = stage_config["params"]
            dist_map.gaussian_from_point(
                goal_pos,
                sigma=params.get("sigma", 2.0),
                favor_near=params.get("favor_near", True)
            )
        elif stage_config["type"] == "distance_goal":
            params = stage_config["params"]
            dist_map.distance_from_point(
                goal_pos,
                power=params.get("power", 1),
                favor_near=params.get("favor_near", True)
            )
        
        distributions.append(dist_map)
        
        # Plot the distribution
        title = f"Stage {i+1}: {stage_config['type']}"
        if "favor_near" in stage_config["params"]:
            title += "\n" + ("Near Goal" if stage_config["params"]["favor_near"] else "Far from Goal")
        
        dist_map.visualize(title=title, ax=ax, show=False)
        
        # Mark goal position
        ax.plot(goal_pos[1], goal_pos[0], 'r*', markersize=10)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    stages_path = os.path.join(output_dir, "stage_based_curriculum.png")
    plt.savefig(stages_path, dpi=300, bbox_inches='tight')
    print(f"Saved stage-based curriculum visualization to: {stages_path}")
    
    # Create animation to show transition between stages
    print("Generating stage transition animation...")
    transition_frames = 20  # Frames for transition between stages
    
    # Initialize the animation figure
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Stage-based Curriculum Learning Animation", fontsize=16)
    
    # Initialize the probability grid plot
    img_plot = ax.imshow(np.zeros(grid_size), origin='upper', vmin=0, vmax=1, 
                        cmap='viridis', interpolation='nearest')
    goal_plot = ax.plot(goal_pos[1], goal_pos[0], 'r*', markersize=15)[0]
    fig.colorbar(img_plot, ax=ax, label='Spawn Probability')
    title = ax.set_title("Stage 1")
    
    # Function to update the animation frame
    def update_frame(frame_num):
        # Calculate which stages we're transitioning between
        stage_duration = transition_frames
        total_frames = num_stages * stage_duration
        
        # For smooth looping
        frame_num = frame_num % total_frames
        
        current_stage = int(frame_num / stage_duration)
        next_stage = (current_stage + 1) % num_stages
        
        # Calculate blend factor (0 = current stage, 1 = next stage)
        blend = (frame_num % stage_duration) / stage_duration
        
        if blend < 0.1 or blend > 0.9:
            # Just show the current stage at the beginning and end of transition
            prob_grid = distributions[current_stage].probabilities
            stage_text = f"Stage {current_stage + 1}: {stages[current_stage]['type']}"
        else:
            # Blend between current and next stage
            current_grid = distributions[current_stage].probabilities
            next_grid = distributions[next_stage].probabilities
            prob_grid = (1 - blend) * current_grid + blend * next_grid
            
            stage_text = f"Transition: Stage {current_stage + 1} → Stage {next_stage + 1}"
            
        # Update the visualization
        img_plot.set_array(prob_grid)
        title.set_text(stage_text)
        
        return img_plot, title
    
    # Create the animation
    anim = animation.FuncAnimation(fig, update_frame, frames=num_stages*transition_frames,
                                  interval=100, blit=False)
    
    # Save the animation
    animation_path = os.path.join(output_dir, "stage_transition_animation.gif")
    anim.save(animation_path, writer='pillow', fps=10, dpi=100)
    plt.close(fig)
    
    print(f"Saved stage transition animation to: {animation_path}")
    return stages_path, animation_path


def visualize_continuous_transition(output_dir="curriculum_visualizations", 
                                   grid_size=(8, 8), 
                                   goal_pos=(7, 7),
                                   num_frames=20):
    """
    Visualize continuous temporal transition between spawn distributions.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save visualizations
    grid_size : tuple
        Size of the grid (height, width)
    goal_pos : tuple
        Position of the goal (y, x)
    num_frames : int
        Number of frames to generate for the transition
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the initial and target distributions
    initial_config = {"type": "poisson_goal", "params": {"lambda_param": 1.0, "favor_near": True}}
    target_config = {"type": "uniform", "params": {}}
    
    # Create initial and target distribution maps
    print("Generating initial distribution...")
    initial_dist = DistributionMap(grid_size)
    initial_dist.poisson_from_point(goal_pos, lambda_param=1.0, favor_near=True)
    
    print("Generating target distribution...")
    target_dist = DistributionMap(grid_size)
    target_dist.uniform()
    
    # Create a figure to show initial and target
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Continuous Transition: Initial → Target", fontsize=16)
    
    # Plot initial and target distributions
    initial_dist.visualize(title="Initial: Poisson (Near Goal)", ax=axes[0], show=False)
    axes[0].plot(goal_pos[1], goal_pos[0], 'r*', markersize=10)
    
    target_dist.visualize(title="Target: Uniform", ax=axes[1], show=False)
    axes[1].plot(goal_pos[1], goal_pos[0], 'r*', markersize=10)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    endpoints_path = os.path.join(output_dir, "continuous_transition_endpoints.png")
    plt.savefig(endpoints_path, dpi=300, bbox_inches='tight')
    print(f"Saved transition endpoints visualization to: {endpoints_path}")
    
    # Create transition frames visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    fig.suptitle("Continuous Transition: Training Progress", fontsize=16)
    
    # Generate frames at specific progress points
    progress_points = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    progress_points = progress_points[:len(axes)]  # Make sure we don't exceed available axes
    
    for i, progress in enumerate(progress_points):
        if i >= len(axes):
            break
            
        # Create blended distribution
        blend_dist = DistributionMap(grid_size)
        blend_grid = (1 - progress) * initial_dist.probabilities + progress * target_dist.probabilities
        blend_dist.set_probabilities(blend_grid)
        
        # Plot this frame
        title = f"Progress: {progress:.0%}"
        blend_dist.visualize(title=title, ax=axes[i], show=False)
        axes[i].plot(goal_pos[1], goal_pos[0], 'r*', markersize=10)
    
    # Hide any unused axes
    for i in range(len(progress_points), len(axes)):
        axes[i].axis('off')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    frames_path = os.path.join(output_dir, "continuous_transition_frames.png")
    plt.savefig(frames_path, dpi=300, bbox_inches='tight')
    print(f"Saved transition frames visualization to: {frames_path}")
    
    # Create smooth animation of the transition
    print("Generating continuous transition animation...")
    
    # Initialize the animation figure
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Continuous Curriculum Learning Animation", fontsize=16)
    
    # Initialize the probability grid plot
    img_plot = ax.imshow(np.zeros(grid_size), origin='upper', vmin=0, vmax=1, 
                        cmap='viridis', interpolation='nearest')
    goal_plot = ax.plot(goal_pos[1], goal_pos[0], 'r*', markersize=15)[0]
    fig.colorbar(img_plot, ax=ax, label='Spawn Probability')
    title = ax.set_title("Training Progress: 0%")
    
    # Function to update the animation frame
    def update_frame(frame_num):
        # Calculate progress (0 to 1, with oscillation for visualization purposes)
        progress = frame_num / (num_frames - 1)
        
        # Use a slow oscillation for better visualization (0->1->0)
        if frame_num >= num_frames:
            # For the second half, go from 1 back to 0
            progress = 2.0 - (frame_num / (num_frames - 1))
        
        # Blend the distributions
        blended_grid = (1 - progress) * initial_dist.probabilities + progress * target_dist.probabilities
        
        # Update the visualization
        img_plot.set_array(blended_grid)
        title.set_text(f"Training Progress: {progress:.0%}")
        
        return img_plot, title
    
    # Create the animation
    anim = animation.FuncAnimation(fig, update_frame, frames=num_frames*2-2,
                                  interval=100, blit=False)
    
    # Save the animation
    animation_path = os.path.join(output_dir, "continuous_transition_animation.gif")
    anim.save(animation_path, writer='pillow', fps=10, dpi=100)
    plt.close(fig)
    
    print(f"Saved continuous transition animation to: {animation_path}")
    return frames_path, animation_path


if __name__ == "__main__":
    # Create output directory
    output_dir = "curriculum_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate stage-based curriculum visualizations
    print("\n===== STAGE-BASED CURRICULUM =====")
    stage_img, stage_anim = visualize_stage_based_curriculum(output_dir)
    
    # Generate continuous transition visualizations
    print("\n===== CONTINUOUS TRANSITION =====")
    trans_img, trans_anim = visualize_continuous_transition(output_dir)
    
    print("\nAll visualizations saved to:", output_dir)
    print("Stage-based curriculum image:", stage_img)
    print("Stage-based curriculum animation:", stage_anim)
    print("Continuous transition image:", trans_img)
    print("Continuous transition animation:", trans_anim) 