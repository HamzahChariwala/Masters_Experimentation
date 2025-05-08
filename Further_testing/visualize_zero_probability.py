#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from minigrid.envs.crossing import CrossingEnv
from SpawnDistributions.spawn_distributions import FlexibleSpawnWrapper

# Create output directory
output_dir = "zero_prob_visualizations"
os.makedirs(output_dir, exist_ok=True)

def generate_enhanced_visualization():
    """
    Create visualizations that clearly show zero probability areas for goal and lava.
    """
    print("Generating enhanced visualizations showing zero probability areas...")
    
    # Create a crossing environment with lava
    env = CrossingEnv(size=9, num_crossings=1)
    
    # Create a wrapper with each type of distribution
    distribution_types = [
        ("uniform", None),
        ("poisson_goal", {"lambda_param": 1.0, "favor_near": True}),
        ("poisson_goal", {"lambda_param": 1.0, "favor_near": False}),
        ("gaussian_goal", {"sigma": 2.0, "favor_near": True}),
        ("distance_goal", {"favor_near": False, "power": 1})
    ]
    
    for dist_type, params in distribution_types:
        # Create descriptive name for file
        if dist_type == "uniform":
            name_suffix = "uniform"
        else:
            favor = "near" if params.get("favor_near", False) else "far"
            name_suffix = f"{dist_type}_{favor}"
        
        # Create wrapper
        wrapper = FlexibleSpawnWrapper(
            env,
            distribution_type=dist_type,
            distribution_params=params
        )
        
        # Reset to initialize
        wrapper.reset()
        
        # Get the distribution map and mask
        prob_map = wrapper.current_distribution.probabilities
        mask = wrapper.valid_cells_mask
        
        # Get goal and lava positions
        goal_pos = wrapper.goal_pos
        
        lava_positions = []
        grid = wrapper.unwrapped.grid
        for i in range(grid.width):
            for j in range(grid.height):
                cell = grid.get(i, j)
                if cell and cell.type == 'lava':
                    lava_positions.append((i, j))
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the probability distribution
        im = ax.imshow(prob_map, cmap='viridis', origin='upper')
        plt.colorbar(im, ax=ax, label='Probability')
        
        # Highlight the goal position
        if goal_pos:
            rect = patches.Rectangle(
                (goal_pos[0] - 0.5, goal_pos[1] - 0.5), 
                1, 1, 
                linewidth=2, 
                edgecolor='red', 
                facecolor='none',
                label='Goal'
            )
            ax.add_patch(rect)
            ax.text(goal_pos[0], goal_pos[1], 'G', 
                    color='white', ha='center', va='center', fontweight='bold')
        
        # Highlight lava positions
        for pos in lava_positions:
            rect = patches.Rectangle(
                (pos[0] - 0.5, pos[1] - 0.5), 
                1, 1, 
                linewidth=2, 
                edgecolor='orange', 
                facecolor='none',
                label='Lava' if pos == lava_positions[0] else ""
            )
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], 'L', 
                    color='white', ha='center', va='center', fontweight='bold')
        
        # Add a contour showing the zero probability boundary
        zero_mask = (prob_map == 0)
        if np.any(zero_mask):
            ax.contour(zero_mask, levels=[0.5], colors=['red'], linewidths=2, 
                      extent=(-0.5, prob_map.shape[1]-0.5, prob_map.shape[0]-0.5, -0.5))
        
        # Set title and labels
        title = f"{dist_type.capitalize()} Distribution"
        if params:
            if "favor_near" in params:
                title += f" ({'Near' if params['favor_near'] else 'Far from'} Goal)"
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add grid lines
        ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Show coordinate numbers
        ax.set_xticks(np.arange(prob_map.shape[1]))
        ax.set_yticks(np.arange(prob_map.shape[0]))
        
        # Save the visualization
        save_path = os.path.join(output_dir, f"zero_prob_{name_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated visualization for {name_suffix}")
    
    # Also create a visualization with temporal transition
    temporal_config = {
        "target_type": "uniform",
        "rate": 1.0
    }
    
    wrapper = FlexibleSpawnWrapper(
        env,
        distribution_type="poisson_goal",
        distribution_params={"lambda_param": 1.0, "favor_near": True},
        total_timesteps=10000,
        temporal_transition=temporal_config
    )
    
    wrapper.reset()
    
    for progress in [0.0, 0.5, 1.0]:
        # Set timestep 
        timestep = int(10000 * progress)
        wrapper.timestep = timestep
        
        # If this is not the initial state, we need to manually update the distribution
        if progress > 0:
            # Create the target distribution
            width, height = wrapper.current_distribution.width, wrapper.current_distribution.height
            target_dist = wrapper.target_distribution
            
            # Interpolate toward target
            wrapper.current_distribution.temporal_interpolation(target_dist, progress)
            
            # Re-apply valid cells mask
            wrapper.current_distribution.mask_cells(wrapper.valid_cells_mask)
            
            # Rebuild the sampling map
            wrapper.current_distribution.build_sampling_map()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the probability distribution
        prob_map = wrapper.current_distribution.probabilities
        im = ax.imshow(prob_map, cmap='viridis', origin='upper')
        plt.colorbar(im, ax=ax, label='Probability')
        
        # Highlight the goal position
        if wrapper.goal_pos:
            goal_pos = wrapper.goal_pos
            rect = patches.Rectangle(
                (goal_pos[0] - 0.5, goal_pos[1] - 0.5), 
                1, 1, 
                linewidth=2, 
                edgecolor='red', 
                facecolor='none',
                label='Goal'
            )
            ax.add_patch(rect)
            ax.text(goal_pos[0], goal_pos[1], 'G', 
                    color='white', ha='center', va='center', fontweight='bold')
        
        # Highlight lava positions
        lava_positions = []
        grid = wrapper.unwrapped.grid
        for i in range(grid.width):
            for j in range(grid.height):
                cell = grid.get(i, j)
                if cell and cell.type == 'lava':
                    lava_positions.append((i, j))
        
        for pos in lava_positions:
            rect = patches.Rectangle(
                (pos[0] - 0.5, pos[1] - 0.5), 
                1, 1, 
                linewidth=2, 
                edgecolor='orange', 
                facecolor='none',
                label='Lava' if pos == lava_positions[0] else ""
            )
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], 'L', 
                    color='white', ha='center', va='center', fontweight='bold')
        
        # Add a contour showing the zero probability boundary
        zero_mask = (prob_map == 0)
        if np.any(zero_mask):
            ax.contour(zero_mask, levels=[0.5], colors=['red'], linewidths=2, 
                      extent=(-0.5, prob_map.shape[1]-0.5, prob_map.shape[0]-0.5, -0.5))
        
        # Set title and labels
        ax.set_title(f"Temporal Transition (Progress: {progress:.1f})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add grid lines
        ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Show coordinate numbers
        ax.set_xticks(np.arange(prob_map.shape[1]))
        ax.set_yticks(np.arange(prob_map.shape[0]))
        
        # Save the visualization
        save_path = os.path.join(output_dir, f"temporal_progress_{int(progress*100)}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated temporal visualization at progress {progress:.1f}")
    
    print("\nAll enhanced visualizations generated successfully.")
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    generate_enhanced_visualization() 