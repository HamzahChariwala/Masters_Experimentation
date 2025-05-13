#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob
import re
from scipy.ndimage import gaussian_filter1d

def extract_tensorboard_data(log_dir):
    """Extract data from TensorBoard logs."""
    print(f"Extracting data from {log_dir}...")
    
    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {log_dir}")
    
    # Use the latest event file
    event_files.sort()
    event_file = event_files[-1]
    print(f"Using event file: {os.path.basename(event_file)}")
    
    # Load the event file
    ea = event_accumulator.EventAccumulator(event_file, 
                                           size_guidance={
                                               event_accumulator.SCALARS: 0,
                                           })
    ea.Reload()
    
    # Get list of available tags
    tags = ea.Tags()['scalars']
    print(f"Available tags: {tags}")
    
    # Extract reward and length data
    data = {}
    
    # Common patterns for reward and episode length
    reward_patterns = [
        'rollout/ep_rew_mean', 
        'charts/episodic_return', 
        'eval/mean_reward',
        'train/reward'
    ]
    
    length_patterns = [
        'rollout/ep_len_mean', 
        'charts/episodic_length', 
        'eval/mean_ep_length',
        'train/ep_len_mean'
    ]
    
    # Find the reward tag
    reward_tag = None
    for pattern in reward_patterns:
        matches = [tag for tag in tags if pattern in tag]
        if matches:
            reward_tag = matches[0]
            break
    
    # Find the episode length tag
    length_tag = None
    for pattern in length_patterns:
        matches = [tag for tag in tags if pattern in tag]
        if matches:
            length_tag = matches[0]
            break
    
    if reward_tag:
        print(f"Found reward tag: {reward_tag}")
        reward_events = ea.Scalars(reward_tag)
        data['reward'] = {
            'steps': [event.step for event in reward_events],
            'values': [event.value for event in reward_events],
            'wall_time': [event.wall_time for event in reward_events]
        }
    else:
        print("Warning: No reward data found!")
        
    if length_tag:
        print(f"Found length tag: {length_tag}")
        length_events = ea.Scalars(length_tag)
        data['length'] = {
            'steps': [event.step for event in length_events],
            'values': [event.value for event in length_events],
            'wall_time': [event.wall_time for event in length_events]
        }
    else:
        print("Warning: No episode length data found!")
    
    return data

def plot_metric(data, metric_name, output_dir, smoothing=0):
    """Plot a metric over time and save the plot."""
    if metric_name not in data:
        print(f"No {metric_name} data to plot")
        return
    
    # Prepare data
    steps = np.array(data[metric_name]['steps'])
    values = np.array(data[metric_name]['values'])
    
    # Apply smoothing if requested
    if smoothing > 0:
        values_smooth = gaussian_filter1d(values, sigma=smoothing)
    else:
        values_smooth = values
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot raw data as light scatter points
    plt.scatter(steps, values, alpha=0.3, s=10, label='Raw data')
    
    # Plot smoothed line
    plt.plot(steps, values_smooth, linewidth=2, label='Smoothed')
    
    # Add horizontal line at y=0 for reward plot
    if metric_name == 'reward':
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Labels and title
    plt.xlabel('Training Steps')
    title_map = {
        'reward': 'Mean Episode Reward',
        'length': 'Mean Episode Length'
    }
    ylabel_map = {
        'reward': 'Reward',
        'length': 'Steps'
    }
    plt.ylabel(ylabel_map.get(metric_name, metric_name.capitalize()))
    plt.title(title_map.get(metric_name, metric_name.capitalize()))
    
    # Add legend
    plt.legend()
    
    # Save the figure
    output_file = os.path.join(output_dir, f"{metric_name}_plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {metric_name} plot to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Visualize TensorBoard logs")
    parser.add_argument("--log_dir", type=str, required=True, 
                        help="Directory containing TensorBoard logs")
    parser.add_argument("--output_dir", type=str, default="./plots",
                        help="Directory to save output plots (default: ./plots)")
    parser.add_argument("--smoothing", type=float, default=5,
                        help="Smoothing factor for the plots (default: 5, 0 for no smoothing)")
    parser.add_argument("--show", action="store_true",
                        help="Show plots after saving")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract data from TensorBoard logs
    data = extract_tensorboard_data(args.log_dir)
    
    # Plot metrics
    reward_plot = plot_metric(data, 'reward', args.output_dir, args.smoothing)
    length_plot = plot_metric(data, 'length', args.output_dir, args.smoothing)
    
    # Show plots if requested
    if args.show:
        plt.show()
    else:
        plt.close('all')
    
    print("Done!")
    
    # Return paths to generated plots
    return [p for p in [reward_plot, length_plot] if p]

if __name__ == "__main__":
    main() 