#!/usr/bin/env python3
"""
Script: visualize_circuit_results.py

Creates visualization plots for circuit verification results showing how logits change
across the first 30 cumulative experiments for different metrics and input examples.

This script creates a 5x3 grid of plots where:
- Rows correspond to different metrics (first 5 metrics found)
- Columns correspond to different input examples (first 3 examples found)
- Each plot shows logit trajectories for the first 30 experiments
- Colors correspond to logit indices (consistent across all plots)
- Uses magma colormap for easy visual comparison
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.colors as mcolors

# Add the Neuron_Selection directory to the path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
neuron_selection_dir = os.path.dirname(script_dir)
if neuron_selection_dir not in sys.path:
    sys.path.insert(0, neuron_selection_dir)


def load_experiment_results(results_file: Path) -> Dict[str, Any]:
    """
    Load experiment results from a JSON file.
    
    Args:
        results_file: Path to the results JSON file
        
    Returns:
        Dictionary containing the experiment results
    """
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results file {results_file}: {e}")
        return {}


def extract_logits_from_experiments(data: Dict[str, Any], max_experiments: int = 30) -> List[List[float]]:
    """
    Extract logits from experiment data for the first N experiments.
    Includes baseline logits as the zeroth entry.
    
    Args:
        data: Experiment results data
        max_experiments: Maximum number of experiments to extract
        
    Returns:
        List of logit arrays, one for each experiment (starting with baseline)
    """
    logits_by_experiment = []
    
    # Get baseline logits from any experiment (they should all have the same baseline)
    baseline_logits = None
    experiment_keys = list(data.keys())
    
    if experiment_keys:
        # Get baseline from the first experiment
        first_exp_data = data[experiment_keys[0]]
        if 'results' in first_exp_data:
            results = first_exp_data['results']
            if 'baseline_output' in results and results['baseline_output']:
                baseline_logits = results['baseline_output'][0]
        elif 'baseline_logits' in first_exp_data:
            baseline_logits = first_exp_data['baseline_logits']
    
    # Add baseline as experiment 0
    if baseline_logits:
        logits_by_experiment.append(baseline_logits)
    
    # Sort experiments by name for consistent ordering and take first max_experiments
    # Handle both cumulative experiments and individual neuron experiments
    if any(k.startswith('cumulative_') for k in experiment_keys):
        # Use cumulative experiments if available
        experiment_names = sorted([k for k in experiment_keys if k.startswith('cumulative_')], 
                                 key=lambda x: int(x.split('_')[1]))
    else:
        # Use individual neuron experiments, sorted alphabetically for consistency
        experiment_names = sorted([k for k in experiment_keys 
                                 if not k.startswith('baseline')])  # Exclude any baseline-only entries
    
    # Take only the first max_experiments
    experiment_names = experiment_names[:max_experiments]
    
    for exp_name in experiment_names:
        if exp_name in data:
            exp_data = data[exp_name]
            
            # Try different possible locations for logits
            logits = None
            if 'results' in exp_data:
                results = exp_data['results']
                if 'patched_output' in results and results['patched_output']:
                    logits = results['patched_output'][0]  # Take first element if it's a list of lists
                elif 'baseline_output' in results and results['baseline_output']:
                    logits = results['baseline_output'][0]  # Fallback to baseline
            elif 'patched_logits' in exp_data:
                logits = exp_data['patched_logits']
            elif 'baseline_logits' in exp_data:
                logits = exp_data['baseline_logits']
            
            if logits:
                logits_by_experiment.append(logits)
    
    return logits_by_experiment


def create_circuit_visualization(
    agent_path: str,
    output_dir: str = None,
    max_experiments: int = 30,
    max_metrics: int = 5,
    max_examples: int = 3
) -> None:
    """
    Create circuit verification visualization plots.
    
    Args:
        agent_path: Path to the agent directory
        output_dir: Directory to save plots (defaults to circuit_verification/plots)
        max_experiments: Maximum number of experiments to plot
        max_metrics: Maximum number of metrics to include
        max_examples: Maximum number of input examples to include
    """
    agent_path = Path(agent_path)
    results_dir = agent_path / "circuit_verification" / "results"
    
    if output_dir is None:
        output_dir = agent_path / "circuit_verification" / "plots"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Find all metric directories
    metric_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name != "experiment_summary.json"]
    metric_dirs = sorted(metric_dirs)[:max_metrics]
    
    if len(metric_dirs) == 0:
        print(f"No metric directories found in {results_dir}")
        return
    
    print(f"Found {len(metric_dirs)} metric directories: {[d.name for d in metric_dirs]}")
    
    # Process both noising and denoising
    for experiment_type in ["noising", "denoising"]:
        print(f"\nProcessing {experiment_type} experiments...")
        
        # Collect data for all metrics and examples
        all_data = {}
        all_examples = set()
        # Track min/max per environment for consistent scaling
        environment_limits = {}
        
        for metric_dir in metric_dirs:
            metric_name = metric_dir.name
            type_dir = metric_dir / experiment_type
            
            if not type_dir.exists():
                print(f"Warning: {experiment_type} directory not found for {metric_name}")
                continue
            
            # Find result files (should be common across metrics)
            result_files = list(type_dir.glob("*.json"))
            result_files = sorted(result_files)[:max_examples]
            
            all_data[metric_name] = {}
            
            for result_file in result_files:
                example_name = result_file.stem
                all_examples.add(example_name)
                
                # Load and extract logits
                data = load_experiment_results(result_file)
                logits_by_experiment = extract_logits_from_experiments(data, max_experiments)
                
                if logits_by_experiment:
                    all_data[metric_name][example_name] = logits_by_experiment
                    
                    # Track min/max per environment
                    if example_name not in environment_limits:
                        environment_limits[example_name] = {'min': float('inf'), 'max': float('-inf')}
                    
                    for logits in logits_by_experiment:
                        if logits:
                            env_min = min(logits)
                            env_max = max(logits)
                            environment_limits[example_name]['min'] = min(environment_limits[example_name]['min'], env_min)
                            environment_limits[example_name]['max'] = max(environment_limits[example_name]['max'], env_max)
        
        # Convert to sorted list for consistent ordering
        all_examples = sorted(list(all_examples))[:max_examples]
        
        if not all_data or not all_examples:
            print(f"No data found for {experiment_type}")
            continue
        
        # Check if we have any valid logits data
        has_valid_data = False
        for metric_name in all_data:
            for example_name in all_data[metric_name]:
                if all_data[metric_name][example_name]:
                    has_valid_data = True
                    break
            if has_valid_data:
                break
        
        if not has_valid_data:
            print(f"No valid logits data found for {experiment_type}")
            continue
        
        # Set reasonable defaults for environments with invalid limits
        for example_name in environment_limits:
            if environment_limits[example_name]['min'] == float('inf'):
                environment_limits[example_name]['min'] = 0.0
            if environment_limits[example_name]['max'] == float('-inf'):
                environment_limits[example_name]['max'] = 1.0
        
        # Create the visualization - 5 metrics (rows) x 3 environments (columns)
        # Use wider aspect ratio for better readability
        fig, axes = plt.subplots(len(metric_dirs), len(all_examples), 
                                figsize=(6*len(all_examples), 3*len(metric_dirs)))
        
        # Handle case where we have only one row or column
        if len(metric_dirs) == 1:
            axes = axes.reshape(1, -1)
        elif len(all_examples) == 1:
            axes = axes.reshape(-1, 1)
        elif len(metric_dirs) == 1 and len(all_examples) == 1:
            axes = np.array([[axes]])
        
        # Set up colormap for logit indices
        n_logits = 5  # Assuming 5 actions based on the data
        colors = plt.cm.magma(np.linspace(0.1, 0.9, n_logits))
        
        # Create plots
        for i, metric_name in enumerate([d.name for d in metric_dirs]):
            for j, example_name in enumerate(all_examples):
                ax = axes[i, j]
                
                if metric_name in all_data and example_name in all_data[metric_name]:
                    logits_by_experiment = all_data[metric_name][example_name]
                    
                    # Plot each logit index as a separate line
                    for logit_idx in range(n_logits):
                        logit_values = []
                        experiment_indices = []
                        
                        for exp_idx, logits in enumerate(logits_by_experiment):
                            if logits and len(logits) > logit_idx:
                                logit_values.append(logits[logit_idx])
                                experiment_indices.append(exp_idx)  # Start from 0 (baseline)
                        
                        if logit_values:
                            ax.plot(experiment_indices, logit_values, 
                                   color=colors[logit_idx], 
                                   marker='o', 
                                   markersize=3,
                                   linewidth=2,
                                   label=f'Logit {logit_idx}')
                
                # Set consistent axis limits per environment (same for all metrics in this column)
                ax.set_xlim(-0.5, max_experiments + 0.5)  # Start from 0 (baseline)
                env_min = environment_limits[example_name]['min']
                env_max = environment_limits[example_name]['max']
                margin = (env_max - env_min) * 0.1 if env_max > env_min else 0.1
                ax.set_ylim(env_min - margin, env_max + margin)
                
                # Set labels
                if i == len(metric_dirs) - 1:  # Only add x-label to bottom plots
                    ax.set_xlabel('Experiment Index (0=Baseline)', fontsize=10)
                if j == 0:  # Only add y-label to leftmost plots
                    ax.set_ylabel('Logit Value', fontsize=10)
                
                # Set title - environment name only on top row
                if i == 0:  # Top row - show environment name
                    ax.set_title(f'{example_name}', fontsize=11, fontweight='bold', pad=15)
                
                ax.grid(True, alpha=0.3)
        
        # Add metric names as row labels on the left side with letters
        metric_letters = []
        for i, metric_name in enumerate([d.name for d in metric_dirs]):
            letter = chr(ord('A') + i)  # A, B, C, D, E, etc.
            metric_letters.append((letter, metric_name))
            # Calculate the exact y position to align with the center of each row of plots
            # Account for the subplot positioning within the figure
            plot_top = 0.88  # top of plot area from subplots_adjust
            plot_bottom = 0.18  # bottom of plot area from subplots_adjust
            plot_height = plot_top - plot_bottom
            row_height = plot_height / len(metric_dirs)
            row_center_y = plot_top - (i + 0.5) * row_height
            fig.text(0.02, row_center_y, letter, 
                    rotation=0, verticalalignment='center', horizontalalignment='center', 
                    fontsize=14, fontweight='bold')
        
        # Create a single legend for the entire figure (colors at bottom right)
        # Use the first subplot to get the legend handles and labels
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            # Place color legend at the bottom right, aligned with right edge of plots
            # Use left margin (0.12) + plot width to align with right edge, then subtract spacing
            plot_left = 0.12  # left margin from subplots_adjust
            plot_right = 0.95  # right margin from subplots_adjust
            legend_x = plot_right - 0.02  # right edge minus small spacing
            color_legend = fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(legend_x, 0.08), 
                      title='Logit Index', title_fontsize=11, fontsize=9, ncol=1)
        
        # Add metric mapping as a simple vertical list aligned with left edge of plots
        metric_text_lines = ["Metrics:"] + [f"{letter}: {metric}" for letter, metric in metric_letters]
        metric_text = "\n".join(metric_text_lines)
        # Align with left edge of plots plus small spacing
        metric_x = plot_left + 0.01  # left edge plus small spacing
        fig.text(metric_x, 0.08, metric_text, ha='left', va='bottom', fontsize=10)
        
        # Add overall title with more space
        fig.suptitle(f'Circuit Verification Results - {experiment_type.title()}\n{len(metric_dirs)} Metrics × {len(all_examples)} Environments (0=Baseline)', 
                    fontsize=14, y=0.95)
        
        # Adjust layout with better spacing to accommodate bottom legends
        plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.18, 
                           hspace=0.35, wspace=0.25)
        
        # Save the plot
        output_file = output_dir / f"circuit_verification_{experiment_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {experiment_type} visualization to: {output_file}")
    
    print(f"\nVisualization complete! Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize circuit verification results")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--output_dir", type=str,
                       help="Directory to save plots (default: circuit_verification/plots)")
    parser.add_argument("--max_experiments", type=int, default=30,
                       help="Maximum number of experiments to plot (default: 30)")
    parser.add_argument("--max_metrics", type=int, default=5,
                       help="Maximum number of metrics to include (default: 5)")
    parser.add_argument("--max_examples", type=int, default=3,
                       help="Maximum number of input examples to include (default: 3)")
    
    args = parser.parse_args()
    
    create_circuit_visualization(
        args.agent_path,
        args.output_dir,
        args.max_experiments,
        args.max_metrics,
        args.max_examples
    )


if __name__ == "__main__":
    main() 