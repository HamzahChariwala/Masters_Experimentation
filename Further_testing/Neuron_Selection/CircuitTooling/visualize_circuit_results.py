#!/usr/bin/env python3
"""
Script: visualize_circuit_results.py

Creates visualization plots for circuit verification results showing how logits change
across the first 30 cumulative experiments for different metrics and input examples.

This script creates separate plots for each metric where:
- Each plot has 2 columns (noising vs denoising) and 5 rows (different environments/states)
- Each subplot shows logit trajectories for the first 30 experiments
- Colors correspond to logit indices (consistent across all plots)
- Same environment/state combinations have consistent axis scaling across noising/denoising
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


def format_environment_name(raw_name: str) -> str:
    """
    Clean up environment names for display by removing common prefixes and suffixes.
    
    Args:
        raw_name: Original environment name (e.g., "MiniGrid_LavaCrossingS11N5_v0_81102_7_8_0_0106")
        
    Returns:
        Cleaned environment name (e.g., "S11N5_v0_81102_7_8_0")
    """
    # Remove "MiniGrid_LavaCrossing" prefix if present
    if raw_name.startswith("MiniGrid_LavaCrossing"):
        name = raw_name[len("MiniGrid_LavaCrossing"):]
    else:
        name = raw_name
    
    # Remove the last "_" and four digits if the pattern matches
    # Look for pattern like "_1234" at the end
    import re
    name = re.sub(r'_\d{4}$', '', name)
    
    return name


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
        # For descending circuit results, experiments are NOT cumulative - they are individual patches
        # The alphabetical sorting creates false patterns in the plots
        # We should either:
        # 1. Skip plotting if there are no cumulative experiments, OR
        # 2. Sort by some meaningful order (e.g., number of neurons patched)
        
        # For now, let's sort by the number of neurons being patched to get a meaningful progression
        def count_neurons_in_experiment(exp_key):
            """Count total neurons being patched in this experiment"""
            count = 0
            # Handle patterns like "q_net.4_neuron_2" (1 neuron) or "q_net.4_neurons_0_1_2_3_4" (5 neurons)
            if '_neuron_' in exp_key:
                count += 1
            elif '_neurons_' in exp_key:
                # Count numbers after the last '_neurons_'
                neurons_part = exp_key.split('_neurons_')[-1]
                count += len([x for x in neurons_part.split('_') if x.isdigit()])
            # Handle patterns like "q_net.4_5n_q_net.0_10n_q_net.2_9n" (5+10+9=24 neurons)
            elif '_n' in exp_key:
                parts = exp_key.split('_')
                for i, part in enumerate(parts):
                    if part.endswith('n') and part[:-1].isdigit():
                        count += int(part[:-1])
            return count
        
        experiment_names = sorted([k for k in experiment_keys if not k.startswith('baseline')], 
                                 key=count_neurons_in_experiment)
    
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
    max_metrics: int = 10,
    max_examples: int = 5,
    subfolder: str = "results"
) -> None:
    """
    Create visualizations for circuit verification results.
    
    Args:
        agent_path: Path to the agent directory
        output_dir: Directory to save plots (default: circuit_verification/descending/plots)
        max_experiments: Maximum number of experiments to plot
        max_metrics: Maximum number of metrics to include
        max_examples: Maximum number of input examples to include
        subfolder: Subfolder name within circuit_verification for results (default: "results")
    """
    agent_path = Path(agent_path)
    if subfolder == "descending":
        results_dir = agent_path / "circuit_verification" / "descending" / "results"
    elif subfolder == "descending/results":
        results_dir = agent_path / "circuit_verification" / "descending" / "results"
    else:
        results_dir = agent_path / "circuit_verification" / subfolder
    
    if output_dir is None:
        output_dir = agent_path / "circuit_verification" / "descending" / "plots"
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
    
    # Collect data for all metrics and examples
    all_data = {}
    all_examples = [
        "MiniGrid_LavaCrossingS11N5_v0_81109_2_3_1_0115",
        "MiniGrid_LavaCrossingS11N5_v0_81103_4_1_0_0026", 
        "MiniGrid_LavaCrossingS11N5_v0_81104_5_2_0_0253",
        "MiniGrid_LavaCrossingS11N5_v0_81109_7_5_0_0145",
        "MiniGrid_LavaCrossingS11N5_v0_81102_7_8_0_0106"
    ]
    # Track min/max per environment for consistent scaling across noising/denoising
    environment_limits = {}
    
    # First pass: collect all data and track limits
    for metric_dir in metric_dirs:
        metric_name = metric_dir.name
        all_data[metric_name] = {}
        
        for experiment_type in ["noising", "denoising"]:
            type_dir = metric_dir / experiment_type
            
            if not type_dir.exists():
                print(f"Warning: {experiment_type} directory not found for {metric_name}")
                continue
            
            # Find result files (should be common across metrics)
            result_files = list(type_dir.glob("*.json"))
            result_files = sorted(result_files)[:max_examples]
            
            all_data[metric_name][experiment_type] = {}
            
            for result_file in result_files:
                example_name = result_file.stem
                
                # Load and extract logits
                data = load_experiment_results(result_file)
                logits_by_experiment = extract_logits_from_experiments(data, max_experiments)
                
                if logits_by_experiment:
                    all_data[metric_name][experiment_type][example_name] = logits_by_experiment
                    
                    # Track min/max per environment across both noising and denoising
                    if example_name not in environment_limits:
                        environment_limits[example_name] = {'min': float('inf'), 'max': float('-inf')}
                    
                    for logits in logits_by_experiment:
                        if logits:
                            env_min = min(logits)
                            env_max = max(logits)
                            environment_limits[example_name]['min'] = min(environment_limits[example_name]['min'], env_min)
                            environment_limits[example_name]['max'] = max(environment_limits[example_name]['max'], env_max)
    
    if not all_data:
        print(f"No data found")
        return
    
    # Set reasonable defaults for environments with invalid limits
    for example_name in environment_limits:
        if environment_limits[example_name]['min'] == float('inf'):
            environment_limits[example_name]['min'] = 0.0
        if environment_limits[example_name]['max'] == float('-inf'):
            environment_limits[example_name]['max'] = 1.0
    
    # Set up colormap for logit indices
    n_logits = 5  # Assuming 5 actions based on the data
    colors = plt.cm.magma(np.linspace(0.1, 0.9, n_logits))
    
    # Create separate plot for each metric
    for metric_dir in metric_dirs:
        metric_name = metric_dir.name
        
        # Check if we have any valid logits data for this metric
        has_valid_data = False
        for exp_type in ["noising", "denoising"]:
            if exp_type in all_data[metric_name]:
                for example_name in all_data[metric_name][exp_type]:
                    if all_data[metric_name][exp_type][example_name]:
                        has_valid_data = True
                        break
                if has_valid_data:
                    break
        
        if not has_valid_data:
            print(f"No valid logits data found for metric {metric_name}")
            continue
        
        print(f"Creating plot for metric: {metric_name}")
        
        # Create the visualization - N environments (rows) x 2 experiment types (columns)
        fig, axes = plt.subplots(len(all_examples), 2, 
                                figsize=(12, 3.5*len(all_examples)))
        
        # Handle case where we have only one row
        if len(all_examples) == 1:
            axes = axes.reshape(1, -1)
        
        experiment_types = ["noising", "denoising"]
        
        # Create plots
        for i, example_name in enumerate(all_examples):
            formatted_name = format_environment_name(example_name)
            
            for j, experiment_type in enumerate(experiment_types):
                ax = axes[i, j]
                
                # Check if we have data for this combination
                if (experiment_type in all_data[metric_name] and 
                    example_name in all_data[metric_name][experiment_type]):
                    
                    logits_by_experiment = all_data[metric_name][experiment_type][example_name]
                    
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
                
                # Set consistent axis limits per environment (same across noising/denoising)
                ax.set_xlim(-0.5, max_experiments + 0.5)  # Start from 0 (baseline)
                if example_name in environment_limits:
                    env_min = environment_limits[example_name]['min']
                    env_max = environment_limits[example_name]['max']
                    margin = (env_max - env_min) * 0.1 if env_max > env_min else 0.1
                    ax.set_ylim(env_min - margin, env_max + margin)
                
                # Set labels
                if i == len(all_examples) - 1:  # Only add x-label to bottom plots
                    ax.set_xlabel('Experiment Index (0=Baseline)', fontsize=10)
                if j == 0:  # Only add y-label to leftmost plots
                    ax.set_ylabel('Logit Value', fontsize=10)
                
                # Set title
                if i == 0:  # Top row - show experiment type
                    ax.set_title(f'{experiment_type.title()}', fontsize=12, fontweight='bold', pad=15)
                
                # Add environment name on the left for leftmost column with better formatting
                if j == 0:
                    ax.text(-0.2, 0.5, formatted_name, rotation=90, verticalalignment='center', 
                           horizontalalignment='center', fontsize=10, fontweight='bold',
                           transform=ax.transAxes)
                
                ax.grid(True, alpha=0.3)
        
        # Create legend for this metric plot
        # Get handles and labels from the first subplot that has data
        handles, labels = None, None
        for i in range(len(all_examples)):
            for j in range(2):
                h, l = axes[i, j].get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            if handles:
                break
        
        if handles:
            # Place legend at the bottom right
            fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, 0.02), 
                      title='Logit Index', title_fontsize=11, fontsize=9, ncol=5)
        
        # Add overall title
        fig.suptitle(f'Circuit Verification Results - {metric_name}\n{len(all_examples)} Environments × 2 Experiment Types (0=Baseline)', 
                    fontsize=14, y=0.95)
        
        # Adjust layout with better spacing to accommodate more environments
        plt.subplots_adjust(left=0.22, right=0.95, top=0.88, bottom=0.08, 
                           hspace=0.25, wspace=0.15)
        
        # Save the plot
        # Replace any problematic characters in metric name for filename
        safe_metric_name = metric_name.replace("/", "_").replace("\\", "_")
        output_file = output_dir / f"circuit_verification_{safe_metric_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {metric_name} visualization to: {output_file}")
    
    print(f"\nVisualization complete! Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize circuit verification results")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--output_dir", type=str,
                       help="Directory to save plots (default: circuit_verification/descending/plots)")
    parser.add_argument("--max_experiments", type=int, default=30,
                       help="Maximum number of experiments to plot (default: 30)")
    parser.add_argument("--max_metrics", type=int, default=10,
                       help="Maximum number of metrics to include (default: 10)")
    parser.add_argument("--max_examples", type=int, default=5,
                       help="Maximum number of input examples to include (default: 5)")
    parser.add_argument("--subfolder", type=str, default="results",
                       help="Subfolder name within circuit_verification for results (default: results)")
    
    args = parser.parse_args()
    
    create_circuit_visualization(
        args.agent_path,
        args.output_dir,
        args.max_experiments,
        args.max_metrics,
        args.max_examples,
        subfolder=args.subfolder
    )


if __name__ == "__main__":
    main() 