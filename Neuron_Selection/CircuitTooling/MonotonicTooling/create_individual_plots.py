#!/usr/bin/env python3
"""
Create Individual Circuit-Style Plots for Monotonic Coalition Results

This script generates individual plots for each metric that exactly match the descending format:
- 5 input pairs (rows) × 2 experiment types (columns)
- Left column: noising, Right column: denoising  
- Shows logit progression as neurons are added to coalition
- Saves plots in each metric's results folder

This version reconstructs the coalition building experiments properly to extract actual logits.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import re
import tempfile
import shutil

# Add the project directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
neuron_selection_dir = os.path.abspath(os.path.join(script_dir, "../.."))
project_root = os.path.abspath(os.path.join(neuron_selection_dir, ".."))

for path in [neuron_selection_dir, project_root]:
    if path not in sys.path:
        sys.path.append(path)


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
    name = re.sub(r'_\d{4}$', '', name)
    
    return name


def load_coalition_progression(summary_file: Path) -> Tuple[List[str], List[str]]:
    """
    Load the final coalition progression from summary.
    
    Returns:
        Tuple of (final_coalition_neurons, input_examples)
    """
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    final_coalition = summary['results']['final_coalition_neurons']
    
    # Get actual input examples from clean_inputs.json file
    agent_path = summary_file.parent.parent.parent.parent
    clean_inputs_file = agent_path / "activation_inputs" / "clean_inputs.json"
    
    # Try to read actual input IDs from the file
    input_examples = []
    if clean_inputs_file.exists():
        with open(clean_inputs_file, 'r') as f:
            clean_inputs_data = json.load(f)
        
        # Take first 5 input IDs for consistency with experiments
        input_examples = list(clean_inputs_data.keys())[:5]
        print(f"Loaded {len(input_examples)} input examples from {clean_inputs_file}")
    
    # If we still don't have examples, this is a critical error
    if not input_examples:
        raise FileNotFoundError(f"Could not find input examples at {clean_inputs_file}")
    
    return final_coalition, input_examples


def cleanup_coalition_results(metric_dir: Path):
    """
    Clean up the massive number of individual coalition building result files.
    """
    # This function was causing data loss - commenting out the cleanup
    # The coalition building files should be preserved for analysis
    pass


def create_experiment_folders(metric_dir: Path, coalition_neurons: List[str], input_examples: List[str]):
    """
    Create experiment folders similar to descending approach and run proper experiments
    to extract actual logits for each coalition size.
    """
    from Neuron_Selection.PatchingTooling.patching_experiment import PatchingExperiment
    
    # Get agent path from metric directory structure
    # Path structure: .../agent_path/circuit_verification/monotonic/metric_name
    agent_path = metric_dir.parent.parent.parent
    
    print(f"Setting up patching experiments for {len(coalition_neurons)} coalition steps...")
    
    # Initialize patching experiment
    experiment = PatchingExperiment(str(agent_path))
    
    # Create results structure like descending
    results_dir = metric_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Create experiment subdirectories
    denoising_dir = results_dir / "denoising"
    noising_dir = results_dir / "noising"
    denoising_dir.mkdir(exist_ok=True)
    noising_dir.mkdir(exist_ok=True)
    
    # Create patch configurations for each coalition size
    patch_configs = []
    coalition_names = []
    
    for i in range(len(coalition_neurons)):
        # Coalition from 1 neuron to i+1 neurons
        current_coalition = coalition_neurons[:i+1]
        
        # Convert to patch config format
        patch_config = {}
        for neuron_name in current_coalition:
            # Parse neuron name: e.g., "q_net.4_neuron_2" -> layer="q_net.4", idx=2
            parts = neuron_name.split("_")
            if len(parts) >= 3 and parts[-2] == "neuron":
                layer_name = "_".join(parts[:-2])
                neuron_idx = int(parts[-1])
                
                if layer_name not in patch_config:
                    patch_config[layer_name] = []
                patch_config[layer_name].append(neuron_idx)
        
        patch_configs.append(patch_config)
        
        # Create descriptive name for this coalition size
        if len(current_coalition) == 1:
            coalition_names.append(current_coalition[0])
        else:
            coalition_names.append(f"coalition_size_{len(current_coalition)}")
    
    print(f"Running bidirectional patching for {len(patch_configs)} coalition configurations...")
    
    # Run the bidirectional patching experiments
    try:
        denoising_files, noising_files = experiment.run_bidirectional_patching(
            clean_input_file="activation_inputs/clean_inputs.json",
            corrupted_input_file="activation_inputs/corrupted_inputs.json", 
            clean_activations_file="activation_logging/clean_activations.npz",
            corrupted_activations_file="activation_logging/corrupted_activations.npz",
            patch_configs=patch_configs,
            input_ids=input_examples
        )
        
        print(f"Experiments completed successfully!")
        print(f"Denoising results: {len(denoising_files)} files")
        print(f"Noising results: {len(noising_files)} files")
        
        return True
        
    except Exception as e:
        print(f"Error running patching experiments: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_logit_progression(metric_dir: Path, input_examples: List[str]) -> Dict[str, Dict[str, List[List[float]]]]:
    """
    Extract actual logit progression data by re-running the coalition experiments.
    """
    import sys
    import os
    from pathlib import Path
    import tempfile
    import shutil
    
    # Add required paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    neuron_selection_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    project_root = os.path.abspath(os.path.join(neuron_selection_dir, ".."))
    
    for path in [neuron_selection_dir, project_root]:
        if path not in sys.path:
            sys.path.append(path)
    
    from Neuron_Selection.PatchingTooling.patching_experiment import PatchingExperiment
    
    # Get agent path
    agent_path = metric_dir.parent.parent.parent
    
    # Load the coalition progression from summary
    summary_file = metric_dir / "summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    final_coalition = summary['results']['final_coalition_neurons']
    max_coalition_size = summary['parameters']['max_coalition_size']
    
    print(f"Extracting logit progression for {max_coalition_size} coalition sizes...")
    
    # Create temporary directories for each coalition size to avoid overwriting
    temp_base_dir = agent_path / "temp_coalition_experiments"
    temp_base_dir.mkdir(exist_ok=True)
    
    # Initialize patching experiment
    experiment = PatchingExperiment(str(agent_path))
    
    # Initialize logit data structure
    logit_data = {}
    for example in input_examples:
        logit_data[example] = {'noising': [], 'denoising': []}
    
    # Store results for each coalition size
    all_results = {'denoising': {}, 'noising': {}}
    
    # Run experiments for each coalition size (0=baseline, 1=first neuron, 2=first+second, etc.)
    # Use max_coalition_size instead of final_coalition length
    for coalition_size in range(max_coalition_size + 1):
        
        if coalition_size == 0:
            # Baseline (no patching) - get baseline from first neuron experiment
            if final_coalition:
                next_coalition = final_coalition[:1]
                patch_config = {}
                for neuron_name in next_coalition:
                    parts = neuron_name.split("_")
                    if len(parts) >= 3 and parts[-2] == "neuron":
                        layer_name = "_".join(parts[:-2])
                        neuron_idx = int(parts[-1])
                        
                        if layer_name not in patch_config:
                            patch_config[layer_name] = []
                        patch_config[layer_name].append(neuron_idx)
            current_coalition = []
        else:
            # Coalition of first N neurons (up to available)
            current_coalition = final_coalition[:min(coalition_size, len(final_coalition))]
            
            # Skip if we don't have enough neurons for this coalition size
            if len(current_coalition) < coalition_size:
                print(f"  Coalition size {coalition_size}: Skipping (only {len(final_coalition)} neurons available)")
                continue
            
            # Convert to patch config format
            patch_config = {}
            for neuron_name in current_coalition:
                # Parse neuron name: e.g., "q_net.4_neuron_2" -> layer="q_net.4", idx=2
                parts = neuron_name.split("_")
                if len(parts) >= 3 and parts[-2] == "neuron":
                    layer_name = "_".join(parts[:-2])
                    neuron_idx = int(parts[-1])
                    
                    if layer_name not in patch_config:
                        patch_config[layer_name] = []
                    patch_config[layer_name].append(neuron_idx)
        
        print(f"  Coalition size {coalition_size}: {len(current_coalition)} neurons")
        
        # Create temporary output directories for this coalition size
        temp_coalition_dir = temp_base_dir / f"coalition_{coalition_size}"
        temp_coalition_dir.mkdir(exist_ok=True)
        
        # Backup original patching results directory
        original_results_dir = agent_path / "patching_results"
        backup_results_dir = temp_coalition_dir / "patching_results_backup"
        if original_results_dir.exists():
            shutil.move(str(original_results_dir), str(backup_results_dir))
        
        try:
            # Run bidirectional patching for this coalition
            denoising_results, noising_results = experiment.run_bidirectional_patching(
                clean_input_file="activation_inputs/clean_inputs.json",
                corrupted_input_file="activation_inputs/corrupted_inputs.json",
                clean_activations_file="activation_logging/clean_activations.npz", 
                corrupted_activations_file="activation_logging/corrupted_activations.npz",
                patch_configs=[patch_config],
                input_ids=input_examples
            )
            
            # Copy results to permanent storage for this coalition size
            coalition_results_dir = temp_coalition_dir / "results"
            shutil.copytree(str(original_results_dir), str(coalition_results_dir))
            
            # Store the result file paths for this coalition size
            all_results['denoising'][coalition_size] = coalition_results_dir / "denoising"
            all_results['noising'][coalition_size] = coalition_results_dir / "noising"
            
        except Exception as e:
            print(f"    Error running experiments for coalition size {coalition_size}: {e}")
        finally:
            # Restore original results directory
            if original_results_dir.exists():
                shutil.rmtree(str(original_results_dir))
            if backup_results_dir.exists():
                shutil.move(str(backup_results_dir), str(original_results_dir))
    
    # Now extract logits from all stored results
    for example in input_examples:
        safe_example = example.replace("-", "_").replace(",", "_")
        
        for coalition_size in range(max_coalition_size + 1):
            for exp_type in ['denoising', 'noising']:
                if coalition_size in all_results[exp_type]:
                    result_dir = all_results[exp_type][coalition_size]
                    result_file = result_dir / f"{safe_example}.json"
                    
                    if result_file.exists():
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        
                        # Get logits from the experiment
                        exp_keys = list(data.keys())
                        if exp_keys and 'results' in data[exp_keys[0]]:
                            results = data[exp_keys[0]]['results']
                            
                            if coalition_size == 0:
                                # For baseline, use baseline_output
                                if 'baseline_output' in results:
                                    logits = results['baseline_output'][0]
                                    logit_data[example][exp_type].append(logits)
                            else:
                                # For patching, use patched_output
                                if 'patched_output' in results:
                                    logits = results['patched_output'][0]
                                    logit_data[example][exp_type].append(logits)
    
    # Clean up temporary directories
    if temp_base_dir.exists():
        shutil.rmtree(str(temp_base_dir))
    
    return logit_data


def create_individual_plot(metric_name: str, logit_data: Dict[str, Dict[str, List[List[float]]]], 
                          input_examples: List[str], output_file: Path):
    """
    Create an individual plot for one metric that matches the descending format exactly.
    """
    print(f"Creating individual plot for {metric_name}...")
    
    # Set up plot structure: N examples (rows) × 2 experiment types (columns)
    fig, axes = plt.subplots(len(input_examples), 2, figsize=(12, 3.5*len(input_examples)))
    
    # Handle single row case
    if len(input_examples) == 1:
        axes = axes.reshape(1, -1)
    
    # Set up colors for 5 logit indices
    n_logits = 5
    colors = plt.cm.magma(np.linspace(0.1, 0.9, n_logits))
    
    experiment_types = ["noising", "denoising"]
    
    # Calculate min/max for each row (environment) separately
    row_limits = []
    for i, example in enumerate(input_examples):
        row_min, row_max = float('inf'), float('-inf')
        for exp_type in experiment_types:
            if example in logit_data and exp_type in logit_data[example]:
                for logits in logit_data[example][exp_type]:
                    if logits:
                        row_min = min(row_min, min(logits))
                        row_max = max(row_max, max(logits))
        
        # Set reasonable defaults if no valid data
        if row_min == float('inf'):
            row_min, row_max = 0.0, 1.0
        
        margin = (row_max - row_min) * 0.1
        row_limits.append((row_min - margin, row_max + margin))
    
    # Create plots
    for i, example in enumerate(input_examples):
        # Format example name to match descending convention
        formatted_name = format_environment_name(example)
        
        for j, experiment_type in enumerate(experiment_types):
            ax = axes[i, j]
            
            # Get logit progression for this example and experiment type
            if example in logit_data and experiment_type in logit_data[example]:
                logits_progression = logit_data[example][experiment_type]
                
                # Plot each logit index as separate line
                for logit_idx in range(n_logits):
                    logit_values = []
                    coalition_sizes = []
                    
                    for coalition_size, logits in enumerate(logits_progression):
                        if logits and len(logits) > logit_idx:
                            logit_values.append(logits[logit_idx])
                            coalition_sizes.append(coalition_size)  # Start from 0 (baseline)
                    
                    if logit_values:
                        ax.plot(coalition_sizes, logit_values,
                               color=colors[logit_idx],
                               marker='o',
                               markersize=3,
                               linewidth=2,
                               label=f'Logit {logit_idx}')
            
            # Set axis properties - use row-specific limits
            max_coalition_size = len(logit_data[input_examples[0]]['noising']) - 1 if input_examples and input_examples[0] in logit_data else 30
            ax.set_xlim(-0.5, max_coalition_size + 0.5)
            ax.set_ylim(row_limits[i][0], row_limits[i][1])
            
            # Labels
            if i == len(input_examples) - 1:
                ax.set_xlabel('Coalition Size (0=Baseline)', fontsize=10)
            if j == 0:
                ax.set_ylabel('Logit Value', fontsize=10)
            
            # Title (only on top row)
            if i == 0:
                ax.set_title(f'{experiment_type.title()}', fontsize=12, fontweight='bold', pad=15)
            
            # Environment name (only on left column)
            if j == 0:
                ax.text(-0.2, 0.5, formatted_name, rotation=90, verticalalignment='center',
                       horizontalalignment='center', fontsize=10, fontweight='bold',
                       transform=ax.transAxes)
            
            ax.grid(True, alpha=0.3)
    
    # Create legend
    handles, labels = None, None
    for i in range(len(input_examples)):
        for j in range(2):
            h, l = axes[i, j].get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            break
    
    if handles:
        fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, 0.02),
                  title='Logit Index', title_fontsize=11, fontsize=9, ncol=5)
    
    # Overall title
    fig.suptitle(f'Monotonic Coalition Building - {metric_name}\n{len(input_examples)} Environments × 2 Experiment Types (0=Baseline)',
                fontsize=14, y=0.95)
    
    # Layout
    plt.subplots_adjust(left=0.22, right=0.95, top=0.88, bottom=0.08,
                       hspace=0.25, wspace=0.15)
    
    # Save plot
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")


def process_metric(agent_path: str, metric_name: str):
    """Process a single metric to create its individual plot with actual experiment data."""
    print(f"\n{'='*60}")
    print(f"PROCESSING METRIC: {metric_name}")
    print(f"{'='*60}")
    
    # Load summary and coalition progression
    metric_dir = Path(agent_path) / "circuit_verification" / "monotonic" / metric_name
    summary_file = metric_dir / "summary.json"
    
    if not summary_file.exists():
        print(f"No summary file found for {metric_name}")
        return
    
    coalition_neurons, input_examples = load_coalition_progression(summary_file)
    
    if not coalition_neurons:
        print(f"No coalition found for {metric_name}")
        return
    
    print(f"Coalition size: {len(coalition_neurons)}")
    print(f"Input examples: {len(input_examples)}")
    
    # Skip experiment creation - just use existing patching data directly
    print("Using existing patching data from agent directory...")
    
    # Extract actual logit progression from existing experiment results
    logit_data = extract_logit_progression(metric_dir, input_examples)
    
    # Create the individual plot
    safe_metric_name = metric_name.replace("/", "_").replace("\\", "_")
    plots_dir = metric_dir / "plots"
    output_file = plots_dir / f"coalition_verification_{safe_metric_name}.png"
    
    create_individual_plot(metric_name, logit_data, input_examples, output_file)


def main():
    parser = argparse.ArgumentParser(description="Create individual circuit-style plots for monotonic coalition results with actual experiment data")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--metrics", type=str, nargs="*",
                       help="Specific metrics to process (default: all found)")
    
    args = parser.parse_args()
    
    # Find available metrics
    monotonic_dir = Path(args.agent_path) / "circuit_verification" / "monotonic"
    
    if not monotonic_dir.exists():
        print(f"Monotonic directory not found: {monotonic_dir}")
        return
    
    available_metrics = []
    for metric_dir in monotonic_dir.iterdir():
        if metric_dir.is_dir() and (metric_dir / "summary.json").exists():
            available_metrics.append(metric_dir.name)
    
    if not available_metrics:
        print("No metrics with summary files found!")
        return
    
    # Filter to requested metrics
    if args.metrics:
        metrics_to_process = [m for m in args.metrics if m in available_metrics]
        if not metrics_to_process:
            print(f"None of the requested metrics found: {args.metrics}")
            print(f"Available metrics: {available_metrics}")
            return
    else:
        metrics_to_process = available_metrics
    
    print(f"Processing {len(metrics_to_process)} metrics: {metrics_to_process}")
    
    # Process each metric
    for metric_name in metrics_to_process:
        try:
            process_metric(args.agent_path, metric_name)
        except Exception as e:
            print(f"Error processing {metric_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("INDIVIDUAL PLOT GENERATION COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 