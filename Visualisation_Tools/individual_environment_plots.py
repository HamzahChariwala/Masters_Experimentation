#!/usr/bin/env python3
"""
Individual Environment Plot Generator

Generates separate plots for each environment showing logit progression.
Works with both descending and monotonic circuit verification results.
Only processes noising experiments.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

# Add the project directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


def format_environment_name(raw_name: str) -> str:
    """Clean up environment names for display."""
    # Remove "MiniGrid_LavaCrossing" prefix if present
    if raw_name.startswith("MiniGrid_LavaCrossing"):
        name = raw_name[len("MiniGrid_LavaCrossing"):]
    else:
        name = raw_name
    
    # Remove the last "_" and four digits if the pattern matches
    import re
    name = re.sub(r'_\d{4}$', '', name)
    
    return name


def load_monotonic_data(agent_path: Path, metric: str, input_examples: List[str]) -> Dict[str, List[List[float]]]:
    """Load REAL logit data from monotonic coalition results by re-running coalition experiments."""
    logit_data = {}
    
    # Load the coalition progression from summary
    summary_file = agent_path / "circuit_verification" / "monotonic" / metric / "summary.json"
    
    if not summary_file.exists():
        print(f"Warning: No summary file found for metric {metric}")
        return logit_data
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    final_coalition = summary['results']['final_coalition_neurons']
    coalition_scores = summary['results']['coalition_scores']
    
    if not final_coalition or not coalition_scores:
        print(f"Warning: No coalition data found for metric {metric}")
        return logit_data
    
    print(f"Loading REAL monotonic data for {metric}: {len(coalition_scores)} coalition sizes")
    print(f"Re-running coalition experiments with real neurons...")
    
    # Import the patching experiment class
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from Neuron_Selection.PatchingTooling.patching_experiment import PatchingExperiment
    
    # Initialize patching experiment
    experiment = PatchingExperiment(str(agent_path))
    
    # Initialize logit data structure for each environment
    for example in input_examples:
        logit_data[example] = []
    
    # For each coalition size, run the actual experiment and extract logits
    for coalition_size in range(len(coalition_scores)):
        if coalition_size == 0:
            # Baseline (no patching)
            print(f"  Coalition size {coalition_size}: Baseline (no patching)")
            current_coalition = []
        else:
            # Coalition of first N neurons
            current_coalition = final_coalition[:coalition_size]
            print(f"  Coalition size {coalition_size}: {len(current_coalition)} neurons")
        
        # Create patch configuration
        patch_config = {}
        for neuron_name in current_coalition:
            # Parse neuron name: e.g., "q_net.0_neuron_7" -> layer="q_net.0", idx=7
            parts = neuron_name.split("_")
            if len(parts) >= 3 and parts[-2] == "neuron":
                layer_name = "_".join(parts[:-2])
                neuron_idx = int(parts[-1])
                
                if layer_name not in patch_config:
                    patch_config[layer_name] = []
                patch_config[layer_name].append(neuron_idx)
        
        try:
            if coalition_size == 0:
                # For baseline, run experiment with empty patch to get baseline logits
                noising_results = experiment.run_patching_experiment(
                    target_input_file="clean_inputs.json",
                    source_activation_file="corrupted_activations.npz",
                    patch_spec={},  # Empty patch for baseline
                    input_ids=input_examples
                )
            else:
                # Run noising experiment (corrupted activations → clean inputs)
                noising_results = experiment.run_patching_experiment(
                    target_input_file="clean_inputs.json",
                    source_activation_file="corrupted_activations.npz",
                    patch_spec=patch_config,
                    input_ids=input_examples
                )
            
            # Extract logits for each environment
            for example in input_examples:
                if example in noising_results:
                    result = noising_results[example]
                    
                    if coalition_size == 0:
                        # For baseline, use baseline_output
                        if 'baseline_output' in result and result['baseline_output']:
                            logits = result['baseline_output'][0]  # First element contains logits
                            logit_data[example].append(logits)
                        else:
                            print(f"    Warning: No baseline output for {example}")
                            logit_data[example].append([0.2] * 5)  # Fallback
                    else:
                        # For patching, use patched_output
                        if 'patched_output' in result and result['patched_output']:
                            logits = result['patched_output'][0]  # First element contains logits
                            logit_data[example].append(logits)
                        else:
                            print(f"    Warning: No patched output for {example} at coalition size {coalition_size}")
                            # Use previous logits if available, otherwise fallback
                            if logit_data[example]:
                                logit_data[example].append(logit_data[example][-1])
                            else:
                                logit_data[example].append([0.2] * 5)
                else:
                    print(f"    Warning: No results for {example} at coalition size {coalition_size}")
                    # Use previous logits if available, otherwise fallback
                    if logit_data[example]:
                        logit_data[example].append(logit_data[example][-1])
                    else:
                        logit_data[example].append([0.2] * 5)
        
        except Exception as e:
            print(f"    Error running experiment for coalition size {coalition_size}: {e}")
            # For error cases, use previous logits or fallback
            for example in input_examples:
                if logit_data[example]:
                    logit_data[example].append(logit_data[example][-1])
                else:
                    logit_data[example].append([0.2] * 5)
    
    # Verify data structure
    for example in input_examples:
        if example in logit_data:
            print(f"  Loaded {len(logit_data[example])} logit vectors for {example}")
    
    return logit_data


def load_descending_data(agent_path: Path, input_examples: List[str]) -> Dict[str, List[List[float]]]:
    """Load logit data from descending circuit results."""
    logit_data = {}
    
    # Find the descending noising results directory
    noising_dir = agent_path / "circuit_verification" / "descending" / "results" / "noising"
    
    if not noising_dir.exists():
        print(f"Warning: No descending noising results found at {noising_dir}")
        return logit_data
    
    for example in input_examples:
        safe_example = example.replace("-", "_").replace(",", "_")
        result_file = noising_dir / f"{safe_example}.json"
        
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract logits from each experiment
            logits_sequence = []
            
            # Sort experiments by key to ensure consistent order
            exp_keys = sorted(data.keys())
            
            for exp_key in exp_keys:
                if 'results' in data[exp_key]:
                    results = data[exp_key]['results']
                    
                    # Get baseline first, then patched
                    if 'baseline_output' in results and results['baseline_output']:
                        logits_sequence.append(results['baseline_output'][0])
                    
                    if 'patched_output' in results and results['patched_output']:
                        logits_sequence.append(results['patched_output'][0])
            
            logit_data[example] = logits_sequence
    
    return logit_data


def get_input_examples(agent_path: Path) -> List[str]:
    """Get the 5 input examples from the agent's clean_inputs.json file."""
    clean_inputs_file = agent_path / "activation_inputs" / "clean_inputs.json"
    
    if clean_inputs_file.exists():
        with open(clean_inputs_file, 'r') as f:
            clean_inputs_data = json.load(f)
        
        # Take first 5 input IDs for consistency
        return list(clean_inputs_data.keys())[:5]
    else:
        raise FileNotFoundError(f"Could not find input examples at {clean_inputs_file}")


def create_individual_environment_plot(environment_name: str, logits_sequence: List[List[float]], 
                                     output_file: Path, analysis_type: str):
    """Create a plot for a single environment showing logit progression."""
    
    if not logits_sequence:
        print(f"No data for environment {environment_name}")
        return
    
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Set up colors for 5 logit indices
    n_logits = 5
    colors = plt.cm.magma(np.linspace(0.1, 0.9, n_logits))
    
    # Plot each logit index as separate line
    for logit_idx in range(n_logits):
        logit_values = []
        steps = []
        
        for step, logits in enumerate(logits_sequence):
            if logits and len(logits) > logit_idx:
                logit_values.append(logits[logit_idx])
                steps.append(step)
        
        if logit_values:
            ax.plot(steps, logit_values,
                   color=colors[logit_idx],
                   marker='o',
                   markersize=4,
                   linewidth=2.5,
                   label=f'Logit {logit_idx}')
    
    # Calculate y-axis limits with margin
    all_values = [val for logits in logits_sequence for val in logits if logits]
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
    
    # Set axis properties
    ax.set_xlim(-0.5, len(logits_sequence) - 0.5)
    ax.set_xlabel('Coalition Size', fontsize=12)
    ax.set_ylabel('Logit Value', fontsize=12)
    
    # Format environment name for title
    formatted_name = format_environment_name(environment_name)
    ax.set_title(f'{analysis_type} - {formatted_name}\nLogit Progression (Noising Only)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(title='Logit Index', loc='best', framealpha=0.9)
    
    # Layout and save
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")


def process_monotonic_environments(agent_path: Path, metric: str):
    """Process and create plots for monotonic coalition environments."""
    print(f"\nProcessing Monotonic Coalition - {metric}")
    
    # Get input examples
    input_examples = get_input_examples(agent_path)
    
    # Load logit data showing coalition progression
    logit_data = load_monotonic_data(agent_path, metric, input_examples)
    
    if not logit_data:
        print(f"No logit data found for monotonic analysis of {metric}")
        return
    
    # Create output directory
    output_dir = agent_path / "circuit_verification" / "monotonic" / metric / "individual_environment_plots"
    
    # Create individual plots for each environment
    for environment in input_examples:
        if environment in logit_data:
            safe_env_name = environment.replace("-", "_").replace(",", "_")
            output_file = output_dir / f"environment_{safe_env_name}_coalition_progression.png"
            
            create_individual_environment_plot(
                environment, 
                logit_data[environment],
                output_file,
                f"Monotonic Coalition - {metric}"
            )


def process_descending_environments(agent_path: Path):
    """Process and create plots for descending circuit environments."""
    print(f"\nProcessing Descending Circuit Analysis")
    
    # Get input examples
    input_examples = get_input_examples(agent_path)
    
    # Load logit data
    logit_data = load_descending_data(agent_path, input_examples)
    
    if not logit_data:
        print(f"No logit data found for descending analysis")
        return
    
    # Create output directory
    output_dir = agent_path / "circuit_verification" / "descending" / "individual_environment_plots"
    
    # Create individual plots for each environment
    for environment in input_examples:
        if environment in logit_data:
            safe_env_name = environment.replace("-", "_").replace(",", "_")
            output_file = output_dir / f"environment_{safe_env_name}_noising.png"
            
            create_individual_environment_plot(
                environment, 
                logit_data[environment],
                output_file,
                "Descending Circuit Analysis"
            )


def main():
    parser = argparse.ArgumentParser(description="Generate individual environment plots for circuit analysis")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--analysis_type", type=str, choices=['monotonic', 'descending', 'both'], 
                       default='both', help="Type of analysis to process")
    parser.add_argument("--metric", type=str, help="Specific metric for monotonic analysis (required if monotonic)")
    
    args = parser.parse_args()
    
    agent_path = Path(args.agent_path)
    
    if not agent_path.exists():
        print(f"Agent path does not exist: {agent_path}")
        return
    
    print(f"Creating individual environment plots for: {agent_path}")
    
    if args.analysis_type in ['descending', 'both']:
        process_descending_environments(agent_path)
    
    if args.analysis_type in ['monotonic', 'both']:
        if args.metric:
            process_monotonic_environments(agent_path, args.metric)
        else:
            # Process all available monotonic metrics
            monotonic_dir = agent_path / "circuit_verification" / "monotonic"
            if monotonic_dir.exists():
                for metric_dir in monotonic_dir.iterdir():
                    if metric_dir.is_dir() and (metric_dir / "summary.json").exists():
                        process_monotonic_environments(agent_path, metric_dir.name)
    
    print("\nIndividual environment plot generation complete!")


if __name__ == "__main__":
    main() 