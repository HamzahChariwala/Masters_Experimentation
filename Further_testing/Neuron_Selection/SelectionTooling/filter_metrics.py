#!/usr/bin/env python3
"""
Script: filter_metrics.py

Filters patching experiment results based on specified thresholds for each metric
and generates separate files containing the filtered results.

Thresholds:
- KL divergence (and reverse KL): > 0.05
- Undirected saturating Chebyshev: > 0.5
- Confidence margin magnitude: > 0.05
- Reversed Pearson correlation: > 0.05

Each file contains:
1. "noising" section with experiments where normalized value > threshold
2. "denoising" section with experiments where normalized value > threshold
3. "common" section with experiments that appear in both noising and denoising

Additionally generates a cross-metric summary file containing neurons that appear
in all metrics' filtered results.

After filtering, also generates cumulative coalition experiments for circuit verification.
"""
import os
import sys
import json
import numpy as np
import argparse
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict

# Add the Neuron_Selection directory to the path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
neuron_selection_dir = os.path.dirname(script_dir)
if neuron_selection_dir not in sys.path:
    sys.path.insert(0, neuron_selection_dir)
    print(f"Added Neuron_Selection directory to Python path: {neuron_selection_dir}")

# Import the cumulative experiment generator
try:
    from CircuitTooling.cumulative_patching_generator import generate_cumulative_experiments
except ImportError as e:
    print(f"Warning: Could not import cumulative experiment generator: {e}")
    generate_cumulative_experiments = None

# Import the coalition to experiments converter
try:
    sys.path.insert(0, os.path.join(neuron_selection_dir, 'ExperimentTooling'))
    from ExperimentTooling.coalition_to_experiments import main as convert_coalitions_to_experiments
except ImportError as e:
    print(f"Warning: Could not import coalition to experiments converter: {e}")
    convert_coalitions_to_experiments = None

# Import the circuit experiments runner
try:
    sys.path.insert(0, os.path.join(neuron_selection_dir, 'CircuitTooling'))
    from CircuitTooling.run_circuit_experiments import run_all_circuit_experiments
    from CircuitTooling.visualize_circuit_results import create_circuit_visualization
except ImportError as e:
    print(f"Warning: Could not import circuit experiments runner or visualizer: {e}")
    run_all_circuit_experiments = None
    create_circuit_visualization = None

def load_summary_data(summary_file: str) -> Dict[str, Any]:
    """
    Load patching summary data from a JSON file.
    
    Args:
        summary_file: Path to the summary JSON file
        
    Returns:
        Dictionary containing the summary data
    """
    try:
        with open(summary_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading summary file {summary_file}: {e}")
        return {}

def filter_experiments_by_metric(
    summary_data: Dict[str, Any], 
    metric_name: str, 
    threshold: float
) -> Dict[str, Dict[str, Any]]:
    """
    Filter experiments where the normalized value of the specified metric exceeds the threshold.
    When threshold is 0.0, returns all experiments ranked by normalized importance.
    
    Args:
        summary_data: Dictionary containing the summary data
        metric_name: Name of the metric to filter by
        threshold: Threshold value for filtering (when 0.0, all experiments included)
        
    Returns:
        Dictionary of filtered experiments with their metrics
    """
    filtered_experiments = {}
    
    for exp_name, exp_data in summary_data.items():
        if 'metrics' in exp_data and metric_name in exp_data['metrics']:
            metric_data = exp_data['metrics'][metric_name]
            
            if threshold == 0.0:
                # Include all experiments when threshold is 0.0
                if 'normalized_value' in metric_data:
                    filtered_experiments[exp_name] = {
                        'metrics': {
                            metric_name: metric_data
                        },
                        'normalized_value': metric_data['normalized_value'],
                        'mean': metric_data['mean'],
                        'input_count': exp_data.get('input_count', 0)
                    }
            else:
                # Original behavior: filter by threshold
                if 'normalized_value' in metric_data and metric_data['normalized_value'] > threshold:
                    # Include the experiment in the filtered results
                    filtered_experiments[exp_name] = {
                        'metrics': {
                            metric_name: metric_data
                        },
                        'normalized_value': metric_data['normalized_value'],
                        'mean': metric_data['mean'],
                        'input_count': exp_data.get('input_count', 0)
                    }
    
    return filtered_experiments

def calculate_summary_statistics(
    filtered_experiments: Dict[str, Dict[str, Any]],
    metric_name: str,
    is_common: bool = False
) -> Dict[str, Any]:
    """
    Calculate summary statistics for the filtered experiments.
    
    Args:
        filtered_experiments: Dictionary of filtered experiments
        metric_name: Name of the metric
        is_common: Whether these are common experiments (different structure)
        
    Returns:
        Dictionary with summary statistics
    """
    if not filtered_experiments:
        return {
            'count': 0,
            'min_normalized': 0,
            'max_normalized': 0,
            'mean_normalized': 0,
            'min_raw': 0,
            'max_raw': 0,
            'mean_raw': 0
        }
    
    if is_common:
        # For common experiments, we average the noising and denoising values
        normalized_values = []
        raw_values = []
        for exp_data in filtered_experiments.values():
            noising_norm = exp_data['noising']['normalized_value']
            denoising_norm = exp_data['denoising']['normalized_value']
            noising_raw = exp_data['noising']['mean']
            denoising_raw = exp_data['denoising']['mean']
            
            # Average of noising and denoising
            normalized_values.append((noising_norm + denoising_norm) / 2)
            raw_values.append((noising_raw + denoising_raw) / 2)
    else:
        # For regular experiments
        normalized_values = [exp_data['normalized_value'] for exp_data in filtered_experiments.values()]
        raw_values = [exp_data['mean'] for exp_data in filtered_experiments.values()]
    
    return {
        'count': len(filtered_experiments),
        'min_normalized': float(np.min(normalized_values)) if normalized_values else 0,
        'max_normalized': float(np.max(normalized_values)) if normalized_values else 0,
        'mean_normalized': float(np.mean(normalized_values)) if normalized_values else 0,
        'min_raw': float(np.min(raw_values)) if raw_values else 0,
        'max_raw': float(np.max(raw_values)) if raw_values else 0,
        'mean_raw': float(np.mean(raw_values)) if raw_values else 0
    }

def write_filtered_results(
    output_file: str,
    metric_name: str,
    threshold: float,
    noising_filtered: Dict[str, Dict[str, Any]],
    denoising_filtered: Dict[str, Dict[str, Any]],
    common_experiments: Dict[str, Dict[str, Dict[str, Any]]]
) -> None:
    """
    Write filtered results to a JSON file.
    
    Args:
        output_file: Path to the output file
        metric_name: Name of the metric
        threshold: Threshold value used for filtering
        noising_filtered: Dictionary of filtered noising experiments
        denoising_filtered: Dictionary of filtered denoising experiments
        common_experiments: Dictionary of experiments that appear in both noising and denoising
    """
    # Calculate summary statistics
    noising_stats = calculate_summary_statistics(noising_filtered, metric_name)
    denoising_stats = calculate_summary_statistics(denoising_filtered, metric_name)
    common_stats = calculate_summary_statistics(common_experiments, metric_name, is_common=True)
    
    # Create 'averaged' section - rank by averaged normalized values across noising and denoising
    averaged_experiments = {}
    for exp_name, exp_data in common_experiments.items():
        noising_norm = exp_data['noising']['normalized_value']
        denoising_norm = exp_data['denoising']['normalized_value']
        averaged_score = (noising_norm + denoising_norm) / 2
        
        averaged_experiments[exp_name] = {
            'noising': exp_data['noising'],
            'denoising': exp_data['denoising'],
            'averaged_normalized_value': averaged_score
        }
    
    # Sort by averaged normalized value (highest to lowest)
    averaged_experiments_sorted = dict(sorted(
        averaged_experiments.items(), 
        key=lambda x: x[1]['averaged_normalized_value'], 
        reverse=True
    ))
    
    # Create 'highest' section - rank by highest normalized value across noising and denoising
    highest_experiments = {}
    for exp_name, exp_data in common_experiments.items():
        noising_norm = exp_data['noising']['normalized_value']
        denoising_norm = exp_data['denoising']['normalized_value']
        highest_score = max(noising_norm, denoising_norm)
        
        highest_experiments[exp_name] = {
            'noising': exp_data['noising'],
            'denoising': exp_data['denoising'],
            'highest_normalized_value': highest_score
        }
    
    # Sort by highest normalized value (highest to lowest)
    highest_experiments_sorted = dict(sorted(
        highest_experiments.items(), 
        key=lambda x: x[1]['highest_normalized_value'], 
        reverse=True
    ))
    
    # Prepare the output data with new structure
    output_data = {
        'metric': metric_name,
        'threshold': threshold,
        'averaged': averaged_experiments_sorted,  # Move to top and use new name
        'highest': highest_experiments_sorted,   # Add new highest section
        'summary': {
            'noising': noising_stats,
            'denoising': denoising_stats,
            'averaged': common_stats,  # Update key name in summary too
            'highest': common_stats   # Use same stats for highest (count is same)
        },
        'noising': noising_filtered,
        'denoising': denoising_filtered,
        # Note: 'common' section is replaced by 'averaged' and 'highest' above
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Filtered results saved to {output_file}")
    print(f"  Noising: {noising_stats['count']} experiments")
    print(f"  Denoising: {denoising_stats['count']} experiments")
    print(f"  Averaged: {len(averaged_experiments_sorted)} experiments")
    print(f"  Highest: {len(highest_experiments_sorted)} experiments")

def write_cross_metric_summary(
    output_file: str,
    cross_metric_common: Dict[str, Dict[str, Dict[str, Any]]],
    metrics_info: Dict[str, Dict[str, Any]]
) -> None:
    """
    Write summary of neurons that appear in all metrics' filtered results.
    
    Args:
        output_file: Path to the output file
        cross_metric_common: Dictionary of experiments common across all metrics
        metrics_info: Dictionary with information about each metric
    """
    # Calculate summary count
    common_count = len(cross_metric_common)
    
    # Prepare the output data
    output_data = {
        'description': 'Neurons that exceed threshold for all metrics',
        'metrics_info': metrics_info,
        'common_count': common_count,
        'common_neurons': cross_metric_common
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nCross-metric summary saved to {output_file}")
    print(f"  Common neurons across all metrics: {common_count}")

def filter_metrics(agent_path: str, max_experiments: int = 30) -> None:
    """
    Filter metrics from patching summary files and generate separate output files.
    
    Args:
        agent_path: Path to the agent directory
        max_experiments: Maximum number of cumulative experiments to generate (default: 30)
    """
    # Paths to summary files
    results_dir = os.path.join(agent_path, "patching_results")
    filtered_dir = os.path.join(results_dir, "filtered")
    
    # Create filtered directory if it doesn't exist
    os.makedirs(filtered_dir, exist_ok=True)
    
    noising_summary_file = os.path.join(results_dir, "patching_summary_noising.json")
    denoising_summary_file = os.path.join(results_dir, "patching_summary_denoising.json")
    
    # Load summary data
    print(f"Loading summary data from {noising_summary_file} and {denoising_summary_file}")
    noising_data = load_summary_data(noising_summary_file)
    denoising_data = load_summary_data(denoising_summary_file)
    
    # Check if we have data
    if not noising_data:
        print(f"No noising summary data found at {noising_summary_file}")
        return
    
    if not denoising_data:
        print(f"No denoising summary data found at {denoising_summary_file}")
        return
    
    # Define metrics and thresholds
    metrics_thresholds = {
        'kl_divergence': 0.0,
        'reverse_kl_divergence': 0.0,
        'undirected_saturating_chebyshev': 0.0,
        'confidence_margin_magnitude': 0.0,
        'reversed_pearson_correlation': 0.0,
        'reversed_undirected_saturating_chebyshev': 0.0,
        'top_logit_delta_magnitude': 0.0,
    }
    
    # Dictionary to store common neurons across metrics
    metrics_common_experiments = {}
    metrics_info = {}
    
    # Process each metric
    for metric_name, threshold in metrics_thresholds.items():
        print(f"\nProcessing metric: {metric_name} (threshold > {threshold})")
        
        # Filter experiments
        noising_filtered = filter_experiments_by_metric(noising_data, metric_name, threshold)
        
        # Use reverse_kl_divergence for denoising if the metric is kl_divergence
        denoising_metric = 'reverse_kl_divergence' if metric_name == 'kl_divergence' else metric_name
        denoising_filtered = filter_experiments_by_metric(denoising_data, denoising_metric, threshold)
        
        # Find common experiments between noising and denoising
        noising_exp_names = set(noising_filtered.keys())
        denoising_exp_names = set(denoising_filtered.keys())
        common_exp_names = noising_exp_names.intersection(denoising_exp_names)
        
        # Create common experiments dictionary
        common_experiments = {}
        for exp_name in common_exp_names:
            common_experiments[exp_name] = {
                'noising': noising_filtered[exp_name],
                'denoising': denoising_filtered[exp_name]
            }
        
        # Store common experiments for this metric
        metrics_common_experiments[metric_name] = common_experiments
        
        # Store metric info
        metrics_info[metric_name] = {
            'threshold': threshold,
            'common_count': len(common_experiments)
        }
        
        # Define output file in the filtered subdirectory
        output_file = os.path.join(filtered_dir, f"filtered_{metric_name}.json")
        
        # Write filtered results
        write_filtered_results(
            output_file,
            metric_name,
            threshold,
            noising_filtered,
            denoising_filtered,
            common_experiments
        )
    
    # Find neurons that are common across all metrics
    cross_metric_common = {}
    
    # Get common experiment names from the first metric
    if metrics_common_experiments:
        first_metric = list(metrics_common_experiments.keys())[0]
        common_exp_set = set(metrics_common_experiments[first_metric].keys())
        
        # Intersect with common experiment names from other metrics
        for metric_name in list(metrics_common_experiments.keys())[1:]:
            metric_exp_set = set(metrics_common_experiments[metric_name].keys())
            common_exp_set = common_exp_set.intersection(metric_exp_set)
        
        # Create cross-metric common experiments dictionary
        for exp_name in common_exp_set:
            cross_metric_common[exp_name] = {
                metric_name: metrics_common_experiments[metric_name][exp_name]
                for metric_name in metrics_common_experiments.keys()
                if exp_name in metrics_common_experiments[metric_name]
            }
    
    # Write cross-metric summary to the main results directory
    cross_metric_file = os.path.join(results_dir, "cross_metric_summary.json")
    write_cross_metric_summary(cross_metric_file, cross_metric_common, metrics_info)

    # Generate cumulative coalition experiments
    if generate_cumulative_experiments:
        print(f"\nGenerating cumulative coalition experiments...")
        try:
            generate_cumulative_experiments(agent_path, max_experiments=max_experiments)
            
            # Convert coalition files to experiment format
            if convert_coalitions_to_experiments:
                print(f"Converting coalition files to experiment format...")
                # Temporarily modify sys.argv to pass arguments to the converter
                original_argv = sys.argv
                sys.argv = ["coalition_to_experiments.py", "--agent_path", agent_path]
                try:
                    convert_coalitions_to_experiments()
                except SystemExit:
                    pass  # Ignore SystemExit from the converter
                finally:
                    sys.argv = original_argv
            else:
                print(f"Skipping coalition to experiments conversion (module not available)")
                
            # Run circuit experiments
            if run_all_circuit_experiments:
                print(f"Running circuit experiments...")
                run_all_circuit_experiments(agent_path, subfolder="descending/results")
                
                # Generate visualizations
                if create_circuit_visualization:
                    print(f"Generating circuit verification visualizations...")
                    create_circuit_visualization(agent_path, subfolder="descending")
                else:
                    print(f"Skipping visualization generation (module not available)")
            else:
                print(f"Skipping circuit experiments (module not available)")
                
        except Exception as e:
            print(f"Error in circuit verification workflow: {e}")
    else:
        print(f"\nSkipping cumulative experiment generation (module not available)")

def main():
    parser = argparse.ArgumentParser(description="Filter metrics based on thresholds")
    parser.add_argument("--agent_path", type=str, required=True,
                      help="Path to the agent directory")
    parser.add_argument("--max_experiments", type=int, default=30,
                      help="Maximum number of cumulative experiments to generate (default: 30)")
    args = parser.parse_args()
    
    # Filter metrics
    filter_metrics(args.agent_path, args.max_experiments)

if __name__ == "__main__":
    main() 