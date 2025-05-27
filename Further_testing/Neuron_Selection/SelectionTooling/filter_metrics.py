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
    
    Args:
        summary_data: Dictionary containing the summary data
        metric_name: Name of the metric to filter by
        threshold: Threshold value for filtering
        
    Returns:
        Dictionary of filtered experiments with their metrics
    """
    filtered_experiments = {}
    
    for exp_name, exp_data in summary_data.items():
        if 'metrics' in exp_data and metric_name in exp_data['metrics']:
            metric_data = exp_data['metrics'][metric_name]
            
            # Check if normalized value exists and exceeds threshold
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
    
    # Prepare the output data
    output_data = {
        'metric': metric_name,
        'threshold': threshold,
        'summary': {
            'noising': noising_stats,
            'denoising': denoising_stats,
            'common': common_stats
        },
        'noising': noising_filtered,
        'denoising': denoising_filtered,
        'common': common_experiments
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Filtered results saved to {output_file}")
    print(f"  Noising: {noising_stats['count']} experiments")
    print(f"  Denoising: {denoising_stats['count']} experiments")
    print(f"  Common: {common_stats['count']} experiments")

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

def filter_metrics(agent_path: str) -> None:
    """
    Filter metrics from patching summary files and generate separate output files.
    
    Args:
        agent_path: Path to the agent directory
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
        'kl_divergence': 0.05,
        'reverse_kl_divergence': 0.05,
        'undirected_saturating_chebyshev': 0.5,
        'confidence_margin_magnitude': 0.05,
        'reversed_pearson_correlation': 0.05
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

def main():
    parser = argparse.ArgumentParser(description="Filter metrics based on thresholds")
    parser.add_argument("--agent_path", type=str, required=True,
                      help="Path to the agent directory")
    args = parser.parse_args()
    
    # Filter metrics
    filter_metrics(args.agent_path)

if __name__ == "__main__":
    main() 