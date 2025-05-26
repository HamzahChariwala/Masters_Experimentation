#!/usr/bin/env python3
"""
Script: patching_summary.py

Analyzes patching experiment results across multiple inputs, calculating
summary statistics (mean, standard deviation) for each metric and normalizing
values across all experiments.
"""
import os
import sys
import json
import glob
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union
import re

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from Neuron_Selection.AnalysisTooling import METRIC_FUNCTIONS


def load_experiment_results(result_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all experiment result files from a directory.
    
    Args:
        result_dir: Directory containing result JSON files
        
    Returns:
        Dictionary mapping input IDs to experiment results
    """
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(result_dir, "*.json"))
    
    # Initialize results dictionary
    results = {}
    
    # Load each file
    for file_path in json_files:
        # Extract input ID from filename
        input_id = os.path.basename(file_path).split('.json')[0]
        
        try:
            with open(file_path, 'r') as f:
                results[input_id] = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results


def process_experiment_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Process experiment results to group by experiment and calculate statistics.
    
    Args:
        results: Dictionary mapping input IDs to experiment results
        
    Returns:
        Dictionary with summary statistics for each experiment
    """
    # Dictionary to store metrics for each experiment across inputs
    # Structure: {experiment_name: {metric_name: [values_across_inputs]}}
    experiment_metrics = defaultdict(lambda: defaultdict(list))
    
    # Collect metrics for each experiment across all inputs
    for input_id, input_results in results.items():
        for exp_name, exp_data in input_results.items():
            # Skip if no patch_analysis
            if "patch_analysis" not in exp_data:
                continue
            
            # Use the original experiment name as the key (preserves "exp_N_...")
            experiment_key = exp_name
            
            # Collect metrics for this experiment
            for metric_name, metric_value in exp_data["patch_analysis"].items():
                # Skip if the metric value is an error string
                if isinstance(metric_value, str) and metric_value == "error":
                    continue
                    
                # For metrics that return dictionaries, extract the main value
                if isinstance(metric_value, dict):
                    if "normalized_margin_change" in metric_value:
                        # For confidence_margin_change, use normalized_margin_change
                        experiment_metrics[experiment_key][metric_name].append(metric_value["normalized_margin_change"])
                    elif "margin_change" in metric_value:
                        # For updated confidence_margin_change, use margin_change
                        experiment_metrics[experiment_key][metric_name].append(metric_value["margin_change"])
                    elif "correlation" in metric_value:
                        # For pearson_correlation, use correlation
                        experiment_metrics[experiment_key][metric_name].append(metric_value["correlation"])
                    elif "distance" in metric_value:
                        # For chebyshev_distance_excluding_top, use distance
                        experiment_metrics[experiment_key][metric_name].append(metric_value["distance"])
                    elif "gap" in metric_value:
                        # For top_action_probability_gap, use gap
                        experiment_metrics[experiment_key][metric_name].append(metric_value["gap"])
                    else:
                        # For other dict metrics, use the first numeric value
                        for key, val in metric_value.items():
                            if isinstance(val, (int, float)):
                                experiment_metrics[experiment_key][metric_name].append(val)
                                break
                else:
                    # For scalar metrics, just use the value
                    experiment_metrics[experiment_key][metric_name].append(metric_value)
    
    # Calculate statistics for each experiment and metric
    summary_stats = {}
    for experiment_key, metrics in experiment_metrics.items():
        summary_stats[experiment_key] = {
            "metrics": {},
            "input_count": 0
        }
        
        # Find the maximum input count across all metrics for this experiment
        max_input_count = max([len(values) for values in metrics.values()], default=0)
        summary_stats[experiment_key]["input_count"] = max_input_count
        
        # Calculate statistics for each metric
        for metric_name, values in metrics.items():
            if not values:
                continue
                
            # Convert to numpy array, filtering out any non-numeric values
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if not numeric_values:
                continue
                
            values_array = np.array(numeric_values)
            
            summary_stats[experiment_key]["metrics"][metric_name] = {
                "mean": float(np.mean(values_array)),
                "std_dev": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "count": len(numeric_values)
            }
    
    # Calculate standardized and normalized values across all experiments
    calculate_metrics_statistics(summary_stats)
    
    return summary_stats


def calculate_metrics_statistics(summary_stats: Dict[str, Dict[str, Any]]) -> None:
    """
    Calculate standardized and normalized values across all experiments.
    
    Args:
        summary_stats: Dictionary with summary statistics to be processed
    """
    # First, collect all mean values for each metric across experiments
    metric_means = defaultdict(list)
    metric_mins = defaultdict(list)
    metric_maxs = defaultdict(list)
    
    for experiment_key, exp_data in summary_stats.items():
        for metric_name, metric_stats in exp_data["metrics"].items():
            metric_means[metric_name].append(metric_stats["mean"])
            metric_mins[metric_name].append(metric_stats["min"])
            metric_maxs[metric_name].append(metric_stats["max"])
    
    # Calculate global statistics for each metric
    metric_globals = {}
    for metric_name, means in metric_means.items():
        if not means:
            continue
            
        means_array = np.array(means)
        min_value = min(metric_mins[metric_name])
        max_value = max(metric_maxs[metric_name])
        
        metric_globals[metric_name] = {
            "global_mean": float(np.mean(means_array)),
            "global_std_dev": float(np.std(means_array)),
            "global_min": float(min_value),
            "global_max": float(max_value)
        }
    
    # Add standardized and normalized values to each metric
    for experiment_key, exp_data in summary_stats.items():
        for metric_name, metric_stats in exp_data["metrics"].items():
            if metric_name in metric_globals:
                global_mean = metric_globals[metric_name]["global_mean"]
                global_std_dev = metric_globals[metric_name]["global_std_dev"]
                global_min = metric_globals[metric_name]["global_min"]
                global_max = metric_globals[metric_name]["global_max"]
                
                # Avoid division by zero for standardization
                if global_std_dev != 0:
                    # Z-score standardization
                    metric_stats["standardized_value"] = (metric_stats["mean"] - global_mean) / global_std_dev
                else:
                    # If std dev is 0, use difference from mean
                    metric_stats["standardized_value"] = metric_stats["mean"] - global_mean
                
                # Avoid division by zero for normalization
                if global_max != global_min:
                    # Min-max normalization to [0, 1] range
                    metric_stats["normalized_value"] = (metric_stats["mean"] - global_min) / (global_max - global_min)
                else:
                    # If all values are the same, normalized value is 0.5
                    metric_stats["normalized_value"] = 0.5
                
                # Add global stats for reference
                metric_stats["global_mean"] = global_mean
                metric_stats["global_std_dev"] = global_std_dev
                metric_stats["global_min"] = global_min
                metric_stats["global_max"] = global_max


def save_summary_results(summary_stats: Dict[str, Dict[str, Any]], output_file: str) -> None:
    """
    Save summary statistics to a JSON file.
    
    Args:
        summary_stats: Dictionary with summary statistics
        output_file: Path to output file
    """
    # Ensure directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Summary statistics saved to {output_file}")


def process_patching_results(results_dir: str, output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Process patching results to calculate statistics across inputs.
    
    Args:
        results_dir: Directory containing patching results
        output_dir: Directory to save summary files (defaults to results_dir)
        
    Returns:
        Dictionary mapping result type to output file path
    """
    # Set default output directory
    if output_dir is None:
        output_dir = results_dir
    
    output_files = {}
    
    # Check for noising and denoising subdirectories
    noising_dir = os.path.join(results_dir, "noising")
    denoising_dir = os.path.join(results_dir, "denoising")
    
    # Process noising results if they exist
    if os.path.exists(noising_dir) and os.path.isdir(noising_dir):
        print(f"Processing noising results from {noising_dir}")
        noising_results = load_experiment_results(noising_dir)
        
        if noising_results:
            print(f"Found {len(noising_results)} input files in noising directory")
            noising_summary = process_experiment_results(noising_results)
            
            # Save noising summary
            noising_output_file = os.path.join(output_dir, "patching_summary_noising.json")
            save_summary_results(noising_summary, noising_output_file)
            output_files["noising"] = noising_output_file
        else:
            print("No results found in noising directory")
    
    # Process denoising results if they exist
    if os.path.exists(denoising_dir) and os.path.isdir(denoising_dir):
        print(f"Processing denoising results from {denoising_dir}")
        denoising_results = load_experiment_results(denoising_dir)
        
        if denoising_results:
            print(f"Found {len(denoising_results)} input files in denoising directory")
            denoising_summary = process_experiment_results(denoising_results)
            
            # Save denoising summary
            denoising_output_file = os.path.join(output_dir, "patching_summary_denoising.json")
            save_summary_results(denoising_summary, denoising_output_file)
            output_files["denoising"] = denoising_output_file
        else:
            print("No results found in denoising directory")
    
    # If there are no noising/denoising subdirectories, process the main directory
    if not output_files:
        print(f"Processing results directly from {results_dir}")
        main_results = load_experiment_results(results_dir)
        
        if main_results:
            print(f"Found {len(main_results)} input files")
            main_summary = process_experiment_results(main_results)
            
            # Save main summary
            main_output_file = os.path.join(output_dir, "patching_summary.json")
            save_summary_results(main_summary, main_output_file)
            output_files["main"] = main_output_file
        else:
            print("No results found in the specified directory")
    
    return output_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate summary statistics for patching experiments")
    parser.add_argument("--agent_path", type=str, required=True,
                        help="Path to the agent directory (e.g., Agent_Storage/SpawnTests/biased/biased-v1)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory containing patching results (defaults to <agent_path>/patching_results)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save summary files (defaults to results_dir)")
    
    args = parser.parse_args()
    
    # Set up the results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(args.agent_path, "patching_results")
    
    # Set up the output directory
    output_dir = args.output_dir or results_dir
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Process the results
    output_files = process_patching_results(results_dir, output_dir)
    
    if output_files:
        print("\nSummary files generated:")
        for result_type, file_path in output_files.items():
            print(f"  {result_type}: {file_path}")
        print("\nProcessing complete.")
    else:
        print("\nNo summary files were generated. Check for errors or missing data.") 