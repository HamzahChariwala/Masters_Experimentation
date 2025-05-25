#!/usr/bin/env python3
"""
Tools for processing activation patching results.

This module provides utilities for loading, analyzing, and updating
the JSON files produced by the activation patching experiments.
"""
import os
import json
import glob
from typing import Dict, List, Any, Union, Optional, Tuple
from .metrics import METRIC_FUNCTIONS


def load_result_file(file_path: str) -> Dict[str, Any]:
    """
    Load a patching result file.
    
    Args:
        file_path: Path to the JSON result file
        
    Returns:
        Dictionary containing the result data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Result file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def save_result_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Save updated result data to a JSON file.
    
    Args:
        data: Updated result data
        file_path: Path to save the JSON file
    """
    # Clean up any non-serializable objects
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    # Convert the data to a serializable format
    serializable_data = make_serializable(data)
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Updated results saved to {file_path}")
    
    # Verify that all metrics were saved
    for exp_name, exp_data in data.items():
        if "patch_analysis" in exp_data:
            metric_count = len(exp_data["patch_analysis"])
            print(f"  Saved {metric_count} metrics for experiment '{exp_name}'")
            
            # Print the metrics if debugging is needed
            if metric_count > 0:
                metrics_str = ", ".join(f"{k}" for k in exp_data["patch_analysis"].keys())
                print(f"  Metrics: {metrics_str}")
                
                # Check for any error values
                error_metrics = [k for k, v in exp_data["patch_analysis"].items() if v == "error"]
                if error_metrics:
                    print(f"  Warning: Metrics with errors: {', '.join(error_metrics)}")
    
    # Reload the file to verify
    try:
        with open(file_path, 'r') as f:
            reloaded = json.load(f)
        
        total_metrics = sum(len(exp.get("patch_analysis", {})) for exp in reloaded.values())
        print(f"  Verification: {total_metrics} total metrics saved successfully")
    except Exception as e:
        print(f"  Warning: Could not verify saved file: {e}")


def analyze_experiment_results(experiment_data: Dict[str, Any], metric_names: List[str] = None) -> Dict[str, Any]:
    """
    Analyze the results of a single experiment by calculating specified metrics.
    
    Args:
        experiment_data: Data for a single experiment from a result file
        metric_names: List of metric names to calculate (if None, calculate all)
        
    Returns:
        Dictionary of calculated metrics
    """
    # Default to all metrics if none specified
    if metric_names is None:
        metric_names = list(METRIC_FUNCTIONS.keys())
    
    # Get the experiment results
    # The actual results are in the 'results' field of the experiment data
    results = experiment_data.get("results", {})
    
    # Extract data needed for metrics
    baseline_output = results.get("baseline_output", [])
    patched_output = results.get("patched_output", [])
    baseline_action = results.get("baseline_action", 0)
    
    # Debug information
    print(f"  Baseline action: {baseline_action}")
    print(f"  Output shapes: baseline={len(baseline_output)}x{len(baseline_output[0]) if baseline_output else 0}, "
          f"patched={len(patched_output)}x{len(patched_output[0]) if patched_output else 0}")
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate each requested metric
    for metric_name in metric_names:
        if metric_name not in METRIC_FUNCTIONS:
            print(f"Warning: Unknown metric '{metric_name}'. Skipping.")
            continue
        
        metric_func = METRIC_FUNCTIONS[metric_name]
        
        try:
            # Check function signature to determine if action is needed
            if metric_name in ["output_logit_delta", "action_probability_delta"]:
                metrics[metric_name] = metric_func(baseline_output, patched_output, baseline_action)
            elif metric_name == "top_action_probability_gap":
                gap, base_top, patched_top = metric_func(baseline_output, patched_output)
                metrics[metric_name] = {
                    "gap": gap,
                    "baseline_top_action": base_top,
                    "patched_top_action": patched_top
                }
            else:
                metrics[metric_name] = metric_func(baseline_output, patched_output)
                
            print(f"  Calculated {metric_name}: {metrics[metric_name]}")
            
        except Exception as e:
            print(f"Error calculating metric {metric_name}: {e}")
            metrics[metric_name] = "error"
    
    return metrics


def process_result_file(file_path: str, metric_names: List[str] = None) -> Dict[str, Any]:
    """
    Process a result file by adding analysis metrics to each experiment.
    
    Args:
        file_path: Path to the JSON result file
        metric_names: List of metric names to calculate (if None, calculate all)
        
    Returns:
        Updated result data
    """
    # Load the result file
    data = load_result_file(file_path)
    
    # Process each experiment in the file
    for exp_name, exp_data in data.items():
        # Check if we need to recalculate any metrics
        metrics_to_calculate = []
        if metric_names is not None:
            # Determine which metrics need to be calculated
            if "patch_analysis" not in exp_data:
                metrics_to_calculate = metric_names
            else:
                # Only calculate metrics that aren't already present
                metrics_to_calculate = [m for m in metric_names if m not in exp_data["patch_analysis"]]
        else:
            # Calculate all metrics if none specified
            metrics_to_calculate = list(METRIC_FUNCTIONS.keys())
            
            # Skip metrics that already exist
            if "patch_analysis" in exp_data:
                metrics_to_calculate = [m for m in metrics_to_calculate if m not in exp_data["patch_analysis"]]
        
        # Skip if no metrics need to be calculated
        if not metrics_to_calculate:
            print(f"Experiment '{exp_name}' already has all requested metrics. Skipping.")
            continue
            
        print(f"Calculating metrics for experiment '{exp_name}': {', '.join(metrics_to_calculate)}")
        
        # Analyze the experiment results with the filtered metrics list
        metrics = analyze_experiment_results(exp_data, metrics_to_calculate)
        
        # Add the metrics to the experiment data
        if "patch_analysis" not in exp_data:
            exp_data["patch_analysis"] = {}
        
        # Update the patch analysis with new metrics
        for metric_name, metric_value in metrics.items():
            exp_data["patch_analysis"][metric_name] = metric_value
            print(f"Added metric {metric_name} to experiment {exp_name}")
    
    # Save the updated data
    save_result_file(data, file_path)
    
    return data


def process_result_directory(directory_path: str, metric_names: List[str] = None) -> Dict[str, str]:
    """
    Process all result files in a directory by adding analysis metrics.
    
    Args:
        directory_path: Path to the directory containing JSON result files
        metric_names: List of metric names to calculate (if None, calculate all)
        
    Returns:
        Dictionary mapping file paths to their updated status
    """
    # Find all JSON files in the directory
    file_paths = glob.glob(os.path.join(directory_path, "*.json"))
    
    if not file_paths:
        print(f"No JSON files found in {directory_path}")
        return {}
    
    results = {}
    
    # Process each file
    for file_path in file_paths:
        try:
            print(f"Processing {file_path}...")
            data = process_result_file(file_path, metric_names)
            
            # Save the updated data
            save_result_file(data, file_path)
            
            results[file_path] = "success"
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            results[file_path] = f"error: {str(e)}"
    
    return results 