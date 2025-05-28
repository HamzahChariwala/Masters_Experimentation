#!/usr/bin/env python3
"""
Script: analyze_metrics.py

A consolidated script that performs all analysis steps on patching metrics:
1. Filters patching experiment results based on specified thresholds
2. Generates filtered metric files
3. Creates a cross-metric summary for neurons that exceed all thresholds
4. Produces distribution plots and histograms for all metrics

This script combines the functionality of filter_metrics.py and plot_metric_distributions.py
with easily configurable parameters.
"""
import os
import sys
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict

# Add the Neuron_Selection directory to the path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
neuron_selection_dir = os.path.dirname(script_dir)
if neuron_selection_dir not in sys.path:
    sys.path.insert(0, neuron_selection_dir)
    print(f"Added Neuron_Selection directory to Python path: {neuron_selection_dir}")

#################################################
# CONFIGURABLE PARAMETERS
#################################################

# Define thresholds for each metric
METRICS_THRESHOLDS = {
    'kl_divergence': 0.05,  # Threshold for KL divergence  
    'reverse_kl_divergence': 0.05,  # Threshold for reverse KL divergence  
    'confidence_margin_magnitude': 0.05,  # Threshold for confidence margin magnitude
    'undirected_saturating_chebyshev': 0.4,  # Threshold for undirected saturating Chebyshev (lowered from 0.5)
    'reversed_pearson_correlation': 0.05,  # Threshold for reversed Pearson correlation
    'reversed_undirected_saturating_chebyshev': 0.3,  # Threshold for reversed undirected saturating Chebyshev
    'top_logit_delta_magnitude': 0.1,  # Threshold for top logit delta magnitude
}

# Default plotting configuration
DEFAULT_METRICS_PAIRS = [
    # Standard metrics pair
    ["kl_divergence", "undirected_saturating_chebyshev"],
    # Additional metrics pair
    ["confidence_margin_magnitude", "reversed_pearson_correlation"]
]

# Plot configuration
PLOT_CONFIG = {
    'figsize': (12, 8),        # Figure size (width, height) in inches
    'dpi': 300,                # DPI for saved figures
    'bin_width': 0.05,         # Width of histogram bins
    'use_normalized': True,    # Whether to use pre-computed normalized values
    'alpha': 0.7,              # Transparency for histogram bars
    'line_width': 0.5,         # Line width for histogram edges
}

# Metrics to normalize (range 0-1)
METRICS_TO_NORMALIZE = [
    'kl_divergence', 
    'reverse_kl_divergence', 
    'confidence_margin_magnitude', 
    'reversed_pearson_correlation',
    'reversed_undirected_saturating_chebyshev',
    'top_logit_delta_magnitude'
    # undirected_saturating_chebyshev is NOT included here as it's already in an acceptable range
]

# Normalization ranges for specific metrics
# Define min/max values for normalization of specific metrics if needed
NORMALIZATION_RANGES = {
    # If specific ranges are needed, define them here
    # 'metric_name': {'min': min_value, 'max': max_value}
}

#################################################
# UTILITY FUNCTIONS
#################################################

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

def normalize_metric_value(metric_name: str, value: float) -> float:
    """
    Normalize a metric value to the range [0, 1].
    Some metrics already have natural bounds, others need scaling.
    
    Args:
        metric_name: Name of the metric
        value: The value to normalize
        
    Returns:
        Normalized value in [0, 1] range
    """
    # Return the value directly if it's already normalized or not a metric we want to normalize
    if metric_name not in METRICS_TO_NORMALIZE:
        return value
    
    # For KL divergence metrics, use a standard scaling approach
    # Maximum values observed in the dataset were around 0.2-0.3
    if metric_name == 'kl_divergence' or metric_name == 'reverse_kl_divergence':
        # Use a scaling factor to ensure values spread across the full [0,1] range
        # Most KL values are small, so use a log-like scaling that emphasizes small differences
        if value <= 0:
            return 0.0
        # Scale so that values around 0.2 (previously max) map to close to 1.0
        normalized = min(1.0, value / 0.2)
        return normalized
    
    # For confidence_margin_magnitude, scale based on observed maximums
    if metric_name == 'confidence_margin_magnitude':
        # Maximum observed was around 0.3, so scale appropriately
        normalized = min(1.0, value / 0.3)
        return normalized
        
    # For reversed_pearson_correlation
    if metric_name == 'reversed_pearson_correlation':
        # Maximum observed was around 0.2, scale to full range
        normalized = min(1.0, value / 0.2)
        return normalized
    
    # For reversed_undirected_saturating_chebyshev
    if metric_name == 'reversed_undirected_saturating_chebyshev':
        # This metric is already in range [0,1] since it's 1 - undirected_saturating_chebyshev
        # Just clamp to ensure it's in bounds
        return max(0.0, min(1.0, value))
    
    # For top_logit_delta_magnitude
    if metric_name == 'top_logit_delta_magnitude':
        # Scale based on observed maximums - most values are quite small
        # Maximum observed was around 0.5, so scale appropriately
        normalized = min(1.0, value / 0.5)
        return normalized
    
    # Default normalization: just clamp to [0, 1]
    return max(0.0, min(1.0, value))

#################################################
# FILTERING FUNCTIONS
#################################################

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
            
            # For undirected_saturating_chebyshev, use the mean value directly as it's already in an acceptable range
            if metric_name == 'undirected_saturating_chebyshev':
                if 'mean' in metric_data and metric_data['mean'] > threshold:
                    filtered_experiments[exp_name] = {
                        'metrics': {
                            metric_name: metric_data
                        },
                        'normalized_value': metric_data['mean'],  # Use mean as the normalized value for chebyshev
                        'mean': metric_data['mean'],
                        'input_count': exp_data.get('input_count', 0)
                    }
            # For other metrics, use the averaged_normalized_value
            elif metric_name in METRICS_TO_NORMALIZE:
                if 'averaged_normalized_value' in metric_data and metric_data['averaged_normalized_value'] > threshold:
                    filtered_experiments[exp_name] = {
                        'metrics': {
                            metric_name: metric_data
                        },
                        'normalized_value': metric_data['averaged_normalized_value'],
                        'mean': metric_data['mean'],
                        'input_count': exp_data.get('input_count', 0)
                    }
            # For any other metrics, use the mean value directly
            else:
                if 'mean' in metric_data and metric_data['mean'] > threshold:
                    filtered_experiments[exp_name] = {
                        'metrics': {
                            metric_name: metric_data
                        },
                        'normalized_value': metric_data['mean'],
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

def filter_metrics(agent_path: str, metrics_thresholds: Dict[str, float]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Filter metrics from patching summary files and generate separate output files.
    
    Args:
        agent_path: Path to the agent directory
        metrics_thresholds: Dictionary mapping metric names to threshold values
        
    Returns:
        Dictionary of common experiments across all metrics
    """
    # Paths to summary files
    results_dir = os.path.join(agent_path, "patching_results")
    filtered_dir = os.path.join(results_dir, "filtered")
    
    # Create filtered directory if it doesn't exist
    os.makedirs(filtered_dir, exist_ok=True)
    
    # Create analysis directory for additional outputs
    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    noising_summary_file = os.path.join(results_dir, "patching_summary_noising.json")
    denoising_summary_file = os.path.join(results_dir, "patching_summary_denoising.json")
    
    # Load summary data
    print(f"Loading summary data from {noising_summary_file} and {denoising_summary_file}")
    noising_data = load_summary_data(noising_summary_file)
    denoising_data = load_summary_data(denoising_summary_file)
    
    # Check if we have data
    if not noising_data:
        print(f"No noising summary data found at {noising_summary_file}")
        return {}
    
    if not denoising_data:
        print(f"No denoising summary data found at {denoising_summary_file}")
        return {}
    
    # Compute averaged normalization based on mean values across experiments
    print("Computing averaged normalization for noising data...")
    compute_averaged_normalization(noising_data, METRICS_TO_NORMALIZE)
    print("Computing averaged normalization for denoising data...")
    compute_averaged_normalization(denoising_data, METRICS_TO_NORMALIZE)
    
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
    
    # Write cross-metric summary directly to the main results directory
    cross_metric_file = os.path.join(results_dir, "cross_metric_summary.json")
    write_cross_metric_summary(cross_metric_file, cross_metric_common, metrics_info)
    
    return cross_metric_common

def normalize_summary_data(summary_data: Dict[str, Any], metrics_to_normalize: List[str]) -> None:
    """
    Add or update normalized values for specified metrics in the summary data.
    This modifies the summary_data in place.
    
    Args:
        summary_data: Dictionary containing the summary data
        metrics_to_normalize: List of metric names to normalize
    """
    # Process each experiment
    for exp_name, exp_data in summary_data.items():
        if 'metrics' not in exp_data:
            continue
            
        for metric_name in metrics_to_normalize:
            if metric_name in exp_data['metrics']:
                metric_data = exp_data['metrics'][metric_name]
                
                # If mean value exists but normalized value doesn't, add it
                if 'mean' in metric_data and ('normalized_value' not in metric_data or metric_data['normalized_value'] is None):
                    metric_data['normalized_value'] = normalize_metric_value(metric_name, metric_data['mean'])

def compute_averaged_normalization(summary_data: Dict[str, Any], metrics_to_normalize: List[str]) -> None:
    """
    Compute normalization based on the mean values across all experiments for specified metrics.
    This adds an 'averaged_normalized_value' field to each metric.
    
    Args:
        summary_data: Dictionary containing the summary data
        metrics_to_normalize: List of metric names to normalize based on their means
    """
    # For each metric, collect all mean values first
    for metric_name in metrics_to_normalize:
        mean_values = []
        experiments_with_metric = []
        
        # Collect all mean values for this metric
        for exp_name, exp_data in summary_data.items():
            if 'metrics' in exp_data and metric_name in exp_data['metrics']:
                metric_data = exp_data['metrics'][metric_name]
                if 'mean' in metric_data:
                    mean_values.append(metric_data['mean'])
                    experiments_with_metric.append(exp_name)
        
        # Compute min-max normalization based on the mean values
        if mean_values:
            normalized_means = normalize_values_to_01_range(mean_values)
            
            # Assign the normalized values back to the experiments
            for i, exp_name in enumerate(experiments_with_metric):
                summary_data[exp_name]['metrics'][metric_name]['averaged_normalized_value'] = normalized_means[i]
            
            print(f"Computed averaged normalization for {metric_name}: {len(mean_values)} experiments, range [{min(normalized_means):.6f}, {max(normalized_means):.6f}]")

def extract_metric_values(summary_data: Dict[str, Any], metric_name: str, use_normalized: bool = True) -> List[float]:
    """
    Extract all values for a specific metric from the summary data.
    
    Args:
        summary_data: Dictionary containing the summary data
        metric_name: Name of the metric to extract
        use_normalized: Whether to use pre-computed normalized values (default: True)
        
    Returns:
        List of metric values across all experiments
    """
    values = []
    
    for exp_name, exp_data in summary_data.items():
        if 'metrics' in exp_data and metric_name in exp_data['metrics']:
            metric_data = exp_data['metrics'][metric_name]
            
            # For undirected_saturating_chebyshev, always use the mean value directly
            if metric_name == 'undirected_saturating_chebyshev':
                if 'mean' in metric_data:
                    values.append(metric_data['mean'])
            # For other metrics that should be normalized, use the averaged_normalized_value
            elif metric_name in METRICS_TO_NORMALIZE:
                # Use the averaged normalized value if it exists
                if 'averaged_normalized_value' in metric_data:
                    values.append(metric_data['averaged_normalized_value'])
                # Fall back to mean if no averaged normalized value exists
                elif 'mean' in metric_data:
                    values.append(metric_data['mean'])
            # For any other metrics, use mean
            else:
                if 'mean' in metric_data:
                    values.append(metric_data['mean'])
    
    return values

#################################################
# PLOTTING FUNCTIONS
#################################################

def normalize_values_to_01_range(values: List[float]) -> List[float]:
    """
    Normalize values to [0, 1] range using min-max normalization.
    
    Args:
        values: List of values to normalize
        
    Returns:
        List of normalized values in [0, 1] range
    """
    if not values:
        return []
    
    # Filter out non-finite values
    finite_values = [v for v in values if np.isfinite(v)]
    if not finite_values:
        return [0.0] * len(values)
    
    min_val = min(finite_values)
    max_val = max(finite_values)
    
    # If min == max, return 0.5 for all values (or 0.0 if you prefer)
    if min_val == max_val:
        return [0.0] * len(values)
    
    # Normalize to [0, 1] using min-max normalization
    normalized = []
    for v in values:
        if np.isfinite(v):
            norm_val = (v - min_val) / (max_val - min_val)
            normalized.append(norm_val)
        else:
            normalized.append(0.0)
    
    return normalized

def normalize_values(values: List[float]) -> List[float]:
    """
    Normalize values to [0, 1] range.
    
    Args:
        values: List of values to normalize
        
    Returns:
        List of normalized values
    """
    if not values:
        return []
    
    # Filter out non-finite values
    finite_values = [v for v in values if np.isfinite(v)]
    if not finite_values:
        return [0.0] * len(values)
    
    min_val = min(finite_values)
    max_val = max(finite_values)
    
    # If min == max, return 0.5 for all values
    if min_val == max_val:
        return [0.5] * len(values)
    
    # Normalize to [0, 1]
    return [(v - min_val) / (max_val - min_val) if np.isfinite(v) else 0.0 for v in values]

def validate_normalized_values(values: List[float]) -> List[float]:
    """
    Validate and correct normalized values to ensure they fall in [0, 1] range.
    This handles any potential errors in the normalization process.
    
    Args:
        values: List of normalized values
        
    Returns:
        List of validated normalized values clamped to [0, 1]
    """
    # Clamp values to [0, 1] range
    return [max(0.0, min(1.0, v)) for v in values]

def save_histogram_data(values: List[float], metric_name: str, experiment_type: str, output_dir: str) -> str:
    """
    Save histogram data to a text file with 0.05 increments.
    
    Args:
        values: List of values to bin
        metric_name: Name of the metric
        experiment_type: Type of experiment (noising or denoising)
        output_dir: Directory to save the file
        
    Returns:
        Path to the saved file
    """
    # Create analysis subfolder
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Create bins from 0 to 1 with 0.05 increments
    bins = np.arange(0, 1.05, 0.05)
    hist, bin_edges = np.histogram(values, bins=bins)
    
    # Prepare histogram data as a string
    histogram_data = []
    histogram_data.append(f"--- {metric_name} ---")
    histogram_data.append(f"Min: {min(values):.6f}, Max: {max(values):.6f}, Mean: {np.mean(values):.6f}, Median: {np.median(values):.6f}")
    histogram_data.append(f"Total values: {len(values)}")
    histogram_data.append(f"Number of zeros: {sum(1 for v in values if v == 0)}, Number of ones: {sum(1 for v in values if v == 1)}")
    histogram_data.append("Bin Range,Count,Percentage")
    
    for i in range(len(hist)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        percentage = (hist[i] / len(values)) * 100 if values else 0
        histogram_data.append(f"{bin_start:.2f}-{bin_end:.2f},{hist[i]},{percentage:.2f}%")
    
    # Create output filename for individual metric (will be combined later)
    combined_file = os.path.join(analysis_dir, f"{experiment_type}_histograms.txt")
    
    # Append to the combined file for this experiment type
    with open(combined_file, 'a') as f:
        f.write("\n".join(histogram_data))
        f.write("\n\n")  # Add extra spacing between metrics
    
    print(f"Histogram data for {metric_name} added to {combined_file}")
    return combined_file

def plot_metric_distributions(
    agent_path: str,
    metrics_pairs: List[List[str]] = None,
    use_normalized: bool = True  # Set default to True to ensure we use normalized values
) -> None:
    """
    Generate distribution plots for metric pairs.
    
    Args:
        agent_path: Path to the agent directory
        metrics_pairs: List of lists containing pairs of metrics to plot together
        use_normalized: Whether to use pre-computed normalized values (default: True)
    """
    # Always use normalized values regardless of the parameter passed
    use_normalized = True
    
    # Set default metrics pairs if not provided
    if metrics_pairs is None:
        metrics_pairs = DEFAULT_METRICS_PAIRS
    
    # Set output directory
    results_dir = os.path.join(agent_path, "patching_results")
    
    # Create analysis subfolder
    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Clear existing combined histogram files
    noising_hist_file = os.path.join(analysis_dir, "noising_histograms.txt")
    denoising_hist_file = os.path.join(analysis_dir, "denoising_histograms.txt")
    
    # Clear existing files if they exist
    for file_path in [noising_hist_file, denoising_hist_file]:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Paths to summary files
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
    
    # The summary files already contain properly computed normalized values, so we don't need to recompute them
    # normalize_summary_data(noising_data, METRICS_TO_NORMALIZE)  # REMOVED - was overwriting correct values
    # normalize_summary_data(denoising_data, METRICS_TO_NORMALIZE)  # REMOVED - was overwriting correct values
    
    # Compute averaged normalization based on mean values across experiments
    print("Computing averaged normalization for noising data...")
    compute_averaged_normalization(noising_data, METRICS_TO_NORMALIZE)
    print("Computing averaged normalization for denoising data...")
    compute_averaged_normalization(denoising_data, METRICS_TO_NORMALIZE)
    
    # Get the magma colormap for consistent colors
    magma = plt.cm.magma
    
    # Define bins with 0.05 increments from 0 to 1
    bins = np.arange(0, 1.05, PLOT_CONFIG['bin_width'])
    
    # Process each pair of metrics
    for pair_index, metrics_pair in enumerate(metrics_pairs):
        pair_name = f"pair_{pair_index + 1}"
        print(f"\nPlotting metric pair {pair_index + 1}: {', '.join(metrics_pair)}")
        
        # Process noising metrics
        noising_plot_file = os.path.join(analysis_dir, f"noising_metrics_{pair_name}.png")
        
        # Calculate rows and columns for subplot grid
        n_metrics = len(metrics_pair)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
        
        # Create the figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=PLOT_CONFIG['figsize'])
        
        # Make axes a 2D array if it's 1D or a single axis
        if n_metrics == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Store value statistics for comparison
        value_stats = {}
        value_type = "normalized"
        
        # Generate colors for each metric from the magma colormap
        noising_color_positions = np.linspace(0.1, 0.9, len(metrics_pair))
        noising_colors = [magma(pos) for pos in noising_color_positions]
        
        # Customize colors if specific metrics are present
        if "kl_divergence" in metrics_pair and "undirected_saturating_chebyshev" in metrics_pair:
            kl_index = metrics_pair.index("kl_divergence")
            cheb_index = metrics_pair.index("undirected_saturating_chebyshev")
            noising_colors[kl_index] = magma(0.15)  # Purple for KL divergence
            noising_colors[cheb_index] = magma(0.55)  # Magenta for Chebyshev
        
        if "confidence_margin_magnitude" in metrics_pair and "reversed_pearson_correlation" in metrics_pair:
            cm_index = metrics_pair.index("confidence_margin_magnitude")
            pearson_index = metrics_pair.index("reversed_pearson_correlation")
            noising_colors[cm_index] = magma(0.75)  # Orange-red for confidence margin
            noising_colors[pearson_index] = magma(0.85)  # Yellowish for pearson
        
        # Plot each metric in its own subplot
        for i, metric in enumerate(metrics_pair):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Extract values - ALWAYS using normalized values
            values = extract_metric_values(noising_data, metric, use_normalized=True)
            
            if not values:
                print(f"No values found for noising metric: {metric}")
                ax.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
                continue
            
            # Validate the normalized values to ensure they're in [0, 1]
            values = validate_normalized_values(values)
            
            # Get statistics about the values
            min_val = min(values)
            max_val = max(values)
            mean_val = np.mean(values)
            median_val = np.median(values)
            
            # Store statistics for debugging
            value_stats[metric] = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "median": median_val,
                "count": len(values),
                "zeros": sum(1 for v in values if v == 0),
                "ones": sum(1 for v in values if v == 1),
                "distribution": np.histogram(values, bins=10, range=(0, 1))[0].tolist()
            }
            
            # Print value distribution details
            print(f"\nValue distribution for {metric} ({value_type}):")
            print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}, Median: {median_val:.4f}")
            print(f"  Total values: {len(values)}, Zeros: {value_stats[metric]['zeros']}, Ones: {value_stats[metric]['ones']}")
            print(f"  Histogram: {value_stats[metric]['distribution']}")
            
            # Save histogram data to combined file
            save_histogram_data(values, metric, "noising", results_dir)
            
            # Plot histogram
            sns.histplot(
                values, 
                bins=bins,
                color=noising_colors[i],
                alpha=PLOT_CONFIG['alpha'],
                edgecolor='black',
                linewidth=PLOT_CONFIG['line_width'],
                ax=ax
            )
            
            # Add vertical line for mean
            ax.axvline(x=mean_val, color='black', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
            
            # Set axis limits and labels
            ax.set_xlim(0, 1)
            ax.set_title(f"{metric}\n(n={len(values)}, zeros={value_stats[metric]['zeros']}, ones={value_stats[metric]['ones']})")
            ax.set_xlabel("Normalized Value (0-1)")
            ax.set_ylabel("Count")
            
            # Add legend for mean line
            ax.legend()
        
        # Remove any unused subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        # Set a global title
        fig.suptitle(f"Histogram of {value_type.capitalize()} Noising Metrics", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for the suptitle
        plt.savefig(noising_plot_file, dpi=PLOT_CONFIG['dpi'])
        plt.close()
        
        # Process denoising metrics
        denoising_plot_file = os.path.join(analysis_dir, f"denoising_metrics_{pair_name}.png")
        
        # Create the figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=PLOT_CONFIG['figsize'])
        
        # Make axes a 2D array if it's 1D or a single axis
        if n_metrics == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Use the same colors for denoising to maintain consistency
        denoising_colors = noising_colors.copy()
        
        # Plot each metric in its own subplot
        for i, metric in enumerate(metrics_pair):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Use reverse_kl_divergence for denoising if the metric is kl_divergence
            denoising_metric = 'reverse_kl_divergence' if metric == 'kl_divergence' else metric
            
            # Extract values - ALWAYS using normalized values
            values = extract_metric_values(denoising_data, denoising_metric, use_normalized=True)
            
            if not values:
                print(f"No values found for denoising metric: {denoising_metric}")
                ax.text(0.5, 0.5, f"No data for {denoising_metric}", ha='center', va='center')
                continue
            
            # Validate the normalized values to ensure they're in [0, 1]
            values = validate_normalized_values(values)
            
            # Get statistics about the values
            min_val = min(values)
            max_val = max(values)
            mean_val = np.mean(values)
            median_val = np.median(values)
            
            # Store statistics for debugging
            value_stats[denoising_metric] = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "median": median_val,
                "count": len(values),
                "zeros": sum(1 for v in values if v == 0),
                "ones": sum(1 for v in values if v == 1),
                "distribution": np.histogram(values, bins=10, range=(0, 1))[0].tolist()
            }
            
            # Print value distribution details
            print(f"\nValue distribution for {denoising_metric} ({value_type}):")
            print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}, Median: {median_val:.4f}")
            print(f"  Total values: {len(values)}, Zeros: {value_stats[denoising_metric]['zeros']}, Ones: {value_stats[denoising_metric]['ones']}")
            print(f"  Histogram: {value_stats[denoising_metric]['distribution']}")
            
            # Save histogram data to combined file
            save_histogram_data(values, denoising_metric, "denoising", results_dir)
            
            # Plot histogram
            sns.histplot(
                values, 
                bins=bins,
                color=denoising_colors[i],
                alpha=PLOT_CONFIG['alpha'],
                edgecolor='black',
                linewidth=PLOT_CONFIG['line_width'],
                ax=ax
            )
            
            # Add vertical line for mean
            ax.axvline(x=mean_val, color='black', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
            
            # Set axis limits and labels
            ax.set_xlim(0, 1)
            ax.set_title(f"{denoising_metric}\n(n={len(values)}, zeros={value_stats[denoising_metric]['zeros']}, ones={value_stats[denoising_metric]['ones']})")
            ax.set_xlabel("Normalized Value (0-1)")
            ax.set_ylabel("Count")
            
            # Add legend for mean line
            ax.legend()
        
        # Remove any unused subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        # Set a global title
        fig.suptitle(f"Histogram of {value_type.capitalize()} Denoising Metrics", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for the suptitle
        plt.savefig(denoising_plot_file, dpi=PLOT_CONFIG['dpi'])
        plt.close()
        
        print(f"Plots saved to {noising_plot_file} and {denoising_plot_file}")

#################################################
# MAIN ANALYSIS FUNCTION
#################################################

def analyze_metrics(agent_path: str, metrics_thresholds: Dict[str, float] = None, metrics_pairs: List[List[str]] = None) -> None:
    """
    Perform comprehensive metrics analysis: filtering, plotting, and summary generation.
    
    Args:
        agent_path: Path to the agent directory
        metrics_thresholds: Dictionary mapping metric names to threshold values
        metrics_pairs: List of lists containing pairs of metrics to plot together
    """
    # Set default metrics_thresholds if not provided
    if metrics_thresholds is None:
        metrics_thresholds = METRICS_THRESHOLDS
    
    # Set default metrics_pairs if not provided
    if metrics_pairs is None:
        metrics_pairs = DEFAULT_METRICS_PAIRS
    
    # Filter metrics and generate filtered files
    print("\n===== FILTERING METRICS =====")
    cross_metric_common = filter_metrics(agent_path, metrics_thresholds)
    
    # Plot metric distributions
    print("\n===== PLOTTING METRIC DISTRIBUTIONS =====")
    plot_metric_distributions(agent_path, metrics_pairs, PLOT_CONFIG['use_normalized'])
    
    # Print final summary
    print("\n===== ANALYSIS COMPLETE =====")
    print(f"- Generated filtered files for {len(metrics_thresholds)} metrics")
    print(f"- Created distribution plots for {len(metrics_pairs)} metric pairs")
    print(f"- Found {len(cross_metric_common)} neurons that exceed thresholds for all metrics")
    
    # Display the common neurons if any
    if cross_metric_common:
        print("\nNeurons that exceed all thresholds:")
        for neuron_name in cross_metric_common.keys():
            print(f"- {neuron_name}")

def main():
    parser = argparse.ArgumentParser(description="Analyze patching metrics: filter, plot, and summarize")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--experiment_file", type=str,
                       help="Path to a JSON file with patch experiment definitions")
    parser.add_argument("--custom_thresholds", type=str,
                       help="Comma-separated list of custom thresholds in format 'metric:threshold', e.g., 'kl_divergence:0.1,undirected_saturating_chebyshev:0.6'")
    
    args = parser.parse_args()
    
    # Process custom thresholds if provided
    metrics_thresholds = METRICS_THRESHOLDS.copy()
    if args.custom_thresholds:
        threshold_items = args.custom_thresholds.split(',')
        for item in threshold_items:
            if ':' in item:
                metric, threshold_str = item.split(':')
                try:
                    threshold = float(threshold_str)
                    metrics_thresholds[metric] = threshold
                    print(f"Using custom threshold for {metric}: {threshold}")
                except ValueError:
                    print(f"Invalid threshold value for {metric}: {threshold_str}")
    
    # If experiment file is provided, run activation patching first
    if args.experiment_file:
        # Import the activation_patching module using the correct path
        sys.path.insert(0, neuron_selection_dir)
        from activation_patching import main as run_activation_patching
        
        print(f"\n===== RUNNING ACTIVATION PATCHING WITH EXPERIMENT FILE =====")
        print(f"Agent path: {args.agent_path}")
        print(f"Experiment file: {args.experiment_file}")
        
        # Set up arguments for activation patching
        activation_patching_args = [
            "--agent_path", args.agent_path,
            "--patches_file", args.experiment_file
        ]
        
        # Run activation patching
        sys.argv = ["activation_patching.py"] + activation_patching_args
        run_activation_patching()
    
    # Run the comprehensive analysis
    analyze_metrics(args.agent_path, metrics_thresholds)

if __name__ == "__main__":
    main() 