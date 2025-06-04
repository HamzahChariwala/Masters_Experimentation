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

# Import circuit verification functions
try:
    from CircuitTooling.cumulative_patching_generator import generate_cumulative_experiments
except ImportError as e:
    print(f"Warning: Could not import cumulative experiment generator: {e}")
    generate_cumulative_experiments = None

try:
    sys.path.insert(0, os.path.join(neuron_selection_dir, 'ExperimentTooling'))
    from ExperimentTooling.coalition_to_experiments import main as convert_coalitions_to_experiments
except ImportError as e:
    print(f"Warning: Could not import coalition to experiments converter: {e}")
    convert_coalitions_to_experiments = None

try:
    from CircuitTooling.run_circuit_experiments import run_all_circuit_experiments
except ImportError as e:
    print(f"Warning: Could not import circuit experiments runner: {e}")
    run_all_circuit_experiments = None

try:
    from CircuitTooling.visualize_circuit_results import create_circuit_visualization
except ImportError as e:
    print(f"Warning: Could not import circuit visualization: {e}")
    create_circuit_visualization = None

#################################################
# CONFIGURABLE PARAMETERS
#################################################

# Define thresholds for each metric
METRICS_THRESHOLDS = {
    'kl_divergence': 0.0,  # Threshold for KL divergence  
    'reverse_kl_divergence': 0.0,  # Threshold for reverse KL divergence  
    'confidence_margin_magnitude': 0.0,  # Threshold for confidence margin magnitude
    'undirected_saturating_chebyshev': 0.0,  # Threshold for undirected saturating Chebyshev 
    'reversed_pearson_correlation': 0.0,  # Threshold for reversed Pearson correlation
    'reversed_undirected_saturating_chebyshev': 0.0,  # Threshold for reversed undirected saturating Chebyshev
    'top_logit_delta_magnitude': 0.0,  # Threshold for top logit delta magnitude
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
    'top_logit_delta_magnitude',
    'logit_difference_norm'
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
    
    # For logit_difference_norm
    if metric_name == 'logit_difference_norm':
        # Scale based on observed maximums - typically around 0.02-0.05
        # Maximum observed was around 0.05, so scale appropriately
        normalized = min(1.0, value / 0.05)
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
    Return all experiments ranked by their within-run normalized importance for the specified metric.
    When threshold is 0, all experiments are included and ranked by importance.
    
    Args:
        summary_data: Dictionary containing the summary data
        metric_name: Name of the metric to filter by
        threshold: Threshold value for filtering (when 0, all experiments included)
        
    Returns:
        Dictionary of experiments with their metrics, ranked by importance
    """
    if threshold == 0.0:
        # New behavior: return all experiments ranked by within-run normalized importance
        within_run_normalized = compute_within_run_normalization(summary_data, metric_name)
        
        # Create filtered experiments with all neurons, including normalized ranking
        filtered_experiments = {}
        for exp_name, exp_data in summary_data.items():
            if exp_name in within_run_normalized and 'metrics' in exp_data and metric_name in exp_data['metrics']:
                metric_data = exp_data['metrics'][metric_name]
                filtered_experiments[exp_name] = {
                    'metrics': {
                        metric_name: metric_data
                    },
                    'within_run_normalized_value': within_run_normalized[exp_name],
                    'mean': metric_data['mean'],
                    'input_count': exp_data.get('input_count', 0)
                }
        
        return filtered_experiments
    
    else:
        # Original behavior: filter by threshold
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
    Calculate summary statistics for a set of filtered experiments.
    
    Args:
        filtered_experiments: Dictionary of filtered experiments
        metric_name: Name of the metric
        is_common: Whether these are common experiments (with both noising and denoising data)
        
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
            # Handle both old and new formats
            if 'noising' in exp_data and 'denoising' in exp_data:
                # New format with nested noising/denoising data
                noising_norm = exp_data['noising'].get('within_run_normalized_value', 
                                                     exp_data['noising'].get('normalized_value', 0))
                denoising_norm = exp_data['denoising'].get('within_run_normalized_value',
                                                         exp_data['denoising'].get('normalized_value', 0))
                noising_raw = exp_data['noising']['mean']
                denoising_raw = exp_data['denoising']['mean']
                
                # Average of noising and denoising
                normalized_values.append((noising_norm + denoising_norm) / 2)
                raw_values.append((noising_raw + denoising_raw) / 2)
            else:
                # Old format - shouldn't happen in new logic but keeping for safety
                norm_val = exp_data.get('within_run_normalized_value', 
                                      exp_data.get('normalized_value', 0))
                normalized_values.append(norm_val)
                raw_values.append(exp_data['mean'])
    else:
        # For regular experiments - handle both new and old formats
        normalized_values = []
        raw_values = []
        for exp_data in filtered_experiments.values():
            # Try new format first, then fall back to old format
            norm_val = exp_data.get('within_run_normalized_value', 
                                   exp_data.get('normalized_value', 0))
            normalized_values.append(norm_val)
            raw_values.append(exp_data['mean'])
    
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
        # Get importance scores from both noising and denoising
        noising_score = 0.0
        denoising_score = 0.0
        
        if 'noising' in exp_data:
            noising_score = exp_data['noising'].get('within_run_normalized_value', 0.0)
        if 'denoising' in exp_data:
            denoising_score = exp_data['denoising'].get('within_run_normalized_value', 0.0)
        
        averaged_score = (noising_score + denoising_score) / 2
        
        # Create simplified entry with essential information
        averaged_entry = {
            'averaged_normalized_value': averaged_score
        }
        
        # Add simplified noising data if available
        if 'noising' in exp_data:
            averaged_entry['noising'] = {
                'within_run_normalized_value': exp_data['noising']['within_run_normalized_value'],
                'raw_mean': exp_data['noising']['mean']
            }
        
        # Add simplified denoising data if available
        if 'denoising' in exp_data:
            averaged_entry['denoising'] = {
                'within_run_normalized_value': exp_data['denoising']['within_run_normalized_value'],
                'raw_mean': exp_data['denoising']['mean']
            }
        
        averaged_experiments[exp_name] = averaged_entry
    
    # Sort by averaged normalized value (highest to lowest)
    averaged_experiments_sorted = dict(sorted(
        averaged_experiments.items(), 
        key=lambda x: x[1]['averaged_normalized_value'], 
        reverse=True
    ))
    
    # Create 'highest' section - rank by highest normalized value across noising and denoising
    highest_experiments = {}
    for exp_name, exp_data in common_experiments.items():
        # Get importance scores from both noising and denoising
        noising_score = 0.0
        denoising_score = 0.0
        
        if 'noising' in exp_data:
            noising_score = exp_data['noising'].get('within_run_normalized_value', 0.0)
        if 'denoising' in exp_data:
            denoising_score = exp_data['denoising'].get('within_run_normalized_value', 0.0)
        
        highest_score = max(noising_score, denoising_score)
        
        # Create simplified entry with essential information
        highest_entry = {
            'highest_normalized_value': highest_score
        }
        
        # Add simplified noising data if available
        if 'noising' in exp_data:
            highest_entry['noising'] = {
                'within_run_normalized_value': exp_data['noising']['within_run_normalized_value'],
                'raw_mean': exp_data['noising']['mean']
            }
        
        # Add simplified denoising data if available
        if 'denoising' in exp_data:
            highest_entry['denoising'] = {
                'within_run_normalized_value': exp_data['denoising']['within_run_normalized_value'],
                'raw_mean': exp_data['denoising']['mean']
            }
        
        highest_experiments[exp_name] = highest_entry
    
    # Sort by highest normalized value (highest to lowest)
    highest_experiments_sorted = dict(sorted(
        highest_experiments.items(), 
        key=lambda x: x[1]['highest_normalized_value'], 
        reverse=True
    ))
    
    # Prepare the output data with new structure (averaged and highest sections first)
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
    print(f"  Averaged: {len(averaged_experiments_sorted)} experiments (sorted by averaged importance)")
    print(f"  Highest: {len(highest_experiments_sorted)} experiments (sorted by highest importance)")

def write_cross_metric_summary(
    output_file: str,
    cross_metric_common: Dict[str, Dict[str, Any]],
    metrics_info: Dict[str, Dict[str, Any]]
) -> None:
    """
    Write summary of neurons ranked by average importance across all metrics.
    
    Args:
        output_file: Path to the output file
        cross_metric_common: Dictionary of experiments ranked by average importance
        metrics_info: Dictionary with information about each metric
    """
    # Calculate summary count
    common_count = len(cross_metric_common)
    
    # Calculate some summary statistics
    if cross_metric_common:
        scores = [exp_data['final_average_importance'] for exp_data in cross_metric_common.values()]
        max_score = max(scores)
        min_score = min(scores)
        avg_score = sum(scores) / len(scores)
    else:
        max_score = min_score = avg_score = 0.0
    
    # Prepare the output data
    output_data = {
        'description': 'Neurons ranked by average importance across all metrics (within-run normalized)',
        'methodology': {
            'normalization': 'Within-run min-max normalization (0-1) for each metric separately',
            'averaging': 'Average of noising and denoising normalized scores per metric',
            'final_ranking': 'Average across all metric scores, sorted by importance (high to low)'
        },
        'summary_statistics': {
            'total_neurons': common_count,
            'max_importance_score': max_score,
            'min_importance_score': min_score,
            'average_importance_score': avg_score
        },
        'metrics_info': metrics_info,
        'common_count': common_count,  # Keep for backward compatibility
        'common_neurons': cross_metric_common
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nCross-metric summary saved to {output_file}")
    print(f"  Total neurons ranked: {common_count}")
    print(f"  Importance score range: [{min_score:.4f}, {max_score:.4f}]")
    print(f"  Average importance score: {avg_score:.4f}")

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
        
        # Collect all experiment names from both runs
        noising_exp_names = set(noising_filtered.keys())
        denoising_exp_names = set(denoising_filtered.keys())
        all_exp_names = noising_exp_names.union(denoising_exp_names)
        
        # Create experiments dictionary including all experiments
        all_experiments = {}
        for exp_name in all_exp_names:
            experiment_data = {}
            
            # Add noising data if available
            if exp_name in noising_filtered:
                experiment_data['noising'] = noising_filtered[exp_name]
            
            # Add denoising data if available  
            if exp_name in denoising_filtered:
                experiment_data['denoising'] = denoising_filtered[exp_name]
            
            all_experiments[exp_name] = experiment_data
        
        # Store all experiments for this metric
        metrics_common_experiments[metric_name] = all_experiments
        
        # For metrics_info, count experiments that have both noising and denoising
        common_count = len([exp for exp in all_experiments.values() 
                          if 'noising' in exp and 'denoising' in exp])
        
        # Store metric info
        metrics_info[metric_name] = {
            'threshold': threshold,
            'common_count': common_count,  # Backward compatibility 
            'total_count': len(all_experiments),
            'noising_count': len(noising_exp_names),
            'denoising_count': len(denoising_exp_names)
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
            all_experiments
        )
    
    # Compute averaged importance rankings across all metrics and runs
    cross_metric_common = {}
    
    if metrics_common_experiments:
        # Collect all unique experiment names across all metrics
        all_exp_names = set()
        for metric_name, metric_experiments in metrics_common_experiments.items():
            all_exp_names.update(metric_experiments.keys())
        
        # For each experiment, calculate average within-run normalized importance
        experiment_avg_scores = {}
        
        for exp_name in all_exp_names:
            metric_scores = []
            metric_data_collection = {}
            
            # Collect scores for this experiment across all metrics
            for metric_name, metric_experiments in metrics_common_experiments.items():
                if exp_name in metric_experiments:
                    # Get noising and denoising scores, handling missing data
                    experiment_data = metric_experiments[exp_name]
                    
                    noising_score = 0.0
                    denoising_score = 0.0
                    valid_scores = 0
                    
                    if 'noising' in experiment_data:
                        noising_score = experiment_data['noising'].get('within_run_normalized_value', 0.0)
                        valid_scores += 1
                    
                    if 'denoising' in experiment_data:
                        denoising_score = experiment_data['denoising'].get('within_run_normalized_value', 0.0)
                        valid_scores += 1
                    
                    # Average the available scores (don't penalize missing data)
                    if valid_scores > 0:
                        avg_score = (noising_score + denoising_score) / valid_scores
                    else:
                        avg_score = 0.0
                    
                    metric_scores.append(avg_score)
                    
                    # Store detailed metric data
                    metric_data_collection[metric_name] = metric_experiments[exp_name]
                else:
                    # If experiment not found in this metric, assign score of 0
                    metric_scores.append(0.0)
                    metric_data_collection[metric_name] = None
            
            # Calculate overall average score across all metrics for this experiment
            if metric_scores:
                overall_avg_score = sum(metric_scores) / len(metric_scores)
                experiment_avg_scores[exp_name] = {
                    'average_importance_score': overall_avg_score,
                    'metric_details': metric_data_collection
                }
        
        # Sort experiments by their average importance score (descending)
        sorted_experiments = sorted(experiment_avg_scores.items(), 
                                   key=lambda x: x[1]['average_importance_score'], 
                                   reverse=True)
        
        # Create the cross-metric common dictionary with sorted ordering
        for exp_name, exp_data in sorted_experiments:
            # Create simplified entry with just the essential metrics
            simplified_entry = {
                'final_average_importance': exp_data['average_importance_score']
            }
            
            # Add normalized scores for each metric
            for metric_name in metrics_common_experiments.keys():
                if metric_name in exp_data['metric_details'] and exp_data['metric_details'][metric_name] is not None:
                    # Calculate the average normalized value for this metric
                    metric_data = exp_data['metric_details'][metric_name]
                    
                    noising_score = 0.0
                    denoising_score = 0.0
                    valid_scores = 0
                    
                    if 'noising' in metric_data:
                        noising_score = metric_data['noising'].get('within_run_normalized_value', 0.0)
                        valid_scores += 1
                    
                    if 'denoising' in metric_data:
                        denoising_score = metric_data['denoising'].get('within_run_normalized_value', 0.0)
                        valid_scores += 1
                    
                    # Average the available scores
                    if valid_scores > 0:
                        avg_score = (noising_score + denoising_score) / valid_scores
                    else:
                        avg_score = 0.0
                    
                    simplified_entry[f'{metric_name}_normalized'] = avg_score
                else:
                    # If metric not available for this neuron, set to 0
                    simplified_entry[f'{metric_name}_normalized'] = 0.0
            
            cross_metric_common[exp_name] = simplified_entry
    
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
    Generate individual distribution plots for each metric.
    
    Args:
        agent_path: Path to the agent directory
        metrics_pairs: List of lists containing pairs of metrics to plot together (unused, kept for compatibility)
        use_normalized: Whether to use pre-computed normalized values (default: True)
    """
    # Always use normalized values regardless of the parameter passed
    use_normalized = True
    
    # List of all metrics to plot individually (sorted alphabetically for consistent color assignment)
    metrics_to_plot = sorted([
        'kl_divergence',
        'reverse_kl_divergence', 
        'confidence_margin_magnitude',
        'undirected_saturating_chebyshev',
        'reversed_pearson_correlation',
        'reversed_undirected_saturating_chebyshev',
        'top_logit_delta_magnitude',
        'logit_difference_norm'
    ])
    
    # Set output directory
    results_dir = os.path.join(agent_path, "patching_results")
    
    # Create analysis subfolder and separate noising/denoising folders
    analysis_dir = os.path.join(results_dir, "analysis")
    noising_plots_dir = os.path.join(analysis_dir, "noising")
    denoising_plots_dir = os.path.join(analysis_dir, "denoising")
    
    os.makedirs(noising_plots_dir, exist_ok=True)
    os.makedirs(denoising_plots_dir, exist_ok=True)
    
    # Remove old paired plots
    old_plot_patterns = [
        "noising_metrics_pair_*.png",
        "denoising_metrics_pair_*.png"
    ]
    for pattern in old_plot_patterns:
        import glob
        for old_plot in glob.glob(os.path.join(analysis_dir, pattern)):
            if os.path.exists(old_plot):
                os.remove(old_plot)
                print(f"Removed old plot: {old_plot}")
    
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
    
    # Compute averaged normalization based on mean values across experiments
    print("Computing averaged normalization for noising data...")
    compute_averaged_normalization(noising_data, METRICS_TO_NORMALIZE)
    print("Computing averaged normalization for denoising data...")
    compute_averaged_normalization(denoising_data, METRICS_TO_NORMALIZE)
    
    # Get the magma colormap for consistent colors
    magma = plt.cm.magma
    
    # Create color mapping for each metric (alphabetically ordered)
    # Use a range from 0.85 to 0.15 to reverse the colors (avoid very light and very dark colors)
    num_metrics = len(metrics_to_plot)
    color_positions = np.linspace(0.85, 0.15, num_metrics)
    metric_colors = {metric: magma(color_positions[i]) for i, metric in enumerate(metrics_to_plot)}
    
    # Define bins with 0.05 increments from 0 to 1
    bins = np.arange(0, 1.05, PLOT_CONFIG['bin_width'])
    
    # Process each metric individually
    for metric in metrics_to_plot:
        print(f"\nPlotting metric: {metric}")
        
        # Get the assigned color for this metric
        metric_color = metric_colors[metric]
        
        # Process noising data
        noising_plot_file = os.path.join(noising_plots_dir, f"{metric}.png")
        
        # Create the figure for noising
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Extract values - ALWAYS using normalized values for metrics that should be normalized
        if metric == 'undirected_saturating_chebyshev':
            # This metric is not normalized, use original values
            values = extract_metric_values(noising_data, metric, use_normalized=False)
        else:
            values = extract_metric_values(noising_data, metric, use_normalized=True)
        
        if not values:
            print(f"No values found for noising metric: {metric}")
            plt.close(fig)
            continue
        
        # Validate values if normalized
        if metric != 'undirected_saturating_chebyshev':
            values = validate_normalized_values(values)
        
        # Get statistics about the values
        min_val = min(values)
        max_val = max(values)
        mean_val = np.mean(values)
        median_val = np.median(values)
        
        # Print value distribution details
        value_type = "original" if metric == 'undirected_saturating_chebyshev' else "normalized"
        print(f"  Noising {metric} ({value_type}): Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
        
        # Save histogram data to combined file
        save_histogram_data(values, metric, "noising", results_dir)
        
        # Use appropriate bins based on metric type
        plot_bins = bins if metric != 'undirected_saturating_chebyshev' else np.arange(0, max(1, max_val) + 0.05, 0.05)
        
        # Plot histogram
        sns.histplot(
            values, 
            bins=plot_bins,
            color=metric_color,
            alpha=PLOT_CONFIG['alpha'],
            edgecolor='black',
            linewidth=PLOT_CONFIG['line_width'],
            ax=ax
        )
        
        # Add vertical line for mean
        ax.axvline(x=mean_val, color='black', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
        
        # Set axis limits and labels
        if metric == 'undirected_saturating_chebyshev':
            ax.set_xlim(0, max(1, max_val))
        else:
            ax.set_xlim(0, 1)
        
        ax.set_title(f"Noising: {metric}\n(n={len(values)})")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(noising_plot_file, dpi=PLOT_CONFIG['dpi'])
        plt.close()
        
        # Process denoising data  
        denoising_plot_file = os.path.join(denoising_plots_dir, f"{metric}.png")
        
        # Create the figure for denoising
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # For denoising, use reverse_kl_divergence if the metric is kl_divergence
        denoising_metric = 'reverse_kl_divergence' if metric == 'kl_divergence' else metric
        
        # Extract values
        if denoising_metric == 'undirected_saturating_chebyshev':
            values = extract_metric_values(denoising_data, denoising_metric, use_normalized=False)
        else:
            values = extract_metric_values(denoising_data, denoising_metric, use_normalized=True)
        
        if not values:
            print(f"No values found for denoising metric: {denoising_metric}")
            plt.close(fig)
            continue
        
        # Validate values if normalized
        if denoising_metric != 'undirected_saturating_chebyshev':
            values = validate_normalized_values(values)
        
        # Get statistics about the values
        min_val = min(values)
        max_val = max(values)
        mean_val = np.mean(values)
        median_val = np.median(values)
        
        # Print value distribution details
        value_type = "original" if denoising_metric == 'undirected_saturating_chebyshev' else "normalized"
        print(f"  Denoising {denoising_metric} ({value_type}): Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
        
        # Save histogram data to combined file
        save_histogram_data(values, denoising_metric, "denoising", results_dir)
        
        # Use appropriate bins based on metric type
        plot_bins = bins if denoising_metric != 'undirected_saturating_chebyshev' else np.arange(0, max(1, max_val) + 0.05, 0.05)
        
        # Plot histogram
        sns.histplot(
            values, 
            bins=plot_bins,
            color=metric_color,  # Use the same color as the noising plot
            alpha=PLOT_CONFIG['alpha'],
            edgecolor='black',
            linewidth=PLOT_CONFIG['line_width'],
            ax=ax
        )
        
        # Add vertical line for mean
        ax.axvline(x=mean_val, color='black', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
        
        # Set axis limits and labels
        if denoising_metric == 'undirected_saturating_chebyshev':
            ax.set_xlim(0, max(1, max_val))
        else:
            ax.set_xlim(0, 1)
        
        ax.set_title(f"Denoising: {denoising_metric}\n(n={len(values)})")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(denoising_plot_file, dpi=PLOT_CONFIG['dpi'])
        plt.close()
        
        print(f"  Plots saved: {noising_plot_file} and {denoising_plot_file}")
    
    print(f"\nAll plots saved to {noising_plots_dir} and {denoising_plots_dir}")
    print(f"Colors assigned in alphabetical order using magma colormap:")
    for i, metric in enumerate(metrics_to_plot):
        print(f"  {i+1}. {metric} -> magma({color_positions[i]:.2f})")

#################################################
# MAIN ANALYSIS FUNCTION
#################################################

def analyze_metrics(agent_path: str, metrics_thresholds: Dict[str, float] = None, metrics_pairs: List[List[str]] = None, max_experiments: int = 30) -> None:
    """
    Perform comprehensive metrics analysis: filtering, plotting, and summary generation.
    
    Args:
        agent_path: Path to the agent directory
        metrics_thresholds: Dictionary mapping metric names to threshold values
        metrics_pairs: List of lists containing pairs of metrics to plot together
        max_experiments: Maximum number of cumulative experiments to generate for circuit verification (default: 30)
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
    
    # Run circuit verification workflow
    print("\n===== CIRCUIT VERIFICATION WORKFLOW =====")
    if generate_cumulative_experiments:
        print(f"Generating cumulative coalition experiments (max {max_experiments} experiments per metric)...")
        try:
            generate_cumulative_experiments(agent_path, max_experiments=max_experiments)
            
            # Convert coalition files to experiment format
            if convert_coalitions_to_experiments:
                print("Converting coalition files to experiment format...")
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
                print("Skipping coalition to experiments conversion (module not available)")
                
            # Run circuit experiments
            if run_all_circuit_experiments:
                print("Running circuit experiments...")
                try:
                    circuit_results = run_all_circuit_experiments(agent_path, subfolder="descending/results")
                    print(f"Circuit experiments completed: {circuit_results.get('successful', 0)} successful")
                except Exception as e:
                    print(f"Error running circuit experiments: {e}")
                
                # Generate visualizations
                if create_circuit_visualization:
                    print("Generating circuit verification visualizations...")
                    try:
                        create_circuit_visualization(agent_path, max_experiments=max_experiments, subfolder="descending")
                        print("Circuit visualization completed")
                    except Exception as e:
                        print(f"Error generating circuit visualization: {e}")
                else:
                    print("Skipping visualization generation (module not available)")
            else:
                print("Skipping circuit experiments (module not available)")
                
        except Exception as e:
            print(f"Error in circuit verification workflow: {e}")
    else:
        print("Skipping cumulative experiment generation (module not available)")
    
    # Display the common neurons if any
    if cross_metric_common:
        print(f"\nNeurons that exceed all thresholds (total: {len(cross_metric_common)}):")
        # Show top 10 neurons
        for i, neuron_name in enumerate(list(cross_metric_common.keys())[:10]):
            importance = cross_metric_common[neuron_name]['final_average_importance']
            print(f"{i+1:2d}. {neuron_name}: {importance:.4f}")
        if len(cross_metric_common) > 10:
            print(f"    ... and {len(cross_metric_common) - 10} more neurons")
    else:
        print("\nNo neurons found that exceed thresholds for all metrics")

def compute_within_run_normalization(summary_data: Dict[str, Any], metric_name: str) -> Dict[str, float]:
    """
    Compute normalization within a single run (noising or denoising) for a specific metric.
    This creates 0-1 normalized values relative to other neurons in the same run.
    
    Args:
        summary_data: Dictionary containing the summary data for one run
        metric_name: Name of the metric to normalize
        
    Returns:
        Dictionary mapping experiment names to normalized values [0-1]
    """
    # Collect all mean values for this metric in this run
    metric_values = []
    experiment_names = []
    
    for exp_name, exp_data in summary_data.items():
        if 'metrics' in exp_data and metric_name in exp_data['metrics']:
            metric_data = exp_data['metrics'][metric_name]
            if 'mean' in metric_data:
                metric_values.append(metric_data['mean'])
                experiment_names.append(exp_name)
    
    # Normalize to [0-1] range using min-max normalization
    normalized_values = {}
    if metric_values:
        min_val = min(metric_values)
        max_val = max(metric_values)
        
        if max_val != min_val:
            for i, exp_name in enumerate(experiment_names):
                original_value = metric_values[i]
                
                # Special handling for reversed_undirected_saturating_chebyshev:
                # If the original value is exactly 0, keep it as 0
                if (metric_name == 'reversed_undirected_saturating_chebyshev' and 
                    original_value == 0.0):  # Only preserve exact zeros
                    normalized_val = 0.0
                else:
                    normalized_val = (original_value - min_val) / (max_val - min_val)
                
                normalized_values[exp_name] = normalized_val
        else:
            # If all values are the same, assign 0.5 to all
            for i, exp_name in enumerate(experiment_names):
                original_value = metric_values[i]
                
                # Special handling for reversed_undirected_saturating_chebyshev:
                # If all values are the same and exactly 0, keep as 0
                if (metric_name == 'reversed_undirected_saturating_chebyshev' and 
                    original_value == 0.0):  # Only preserve exact zeros
                    normalized_values[exp_name] = 0.0
                else:
                    normalized_values[exp_name] = 0.5
    
    return normalized_values

def main():
    parser = argparse.ArgumentParser(description="Analyze patching metrics: filter, plot, and summarize")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--experiment_file", type=str,
                       help="Path to a JSON file with patch experiment definitions")
    parser.add_argument("--custom_thresholds", type=str,
                       help="Comma-separated list of custom thresholds in format 'metric:threshold', e.g., 'kl_divergence:0.1,undirected_saturating_chebyshev:0.6'")
    parser.add_argument("--max_experiments", type=int, default=30,
                       help="Maximum number of cumulative experiments to generate for circuit verification (default: 30)")
    
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
    analyze_metrics(args.agent_path, metrics_thresholds, max_experiments=args.max_experiments)

if __name__ == "__main__":
    main() 