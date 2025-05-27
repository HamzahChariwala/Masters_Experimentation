#!/usr/bin/env python3
"""
Script: plot_metric_distributions.py

Generates distribution plots for activation patching metrics.
Creates two separate plots:
1. Noising metrics: KL divergence and undirected saturating chebyshev
2. Denoising metrics: Reverse KL divergence and undirected saturating chebyshev

The distributions are overlapped with reduced opacity and use normalized values.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import argparse
from typing import Dict, List, Any, Tuple
import glob
import seaborn as sns

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
            
            # Check if we should use pre-computed normalized values
            if use_normalized and 'normalized_value' in metric_data:
                values.append(metric_data['normalized_value'])
            # Fall back to mean values if normalized values aren't available
            elif 'mean' in metric_data:
                values.append(metric_data['mean'])
    
    return values

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
    # Create bins from 0 to 1 with 0.05 increments
    bins = np.arange(0, 1.05, 0.05)
    hist, bin_edges = np.histogram(values, bins=bins)
    
    # Create output filename
    output_file = os.path.join(output_dir, f"{experiment_type}_{metric_name}_histogram.txt")
    
    # Write histogram data to file
    with open(output_file, 'w') as f:
        f.write(f"Histogram data for {metric_name} ({experiment_type})\n")
        f.write(f"Total values: {len(values)}\n")
        f.write(f"Min: {min(values):.6f}, Max: {max(values):.6f}, Mean: {np.mean(values):.6f}, Median: {np.median(values):.6f}\n")
        f.write(f"Number of zeros: {sum(1 for v in values if v == 0)}, Number of ones: {sum(1 for v in values if v == 1)}\n\n")
        f.write("Bin Range,Count,Percentage\n")
        
        for i in range(len(hist)):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i+1]
            percentage = (hist[i] / len(values)) * 100 if values else 0
            f.write(f"{bin_start:.2f}-{bin_end:.2f},{hist[i]},{percentage:.2f}%\n")
    
    print(f"Histogram data saved to {output_file}")
    return output_file

def plot_metric_distributions(
    agent_path: str,
    output_dir: str = None,
    noising_metrics: List[str] = None,
    denoising_metrics: List[str] = None,
    use_normalized: bool = True
) -> Tuple[str, str]:
    """
    Generate distribution plots for noising and denoising metrics.
    
    Args:
        agent_path: Path to the agent directory
        output_dir: Directory to save plots (defaults to agent's patching_results)
        noising_metrics: List of metrics to plot for noising (defaults to KL divergence and undirected chebyshev)
        denoising_metrics: List of metrics to plot for denoising (defaults to reverse KL and undirected chebyshev)
        use_normalized: Whether to use pre-computed normalized values (default: True)
        
    Returns:
        Tuple of paths to the generated plot files
    """
    # Set default metrics if not provided
    if noising_metrics is None:
        noising_metrics = ["kl_divergence", "undirected_saturating_chebyshev"]
    
    if denoising_metrics is None:
        denoising_metrics = ["reverse_kl_divergence", "undirected_saturating_chebyshev"]
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(agent_path, "patching_results")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths to summary files
    noising_summary_file = os.path.join(agent_path, "patching_results", "patching_summary_noising.json")
    denoising_summary_file = os.path.join(agent_path, "patching_results", "patching_summary_denoising.json")
    
    # Output file paths
    noising_plot_file = os.path.join(output_dir, "noising_metrics_distribution.png")
    denoising_plot_file = os.path.join(output_dir, "denoising_metrics_distribution.png")
    
    # Load summary data
    print(f"Loading summary data from {noising_summary_file} and {denoising_summary_file}")
    noising_data = load_summary_data(noising_summary_file)
    denoising_data = load_summary_data(denoising_summary_file)
    
    # Check if we have data
    if not noising_data:
        print(f"No noising summary data found at {noising_summary_file}")
        return None, None
    
    if not denoising_data:
        print(f"No denoising summary data found at {denoising_summary_file}")
        return None, None
    
    # Get the magma colormap with more distinct colors
    magma = plt.cm.magma
    
    # Generate colors for each metric
    noising_color_positions = np.linspace(0.1, 0.9, len(noising_metrics))
    noising_colors = [magma(pos) for pos in noising_color_positions]
    
    # Customize colors if specific metrics are present
    if "kl_divergence" in noising_metrics and "undirected_saturating_chebyshev" in noising_metrics:
        kl_index = noising_metrics.index("kl_divergence")
        cheb_index = noising_metrics.index("undirected_saturating_chebyshev")
        noising_colors[kl_index] = magma(0.2)  # Light purple for KL divergence
        noising_colors[cheb_index] = magma(0.55)  # Magenta for Chebyshev
        
    denoising_color_positions = np.linspace(0.1, 0.9, len(denoising_metrics))
    denoising_colors = [magma(pos) for pos in denoising_color_positions]
    
    # Customize colors if specific metrics are present
    if "reverse_kl_divergence" in denoising_metrics and "undirected_saturating_chebyshev" in denoising_metrics:
        kl_index = denoising_metrics.index("reverse_kl_divergence")
        cheb_index = denoising_metrics.index("undirected_saturating_chebyshev")
        denoising_colors[kl_index] = magma(0.3)  # Light purple for reverse KL divergence
        denoising_colors[cheb_index] = magma(0.55)  # Magenta for Chebyshev
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Process noising metrics
    print(f"Plotting noising metrics: {', '.join(noising_metrics)}")
    
    # Calculate rows and columns for subplot grid
    n_metrics = len(noising_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    # Create the figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    
    # Make axes a 2D array if it's 1D or a single axis
    if n_metrics == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Store value statistics for comparison
    value_stats = {}
    value_type = "pre-normalized" if use_normalized else "manually normalized"
    
    # Define bins with 0.05 increments from 0 to 1
    bins = np.arange(0, 1.05, 0.05)
    
    # Plot each metric in its own subplot
    for i, metric in enumerate(noising_metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Extract values using pre-computed normalized values if requested
        values = extract_metric_values(noising_data, metric, use_normalized)
        
        if not values:
            print(f"No values found for noising metric: {metric}")
            ax.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
            continue
        
        # Store original values for comparison
        orig_values = values.copy()
        
        # Only normalize if we're not using pre-computed normalized values
        if not use_normalized:
            values = normalize_values(values)
            value_type = "manually normalized"
        else:
            # Validate the normalized values to ensure they're in [0, 1]
            values = validate_normalized_values(values)
            value_type = "pre-normalized"
        
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
        
        # Save histogram data with 0.05 increments
        save_histogram_data(values, metric, "noising", output_dir)
        
        # Plot histogram instead of KDE
        sns.histplot(
            values, 
            bins=bins,
            color=noising_colors[i],
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
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
    plt.savefig(noising_plot_file, dpi=300)
    plt.close()
    
    # Process denoising metrics
    print(f"\nPlotting denoising metrics: {', '.join(denoising_metrics)}")
    
    # Calculate rows and columns for subplot grid
    n_metrics = len(denoising_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    # Create the figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    
    # Make axes a 2D array if it's 1D or a single axis
    if n_metrics == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each metric in its own subplot
    for i, metric in enumerate(denoising_metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Extract values using pre-computed normalized values if requested
        values = extract_metric_values(denoising_data, metric, use_normalized)
        
        if not values:
            print(f"No values found for denoising metric: {metric}")
            ax.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
            continue
        
        # Store original values for comparison
        orig_values = values.copy()
        
        # Only normalize if we're not using pre-computed normalized values
        if not use_normalized:
            values = normalize_values(values)
            value_type = "manually normalized"
        else:
            # Validate the normalized values to ensure they're in [0, 1]
            values = validate_normalized_values(values)
            value_type = "pre-normalized"
        
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
        
        # Save histogram data with 0.05 increments
        save_histogram_data(values, metric, "denoising", output_dir)
        
        # Plot histogram instead of KDE
        sns.histplot(
            values, 
            bins=bins,
            color=denoising_colors[i],
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
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
    fig.suptitle(f"Histogram of {value_type.capitalize()} Denoising Metrics", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the suptitle
    plt.savefig(denoising_plot_file, dpi=300)
    plt.close()
    
    print(f"\nPlots saved to {noising_plot_file} and {denoising_plot_file}")
    
    return noising_plot_file, denoising_plot_file

def main():
    parser = argparse.ArgumentParser(description="Generate distribution plots for patching metrics")
    parser.add_argument("--agent_path", type=str, required=True,
                      help="Path to the agent directory")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Directory to save plots (defaults to agent's patching_results)")
    parser.add_argument("--use_raw", action="store_false", dest="use_normalized",
                      help="Use raw values instead of pre-computed normalized values")
    parser.add_argument("--metrics", type=str, default=None,
                      help="Comma-separated list of metrics to plot (defaults to KL divergence and undirected saturating chebyshev)")
    args = parser.parse_args()
    
    # Parse metrics if provided
    noising_metrics = None
    denoising_metrics = None
    if args.metrics:
        metrics_list = args.metrics.split(',')
        noising_metrics = metrics_list
        denoising_metrics = metrics_list
    
    # Generate the plots
    noising_plot, denoising_plot = plot_metric_distributions(
        args.agent_path, 
        args.output_dir,
        noising_metrics=noising_metrics,
        denoising_metrics=denoising_metrics,
        use_normalized=args.use_normalized
    )
    
    if noising_plot and denoising_plot:
        print("Plot generation completed successfully.")
    else:
        print("Error generating plots. Check the logs for details.")

if __name__ == "__main__":
    main() 