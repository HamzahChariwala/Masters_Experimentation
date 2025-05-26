#!/usr/bin/env python3
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# Paths to result directories
noising_dir = "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6/patching_results/noising"
denoising_dir = "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6/patching_results/denoising"

# Function to extract metrics from experiment files
def extract_metrics_from_files(directory_path, metrics_list):
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    print(f"Found {len(json_files)} experiment files in {directory_path}")
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return [], {}
    
    # Initialize data structures
    experiments = []
    metrics_data = {}
    for metric in metrics_list:
        metrics_data[metric] = []
    
    # Track number of non-zero metrics found
    non_zero_counts = {metric: 0 for metric in metrics_list}
    
    # Process each JSON file (which contains multiple experiments for various inputs)
    for json_file in json_files:
        print(f"Processing file: {os.path.basename(json_file)}")
        with open(json_file, 'r') as f:
            file_data = json.load(f)
        
        # Each file contains multiple experiments
        for exp_name, exp_data in file_data.items():
            if not exp_name.startswith("exp_"):
                continue
            
            # Extract experiment details
            parts = exp_name.split('_')
            if len(parts) >= 3:
                neuron_name = '_'.join(parts[2:])
                exp_id = f"{parts[1]}_{neuron_name}"
                
                # Check if we've already processed this experiment
                if exp_id in experiments:
                    continue
                
                experiments.append(exp_id)
                
                # Extract metrics for this experiment from the patch_analysis field
                patch_analysis = exp_data.get("patch_analysis", {})
                
                for metric in metrics_list:
                    value = 0.0
                    
                    if metric in patch_analysis:
                        if metric == "chebyshev_ratio":
                            if isinstance(patch_analysis[metric], dict) and "ratio" in patch_analysis[metric]:
                                value = patch_analysis[metric]["ratio"]
                            else:
                                value = patch_analysis[metric]
                        elif metric == "confidence_margin_change":
                            if isinstance(patch_analysis[metric], dict) and "margin_change" in patch_analysis[metric]:
                                value = patch_analysis[metric]["margin_change"]
                            else:
                                value = patch_analysis[metric]
                        elif metric == "pearson_correlation":
                            if isinstance(patch_analysis[metric], dict) and "correlation" in patch_analysis[metric]:
                                value = patch_analysis[metric]["correlation"]
                            else:
                                value = patch_analysis[metric]
                        else:
                            # For kl_divergence and reverse_kl_divergence
                            value = patch_analysis[metric]
                    
                    metrics_data[metric].append(value)
                    
                    # Count non-zero values
                    if abs(value) > 1e-6:
                        non_zero_counts[metric] += 1
    
    # Debug: Print information about extracted data
    print(f"Extracted {len(experiments)} unique experiments")
    if experiments:
        print(f"First experiment: {experiments[0]}")
    
    for metric in metrics_list:
        print(f"Metric {metric}: {len(metrics_data[metric])} values, {non_zero_counts[metric]} non-zero")
        non_zero_values = [v for v in metrics_data[metric] if abs(v) > 1e-6]
        if non_zero_values:
            print(f"  First few non-zero values: {non_zero_values[:5]}")
            print(f"  Range: {min(non_zero_values)} to {max(non_zero_values)}")
    
    return experiments, metrics_data

# Function to create a grouped bar plot
def create_plot(experiments, metrics_data, metrics_list, title, output_path):
    # Debug: Check data before processing
    print(f"Creating plot: {title}")
    print(f"Number of experiments: {len(experiments)}")
    
    # Sort experiments by layer then by number
    sorted_indices = sorted(range(len(experiments)), 
                           key=lambda i: (experiments[i].split('_', 1)[1], int(experiments[i].split('_')[0])))
    
    sorted_experiments = [experiments[i] for i in sorted_indices]
    
    # Create the sorted metrics data
    sorted_metrics = {}
    for metric in metrics_list:
        sorted_metrics[metric] = [metrics_data[metric][i] for i in sorted_indices]
    
    # Group by layer
    layers = []
    current_layer = None
    layer_indices = []
    for i, exp in enumerate(sorted_experiments):
        layer = exp.split('_', 1)[1]
        if layer != current_layer:
            current_layer = layer
            layers.append(layer)
            layer_indices.append(i)
    layer_indices.append(len(sorted_experiments))
    
    # Debug: Check layer information
    print(f"Found layers: {layers}")
    print(f"Layer indices: {layer_indices}")
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Number of metrics and experiments
    n_metrics = len(metrics_list)
    n_experiments = len(sorted_experiments)
    
    # Set up the x positions
    bar_width = 0.8 / n_metrics
    x = np.arange(n_experiments)
    
    # Get the magma colormap
    magma = plt.cm.magma
    colors = [magma(i / (n_metrics - 0.5)) for i in range(n_metrics)]
    
    # Normalize metrics for better visualization
    normalized_metrics = {}
    for metric in metrics_list:
        values = sorted_metrics[metric]
        non_zero_values = [v for v in values if abs(v) > 1e-6]
        if non_zero_values:
            print(f"Normalizing {metric} with values ranging from {min(non_zero_values)} to {max(non_zero_values)}")
        else:
            print(f"Normalizing {metric} - all values are zero")
        
        if metric == "pearson_correlation":
            # For Pearson correlation, most values are close to 1
            # Subtract 0.99 and scale by 100 to make differences visible
            adjusted = []
            for v in values:
                if v > 0.99:  # Only adjust values close to 1
                    adjusted.append((v - 0.99) * 100)
                else:
                    adjusted.append(v)
            normalized_metrics[metric] = adjusted
            print(f"  Pearson correlation adjusted: Subtracted 0.99 and scaled by 100 for values > 0.99")
        elif metric == "chebyshev_ratio":
            # For chebyshev ratio, we want to preserve sign but limit extreme values
            normalized = []
            for v in values:
                if abs(v) > 3:
                    # Cap at +/- 3 and add a note about extreme values
                    normalized.append(3.0 * (1 if v > 0 else -1))
                else:
                    normalized.append(v)
            normalized_metrics[metric] = normalized
            
            # Find top 5 most extreme neurons (both positive and negative)
            extreme_indices = sorted(range(len(values)), key=lambda i: -abs(values[i]))[:5]
            print(f"  Top 5 extreme chebyshev_ratio neurons:")
            for i in extreme_indices:
                print(f"    Neuron {sorted_experiments[i]}: {values[i]}")
        else:
            # For other metrics, normalize to [-1, 1] range if non-zero values exist
            max_abs = max([abs(v) for v in values]) if non_zero_values else 1.0
            if max_abs > 0:
                normalized_metrics[metric] = [v / max_abs for v in values]
            else:
                normalized_metrics[metric] = values
        
        # Debug: Check normalized values
        non_zero = sum(1 for v in normalized_metrics[metric] if abs(v) > 1e-6)
        print(f"  After normalization: {non_zero} non-zero values")
        if non_zero > 0:
            print(f"  First few non-zero values: {[v for v in normalized_metrics[metric][:20] if abs(v) > 1e-6][:5]}")
    
    # Plot each metric
    for i, metric in enumerate(metrics_list):
        ax.bar(x + i * bar_width - 0.4 + bar_width/2, normalized_metrics[metric], 
               width=bar_width, label=metric, color=colors[i], alpha=0.8)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add layer separators and labels
    for i in range(len(layers)):
        if i < len(layers) - 1:
            ax.axvline(x=layer_indices[i+1] - 0.5, color='gray', linestyle='--', alpha=0.5)
        
        midpoint = (layer_indices[i] + layer_indices[i+1]) / 2
        ax.text(midpoint, ax.get_ylim()[1] * 1.05, layers[i], 
                horizontalalignment='center', fontsize=12, weight='bold')
    
    # Set labels and title
    ax.set_xticks(x)
    ax.set_xticklabels([exp.split('_')[0] for exp in sorted_experiments], rotation=90, fontsize=8)
    ax.set_xlabel('Neuron Number')
    ax.set_ylabel('Normalized Metric Value')
    ax.set_title(title, fontsize=16)
    
    # Set y-axis limits explicitly
    ax.set_ylim(-1.2, 1.2)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()

# Define metrics for each plot
noising_metrics = ["kl_divergence", "chebyshev_ratio", "confidence_margin_change", "pearson_correlation"]
denoising_metrics = ["reverse_kl_divergence", "chebyshev_ratio", "confidence_margin_change", "pearson_correlation"]

# Extract data from individual experiment files
print("Extracting noising metrics...")
noising_experiments, noising_metrics_data = extract_metrics_from_files(noising_dir, noising_metrics)
print("\nExtracting denoising metrics...")
denoising_experiments, denoising_metrics_data = extract_metrics_from_files(denoising_dir, denoising_metrics)

# Create plots
if noising_experiments:
    create_plot(noising_experiments, noising_metrics_data, noising_metrics, 
               "Noising Metrics for biased-v1 Agent Neurons", "noising_metrics_plot.png")
else:
    print("No noising data to plot")

if denoising_experiments:
    create_plot(denoising_experiments, denoising_metrics_data, denoising_metrics, 
               "Denoising Metrics for biased-v1 Agent Neurons", "denoising_metrics_plot.png")
else:
    print("No denoising data to plot")

print("Plots created: noising_metrics_plot.png and denoising_metrics_plot.png")