#!/usr/bin/env python3
"""
Monotonic Metric Progression Plot Generator

Generates plots showing how metrics change as neurons are added to the monotonic coalition.
Reads data from summary.json files in each metric's directory.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Add the project directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from Neuron_Selection.PatchingTooling.patching_experiment import PatchingExperiment


def load_metric_progression(summary_file: Path) -> Tuple[List[str], List[float], Dict]:
    """
    Load metric progression data from summary file.
    
    Returns:
        Tuple of (coalition_neurons, coalition_scores, metadata)
    """
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    coalition_neurons = summary['results']['final_coalition_neurons']
    coalition_scores = summary['results']['coalition_scores']
    
    metadata = {
        'metric': summary['parameters']['metric'],
        'final_coalition_size': summary['results']['final_coalition_size'],
        'total_experiments': summary['results']['total_experiments_run'],
        'initial_score': summary['results']['score_progression']['initial_score'],
        'final_score': summary['results']['score_progression']['final_score'],
        'total_improvement': summary['results']['score_progression']['total_improvement'],
        'highest': summary['parameters']['highest']
    }
    
    return coalition_neurons, coalition_scores, metadata


def create_metric_progression_plot(metric_name: str, coalition_neurons: List[str], 
                                 coalition_scores: List[float], metadata: Dict, 
                                 output_dir: Path):
    """Create separate individual plots showing metric progression as neurons are added."""
    
    if not coalition_scores:
        print(f"No score data for metric {metric_name}")
        return
    
    # Colors from magma colormap
    line_color = plt.cm.magma(0.7)
    marker_color = plt.cm.magma(0.5)
    
    # Coalition sizes (0-based indexing: 0=baseline, 1=first neuron, etc.)
    coalition_sizes = list(range(len(coalition_scores)))
    
    # Create first plot: Raw metric scores (SEPARATE FIGURE)
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 8))
    
    ax1.plot(coalition_sizes, coalition_scores, 
             color=line_color, linewidth=3.5, marker='o', 
             markersize=6, markerfacecolor=marker_color, 
             markeredgecolor='white', markeredgewidth=1.5)
    
    ax1.set_xlabel('Coalition Size', fontsize=12)
    ax1.set_ylabel('Metric Score', fontsize=12)
    ax1.set_title(f'Monotonic Coalition Building - {metric_name}\nMetric Progression (Raw Scores)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add some key statistics as text
    improvement_direction = "Higher is better" if metadata['highest'] else "Lower is better"
    ax1.text(0.02, 0.98, f"Optimization: {improvement_direction}", 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set x-axis limits with some padding
    ax1.set_xlim(-0.5, len(coalition_scores) - 0.5)
    ax1.set_xticks(range(0, len(coalition_scores), max(1, len(coalition_scores) // 10)))
    
    # Save first plot
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_scores_file = output_dir / f"metric_progression_{metric_name}_raw_scores.png"
    plt.savefig(raw_scores_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {raw_scores_file}")
    
    # Create second plot: Improvement metrics (SEPARATE FIGURE)
    if len(coalition_scores) > 1:
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
        
        # Calculate improvements from baseline
        baseline_score = coalition_scores[0]
        percentage_improvements = [100 * (score - baseline_score) / abs(baseline_score) 
                                 if baseline_score != 0 else 0 
                                 for score in coalition_scores]
        
        # Plot percentage improvements
        ax2.plot(coalition_sizes, percentage_improvements, 
                color=line_color, linewidth=3.5, marker='s', 
                markersize=5, markerfacecolor=marker_color,
                markeredgecolor='white', markeredgewidth=1.5)
        
        ax2.set_xlabel('Coalition Size', fontsize=12)
        ax2.set_ylabel('Improvement from Baseline (%)', fontsize=12)
        ax2.set_title(f'Monotonic Coalition Building - {metric_name}\nPercentage Improvement from Baseline', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add zero line
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add statistics text
        final_improvement = percentage_improvements[-1]
        avg_improvement_per_neuron = final_improvement / (len(coalition_scores) - 1) if len(coalition_scores) > 1 else 0
        
        stats_text = (f"Final improvement: {final_improvement:.2f}%\n"
                     f"Avg per neuron: {avg_improvement_per_neuron:.2f}%\n"
                     f"Total experiments: {metadata['total_experiments']}")
        
        ax2.text(0.98, 0.02, stats_text, 
                transform=ax2.transAxes, fontsize=10, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Set x-axis limits with some padding
        ax2.set_xlim(-0.5, len(coalition_scores) - 0.5)
        ax2.set_xticks(range(0, len(coalition_scores), max(1, len(coalition_scores) // 10)))
        
        # Save second plot
        plt.tight_layout()
        improvement_file = output_dir / f"metric_progression_{metric_name}_improvements.png"
        plt.savefig(improvement_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {improvement_file}")


def create_comparative_progression_plot(metrics_data: Dict[str, Tuple], output_file: Path):
    """Create a comparative plot showing progression across multiple metrics."""
    
    if not metrics_data:
        return
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Colors for different metrics
    colors = plt.cm.magma(np.linspace(0.15, 0.85, len(metrics_data)))
    
    for i, (metric_name, (coalition_neurons, coalition_scores, metadata)) in enumerate(metrics_data.items()):
        if not coalition_scores:
            continue
            
        coalition_sizes = list(range(len(coalition_scores)))
        color = colors[i]
        
        # Top plot: Normalized scores (0-1 scale)
        if coalition_scores:
            min_score, max_score = min(coalition_scores), max(coalition_scores)
            if max_score != min_score:
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in coalition_scores]
            else:
                normalized_scores = [0.5] * len(coalition_scores)
            
            ax1.plot(coalition_sizes, normalized_scores, 
                    color=color, linewidth=3, marker='o', markersize=4,
                    label=metric_name.replace('_', ' ').title())
        
        # Bottom plot: Percentage improvements
        if len(coalition_scores) > 1:
            baseline_score = coalition_scores[0]
            percentage_improvements = [100 * (score - baseline_score) / abs(baseline_score) 
                                     if baseline_score != 0 else 0 
                                     for score in coalition_scores]
            
            ax2.plot(coalition_sizes, percentage_improvements, 
                    color=color, linewidth=3, marker='s', markersize=4,
                    label=metric_name.replace('_', ' ').title())
    
    # Configure top plot
    ax1.set_xlabel('Coalition Size', fontsize=12)
    ax1.set_ylabel('Normalized Score (0-1)', fontsize=12)
    ax1.set_title('Comparative Metric Progression - Normalized Scores', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Configure bottom plot
    ax2.set_xlabel('Coalition Size', fontsize=12)
    ax2.set_ylabel('Improvement from Baseline (%)', fontsize=12)
    ax2.set_title('Comparative Percentage Improvements', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Layout and save
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")


def create_combined_metric_plots(metrics_data: Dict[str, Tuple], output_dir: Path):
    """Create the three combined plots requested by the user, split into two graphs each."""
    
    if not metrics_data:
        return
    
    # Define the two groups of metrics
    group1_metrics = ['reversed_pearson_correlation', 'kl_divergence', 'top_logit_delta_magnitude']
    group2_metrics = ['confidence_margin_magnitude', 'undirected_saturating_chebyshev', 'reversed_undirected_saturating_chebyshev']
    
    # Filter metrics into groups
    group1_data = {k: v for k, v in metrics_data.items() if k in group1_metrics}
    group2_data = {k: v for k, v in metrics_data.items() if k in group2_metrics}
    
    # Use CONSISTENT metric ordering for color mapping across all plots
    monotonic_metric_order = ['reversed_pearson_correlation', 'kl_divergence', 'top_logit_delta_magnitude', 
                             'confidence_margin_magnitude', 'undirected_saturating_chebyshev', 'reversed_undirected_saturating_chebyshev']
    original_colors = plt.cm.magma(np.linspace(0.15, 0.85, len(monotonic_metric_order)))
    color_mapping = dict(zip(monotonic_metric_order, original_colors))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Progression of the metric normalized value - SIDE BY SIDE
    fig1, (ax1_left, ax1_right) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Group 1 on left
    for metric_name, (coalition_neurons, coalition_scores, metadata) in group1_data.items():
        if not coalition_scores:
            continue
            
        coalition_sizes = list(range(len(coalition_scores)))
        color = color_mapping[metric_name]
        
        # Normalize scores (0-1 scale)
        min_score, max_score = min(coalition_scores), max(coalition_scores)
        if max_score != min_score:
            normalized_scores = [(score - min_score) / (max_score - min_score) for score in coalition_scores]
        else:
            normalized_scores = [0.5] * len(coalition_scores)
        
        ax1_left.plot(coalition_sizes, normalized_scores, 
                color=color, linewidth=2.5, marker='o', markersize=3,
                label=metric_name.replace('_', ' ').title())
    
    # Group 2 on right
    for metric_name, (coalition_neurons, coalition_scores, metadata) in group2_data.items():
        if not coalition_scores:
            continue
            
        coalition_sizes = list(range(len(coalition_scores)))
        color = color_mapping[metric_name]
        
        # Normalize scores (0-1 scale)
        min_score, max_score = min(coalition_scores), max(coalition_scores)
        if max_score != min_score:
            normalized_scores = [(score - min_score) / (max_score - min_score) for score in coalition_scores]
        else:
            normalized_scores = [0.5] * len(coalition_scores)
        
        ax1_right.plot(coalition_sizes, normalized_scores, 
                color=color, linewidth=2.5, marker='o', markersize=3,
                label=metric_name.replace('_', ' ').title())
    
    # Configure both subplots
    for ax in [ax1_left, ax1_right]:
        ax.set_xlabel('Coalition Size', fontsize=10)
        ax.set_ylabel('Normalized Metric Value (0-1)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    
    # Synchronize y-axis bounds for normalized plot
    all_y_mins = [ax.get_ylim()[0] for ax in [ax1_left, ax1_right]]
    all_y_maxs = [ax.get_ylim()[1] for ax in [ax1_left, ax1_right]]
    y_min, y_max = min(all_y_mins), max(all_y_maxs)
    for ax in [ax1_left, ax1_right]:
        ax.set_ylim(y_min, y_max)
    
    ax1_left.set_title('Group 1: Pearson, KL Divergence, Top Logit', fontsize=11, fontweight='bold')
    ax1_right.set_title('Group 2: Confidence, Chebyshev Metrics', fontsize=11, fontweight='bold')
    
    # Combine legends and put horizontally at bottom
    handles1, labels1 = ax1_left.get_legend_handles_labels()
    handles2, labels2 = ax1_right.get_legend_handles_labels()
    fig1.legend(handles1 + handles2, labels1 + labels2, 
               loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=9)
    
    plt.suptitle('Metric Progression - Normalized Values with Coalition Growth', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot1_file = output_dir / "combined_metric_progression_normalized_split.png"
    plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot1_file}")
    
    # Plot 2: Percentage improvement from baseline - SIDE BY SIDE
    fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Group 1 on left
    for metric_name, (coalition_neurons, coalition_scores, metadata) in group1_data.items():
        if not coalition_scores or len(coalition_scores) <= 1:
            continue
            
        coalition_sizes = list(range(len(coalition_scores)))
        color = color_mapping[metric_name]
        
        # Calculate percentage improvements from baseline
        baseline_score = coalition_scores[0]
        percentage_improvements = [100 * (score - baseline_score) / abs(baseline_score) 
                                 if baseline_score != 0 else 0 
                                 for score in coalition_scores]
        
        ax2_left.plot(coalition_sizes, percentage_improvements, 
                color=color, linewidth=2.5, marker='s', markersize=3,
                label=metric_name.replace('_', ' ').title())
    
    # Group 2 on right
    for metric_name, (coalition_neurons, coalition_scores, metadata) in group2_data.items():
        if not coalition_scores or len(coalition_scores) <= 1:
            continue
            
        coalition_sizes = list(range(len(coalition_scores)))
        color = color_mapping[metric_name]
        
        # Calculate percentage improvements from baseline
        baseline_score = coalition_scores[0]
        percentage_improvements = [100 * (score - baseline_score) / abs(baseline_score) 
                                 if baseline_score != 0 else 0 
                                 for score in coalition_scores]
        
        ax2_right.plot(coalition_sizes, percentage_improvements, 
                color=color, linewidth=2.5, marker='s', markersize=3,
                label=metric_name.replace('_', ' ').title())
    
    # Configure both subplots
    for ax in [ax2_left, ax2_right]:
        ax.set_xlabel('Coalition Size', fontsize=10)
        ax.set_ylabel('Improvement from Baseline (%)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=9)
    
    # Synchronize y-axis bounds for baseline improvement plot
    all_y_mins = [ax.get_ylim()[0] for ax in [ax2_left, ax2_right]]
    all_y_maxs = [ax.get_ylim()[1] for ax in [ax2_left, ax2_right]]
    y_min, y_max = min(all_y_mins), max(all_y_maxs)
    for ax in [ax2_left, ax2_right]:
        ax.set_ylim(y_min, y_max)
    
    ax2_left.set_title('Group 1: Pearson, KL Divergence, Top Logit', fontsize=11, fontweight='bold')
    ax2_right.set_title('Group 2: Confidence, Chebyshev Metrics', fontsize=11, fontweight='bold')
    
    # Combine legends and put horizontally at bottom
    handles1, labels1 = ax2_left.get_legend_handles_labels()
    handles2, labels2 = ax2_right.get_legend_handles_labels()
    fig2.legend(handles1 + handles2, labels1 + labels2, 
               loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=9)
    
    plt.suptitle('Metric Progression - Percentage Improvement from Baseline', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot2_file = output_dir / "combined_metric_progression_baseline_improvement_split.png"
    plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot2_file}")
    
    # Plot 3: Percentage change from previous step - SIDE BY SIDE
    fig3, (ax3_left, ax3_right) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Group 1 on left
    for metric_name, (coalition_neurons, coalition_scores, metadata) in group1_data.items():
        if not coalition_scores or len(coalition_scores) <= 1:
            continue
            
        coalition_sizes = list(range(1, len(coalition_scores)))  # Start from 1 since no previous step for first point
        color = color_mapping[metric_name]
        
        # Calculate percentage changes from previous step
        step_percentage_changes = []
        for j in range(1, len(coalition_scores)):
            prev_score = coalition_scores[j-1]
            curr_score = coalition_scores[j]
            if prev_score != 0:
                change = 100 * (curr_score - prev_score) / abs(prev_score)
            else:
                change = 0
            step_percentage_changes.append(change)
        
        ax3_left.plot(coalition_sizes, step_percentage_changes, 
                color=color, linewidth=2.5, marker='^', markersize=3,
                label=metric_name.replace('_', ' ').title())
    
    # Group 2 on right
    for metric_name, (coalition_neurons, coalition_scores, metadata) in group2_data.items():
        if not coalition_scores or len(coalition_scores) <= 1:
            continue
            
        coalition_sizes = list(range(1, len(coalition_scores)))  # Start from 1 since no previous step for first point
        color = color_mapping[metric_name]
        
        # Calculate percentage changes from previous step
        step_percentage_changes = []
        for j in range(1, len(coalition_scores)):
            prev_score = coalition_scores[j-1]
            curr_score = coalition_scores[j]
            if prev_score != 0:
                change = 100 * (curr_score - prev_score) / abs(prev_score)
            else:
                change = 0
            step_percentage_changes.append(change)
        
        ax3_right.plot(coalition_sizes, step_percentage_changes, 
                color=color, linewidth=2.5, marker='^', markersize=3,
                label=metric_name.replace('_', ' ').title())
    
    # Configure both subplots
    for ax in [ax3_left, ax3_right]:
        ax.set_xlabel('Coalition Size', fontsize=10)
        ax.set_ylabel('Change from Previous Step (%)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=9)
    
    # Synchronize y-axis bounds for step changes plot
    all_y_mins = [ax.get_ylim()[0] for ax in [ax3_left, ax3_right]]
    all_y_maxs = [ax.get_ylim()[1] for ax in [ax3_left, ax3_right]]
    y_min, y_max = min(all_y_mins), max(all_y_maxs)
    for ax in [ax3_left, ax3_right]:
        ax.set_ylim(y_min, y_max)
    
    ax3_left.set_title('Group 1: Pearson, KL Divergence, Top Logit', fontsize=11, fontweight='bold')
    ax3_right.set_title('Group 2: Confidence, Chebyshev Metrics', fontsize=11, fontweight='bold')
    
    # Combine legends and put horizontally at bottom
    handles1, labels1 = ax3_left.get_legend_handles_labels()
    handles2, labels2 = ax3_right.get_legend_handles_labels()
    fig3.legend(handles1 + handles2, labels1 + labels2, 
               loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize=9)
    
    plt.suptitle('Metric Progression - Step-by-Step Percentage Change', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot3_file = output_dir / "combined_metric_progression_step_changes_split.png"
    plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot3_file}")


def process_metric_progression(agent_path: Path, metric: str):
    """Process and create progression plot for a single metric."""
    print(f"\nProcessing Metric Progression - {metric}")
    
    # Load summary data
    summary_file = agent_path / "circuit_verification" / "monotonic" / metric / "summary.json"
    
    if not summary_file.exists():
        print(f"No summary file found for metric {metric}")
        return None
    
    coalition_neurons, coalition_scores, metadata = load_metric_progression(summary_file)
    
    # Create output directory
    output_dir = agent_path / "circuit_verification" / "monotonic" / metric / "metric_progression_plots"
    
    # Create the separate plots
    create_metric_progression_plot(metric, coalition_neurons, coalition_scores, metadata, output_dir)
    
    return (coalition_neurons, coalition_scores, metadata)


def load_descending_progression(agent_path: Path, metric: str) -> Tuple[List[float], Dict]:
    """Load metric progression data from descending circuit results."""
    
    # Load the experiment progression from the experiments file
    experiments_file = agent_path / "circuit_verification" / "descending" / "experiments" / f"experiments_{metric}.json"
    
    if not experiments_file.exists():
        print(f"Warning: No experiments file found for descending {metric}")
        return [], {}
    
    with open(experiments_file, 'r') as f:
        experiments = json.load(f)
    
    if not experiments:
        print(f"Warning: No experiments data found for descending {metric}")
        return [], {}
    
    # For now, since we don't have actual metric scores for each experiment step,
    # we return empty data to indicate that descending progression isn't available
    # In practice, this would need to be implemented based on how descending results are calculated
    print(f"Note: Descending coalition progression for {metric} requires metric calculation implementation")
    
    return [], {}


def create_descending_progression_plot(metric_name: str, metric_scores: List[float], 
                                     metadata: Dict, output_file: Path):
    """Create plots showing metric progression for descending circuit analysis."""
    
    if not metric_scores:
        print(f"No descending score data available for metric {metric_name}")
        return
    
    # Colors from magma colormap
    line_color = plt.cm.magma(0.3)  # Different color from monotonic
    marker_color = plt.cm.magma(0.1)
    
    # Experiment indices
    experiment_indices = list(range(len(metric_scores)))
    
    # Create plot: Raw metric scores
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    ax.plot(experiment_indices, metric_scores, 
             color=line_color, linewidth=3.5, marker='o', 
             markersize=6, markerfacecolor=marker_color, 
             markeredgecolor='white', markeredgewidth=1.5)
    
    ax.set_xlabel('Experiment Index', fontsize=12)
    ax.set_ylabel('Metric Score', fontsize=12)
    ax.set_title(f'Descending Circuit Analysis - {metric_name}\nMetric Progression', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits with some padding
    ax.set_xlim(-0.5, len(metric_scores) - 0.5)
    ax.set_xticks(range(0, len(metric_scores), max(1, len(metric_scores) // 10)))
    
    # Save plot
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def create_descending_combined_plots(metrics_data: Dict[str, Tuple], output_dir: Path):
    """Create the same three combined plots for descending data."""
    
    if not metrics_data:
        return
    
    # Define the two groups of metrics (same as monotonic)
    group1_metrics = ['reversed_pearson_correlation', 'kl_divergence', 'top_logit_delta_magnitude']
    group2_metrics = ['confidence_margin_magnitude', 'undirected_saturating_chebyshev', 'reversed_undirected_saturating_chebyshev']
    
    # Filter metrics into groups
    group1_data = {k: v for k, v in metrics_data.items() if k in group1_metrics}
    group2_data = {k: v for k, v in metrics_data.items() if k in group2_metrics}
    
    # EXACT same colors as monotonic - use the same metric order as monotonic for consistency
    monotonic_metric_order = ['reversed_pearson_correlation', 'kl_divergence', 'top_logit_delta_magnitude', 
                             'confidence_margin_magnitude', 'undirected_saturating_chebyshev', 'reversed_undirected_saturating_chebyshev']
    original_colors = plt.cm.magma(np.linspace(0.15, 0.85, len(monotonic_metric_order)))
    color_mapping = dict(zip(monotonic_metric_order, original_colors))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Since we don't have real descending progression data yet, 
    # we'll skip generating the plots for now
    print("  Skipping descending combined plots - need metric calculation implementation")
    return


def process_descending_progression(agent_path: Path, metric: str):
    """Process and create progression plot for descending analysis."""
    print(f"\nProcessing Descending Progression - {metric}")
    
    # Load descending data
    metric_scores, metadata = load_descending_progression(agent_path, metric)
    
    if not metric_scores:
        print(f"No descending progression data found for metric {metric}")
        return None
    
    # Create output directory
    output_dir = agent_path / "circuit_verification" / "descending" / "metric_progression_plots"
    output_file = output_dir / f"descending_progression_{metric}.png"
    
    # Create the plot
    create_descending_progression_plot(metric, metric_scores, metadata, output_file)
    
    return (metric_scores, metadata)


def load_descending_coalition_data(agent_path: Path, metric: str, target_environments: List[str]) -> Dict[str, List[List[float]]]:
    """Load logit data for descending coalition progression for specific environments."""
    
    # Load the experiments file to get coalition progression
    experiments_file = agent_path / "circuit_verification" / "descending" / "experiments" / f"experiments_{metric}.json"
    
    if not experiments_file.exists():
        print(f"Warning: No experiments file found for descending {metric}")
        return {}
    
    with open(experiments_file, 'r') as f:
        experiments = json.load(f)
    
    if not experiments:
        print(f"Warning: No experiments data found for descending {metric}")
        return {}
    
    # Initialize patching experiment (using import from top of file)
    experiment = PatchingExperiment(str(agent_path))
    
    # Get input examples data
    clean_inputs_file = agent_path / "activation_inputs" / "clean_inputs.json"
    if not clean_inputs_file.exists():
        print(f"Warning: Could not find clean inputs at {clean_inputs_file}")
        return {}
    
    with open(clean_inputs_file, 'r') as f:
        all_input_examples = json.load(f)
    
    # Filter to target environments
    input_examples = {k: v for k, v in all_input_examples.items() if k in target_environments}
    
    logit_data = {env: [] for env in target_environments}
    
    print(f"  Running coalition progression experiments for {len(experiments)} coalition sizes")
    
    # Process each coalition step
    for coalition_idx, coalition_config in enumerate(experiments):
        print(f"    Coalition step {coalition_idx + 1}/{len(experiments)}")
        
        try:
            # Run patching experiment for this coalition configuration
            noising_results = experiment.run_patching_experiment(
                target_input_file="clean_inputs.json",
                source_activation_file="corrupted_activations.npz",
                patch_spec=coalition_config,
                input_ids=list(input_examples.keys())
            )
            
            # Extract logits for each target environment
            for env_name in target_environments:
                if env_name in noising_results:
                    result = noising_results[env_name]
                    
                    if coalition_idx == 0:
                        # For first coalition step, use baseline_output
                        if 'baseline_output' in result and result['baseline_output']:
                            logits = result['baseline_output'][0]
                            logit_data[env_name].append(logits)
                        else:
                            print(f"    Warning: No baseline output for {env_name}")
                            logit_data[env_name].append([0.2] * 5)
                    else:
                        # For subsequent steps, use patched_output
                        if 'patched_output' in result and result['patched_output']:
                            logits = result['patched_output'][0]
                            logit_data[env_name].append(logits)
                        else:
                            print(f"    Warning: No patched output for {env_name} at coalition step {coalition_idx}")
                            # Use previous logits if available
                            if logit_data[env_name]:
                                logit_data[env_name].append(logit_data[env_name][-1])
                            else:
                                logit_data[env_name].append([0.2] * 5)
                else:
                    print(f"    Warning: No results for {env_name} at coalition step {coalition_idx}")
                    # Use previous logits if available
                    if logit_data[env_name]:
                        logit_data[env_name].append(logit_data[env_name][-1])
                    else:
                        logit_data[env_name].append([0.2] * 5)
        
        except Exception as e:
            print(f"    Error running experiment for coalition step {coalition_idx}: {e}")
            # For error cases, use previous logits or fallback
            for env_name in target_environments:
                if logit_data[env_name]:
                    logit_data[env_name].append(logit_data[env_name][-1])
                else:
                    logit_data[env_name].append([0.2] * 5)
    
    # Verify data structure
    for env_name in target_environments:
        if env_name in logit_data:
            print(f"  Loaded {len(logit_data[env_name])} logit vectors for {env_name}")
    
    return logit_data


def create_descending_environment_plot(environment_name: str, logits_sequence: List[List[float]], 
                                     output_file: Path, metric: str):
    """Create a plot for a single environment showing descending coalition logit progression."""
    
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
    formatted_name = environment_name.replace("_", "-")
    ax.set_title(f'Descending Coalition - {metric}\n{formatted_name} - Logit Progression', 
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


def process_descending_individual_environments(agent_path: Path, metric: str):
    """Process and create individual environment plots for descending coalitions."""
    print(f"\nProcessing Descending Individual Environments - {metric}")
    
    # Target environments as requested
    target_patterns = ["81102-2,3,0", "81105-1,3,0"]
    
    # Find environments that contain the requested patterns
    clean_inputs_file = agent_path / "activation_inputs" / "clean_inputs.json"
    if not clean_inputs_file.exists():
        print(f"Warning: Could not find clean inputs at {clean_inputs_file}")
        return
    
    with open(clean_inputs_file, 'r') as f:
        all_input_examples = json.load(f)
    
    # Find matching environments
    matching_environments = []
    for env_key in all_input_examples.keys():
        for pattern in target_patterns:
            if pattern in env_key:
                matching_environments.append(env_key)
                break
    
    if not matching_environments:
        print(f"Warning: No environments found matching patterns {target_patterns}")
        return
    
    print(f"  Found {len(matching_environments)} matching environments: {matching_environments}")
    
    # Load logit data showing coalition progression
    logit_data = load_descending_coalition_data(agent_path, metric, matching_environments)
    
    if not logit_data:
        print(f"No logit data found for descending analysis of {metric}")
        return
    
    # Create output directory
    output_dir = agent_path / "circuit_verification" / "descending" / metric / "individual_env_plots"
    
    # Create individual plots for each environment
    for environment in matching_environments:
        if environment in logit_data and logit_data[environment]:
            safe_env_name = environment.replace("-", "_").replace(",", "_")
            output_file = output_dir / f"environment_{safe_env_name}_coalition_progression.png"
            
            create_descending_environment_plot(
                environment, 
                logit_data[environment],
                output_file,
                metric
            )


def main():
    parser = argparse.ArgumentParser(description="Generate metric progression plots for coalition building")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--metric", type=str, help="Specific metric to process (default: all)")
    parser.add_argument("--analysis_type", type=str, choices=['monotonic', 'descending', 'both'], 
                       default='monotonic', help="Type of analysis to process")
    parser.add_argument("--comparative", action="store_true", 
                       help="Generate comparative plot across all metrics (monotonic only)")
    parser.add_argument("--individual_env_plots", action="store_true",
                       help="Generate individual environment plots for descending coalitions")
    
    args = parser.parse_args()
    
    agent_path = Path(args.agent_path)
    
    if not agent_path.exists():
        print(f"Agent path does not exist: {agent_path}")
        return
    
    print(f"Creating metric progression plots for: {agent_path}")
    
    # Process monotonic analysis
    if args.analysis_type in ['monotonic', 'both']:
        print(f"\n=== PROCESSING MONOTONIC COALITIONS ===")
        
        # Find available monotonic metrics
        monotonic_dir = agent_path / "circuit_verification" / "monotonic"
        
        if not monotonic_dir.exists():
            print(f"Monotonic directory not found: {monotonic_dir}")
        else:
            available_metrics = []
            for metric_dir in monotonic_dir.iterdir():
                if metric_dir.is_dir() and (metric_dir / "summary.json").exists():
                    available_metrics.append(metric_dir.name)
            
            if not available_metrics:
                print("No monotonic metrics with summary files found!")
            else:
                # Process metrics
                if args.metric:
                    if args.metric in available_metrics:
                        process_metric_progression(agent_path, args.metric)
                    else:
                        print(f"Metric {args.metric} not found. Available: {available_metrics}")
                else:
                    # Process all metrics individually
                    metrics_data = {}
                    for metric in available_metrics:
                        result = process_metric_progression(agent_path, metric)
                        if result:
                            metrics_data[metric] = result
                    
                    # Filter out reverse_kl_divergence as requested (except reverse KLD)
                    if 'reverse_kl_divergence' in metrics_data:
                        print("  Filtering out reverse_kl_divergence as requested")
                        del metrics_data['reverse_kl_divergence']
                    
                    # Generate combined plots automatically when processing all metrics
                    if len(metrics_data) > 1:
                        print(f"\nCreating combined metric plots for {len(metrics_data)} metrics")
                        combined_output_dir = agent_path / "circuit_verification" / "monotonic" / "combined_plots"
                        create_combined_metric_plots(metrics_data, combined_output_dir)
                    
                    # Create comparative plot ONLY if explicitly requested
                    if args.comparative and len(metrics_data) > 1:
                        print(f"\nCreating additional comparative plot for {len(metrics_data)} metrics")
                        comparative_output = agent_path / "circuit_verification" / "monotonic" / "comparative_metric_progression.png"
                        create_comparative_progression_plot(metrics_data, comparative_output)
    
    # Process descending analysis
    if args.analysis_type in ['descending', 'both']:
        print(f"\n=== PROCESSING DESCENDING COALITIONS ===")
        
        # Check if individual environment plots are requested
        if args.individual_env_plots:
            print(f"\nProcessing Descending Individual Environment Plots")
            
            # Standard metrics for descending analysis
            standard_metrics = [
                'kl_divergence',
                'reverse_kl_divergence', 
                'confidence_margin_magnitude',
                'undirected_saturating_chebyshev',
                'reversed_pearson_correlation',
                'reversed_undirected_saturating_chebyshev',
                'top_logit_delta_magnitude'
            ]
            
            if args.metric:
                if args.metric in standard_metrics:
                    process_descending_individual_environments(agent_path, args.metric)
                else:
                    print(f"Metric {args.metric} not supported for descending individual environment plots. Supported: {standard_metrics}")
            else:
                # Process all standard metrics for descending individual environments
                for metric in standard_metrics:
                    try:
                        process_descending_individual_environments(agent_path, metric)
                    except Exception as e:
                        print(f"Error processing descending individual environments for {metric}: {e}")
                        continue
        else:
            # Original descending progression logic (currently disabled)
            descending_dir = agent_path / "circuit_verification" / "descending"
            
            if not descending_dir.exists():
                print(f"Descending directory not found: {descending_dir}")
            else:
                # Standard metrics that would be used in descending analysis
                standard_metrics = [
                    'kl_divergence',
                    'reverse_kl_divergence', 
                    'confidence_margin_magnitude',
                    'undirected_saturating_chebyshev',
                    'reversed_pearson_correlation',
                    'reversed_undirected_saturating_chebyshev',
                    'top_logit_delta_magnitude'
                ]
                
                if args.metric:
                    if args.metric in standard_metrics:
                        process_descending_progression(agent_path, args.metric)
                    else:
                        print(f"Metric {args.metric} not supported for descending analysis. Supported: {standard_metrics}")
                else:
                    # Process all standard metrics for descending
                    metrics_data = {}
                    for metric in standard_metrics:
                        result = process_descending_progression(agent_path, metric)
                        if result:
                            metrics_data[metric] = result
                    
                    # Filter out reverse_kl_divergence for descending too
                    if 'reverse_kl_divergence' in metrics_data:
                        print("  Filtering out reverse_kl_divergence for descending as requested")
                        del metrics_data['reverse_kl_divergence']
                    
                    # Generate combined plots for descending (currently disabled)
                    if len(metrics_data) > 1:
                        print(f"\nCreating combined descending metric plots for {len(metrics_data)} metrics")
                        combined_output_dir = agent_path / "circuit_verification" / "descending" / "combined_plots"
                        create_descending_combined_plots(metrics_data, combined_output_dir)
    
    print("\nMetric progression plot generation complete!")
    print("NOTE: Each metric now has completely separate individual plots:")
    print("  - Raw scores plot")
    print("  - Improvements plot") 
    print("  - No combined/overlapping plots unless --comparative is specified")
    if args.individual_env_plots:
        print("  - Individual environment plots for descending coalitions")


if __name__ == "__main__":
    main() 