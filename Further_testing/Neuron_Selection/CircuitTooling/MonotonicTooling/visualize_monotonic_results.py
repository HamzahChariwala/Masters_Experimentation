#!/usr/bin/env python3
"""
Visualization script for Monotonic Coalition Builder results.

This script creates visualizations showing:
1. Coalition score progression for each metric
2. Comparison of final coalition scores across metrics  
3. Improvement rates across iterations for each metric
4. Coalition composition analysis across metrics

The script reads from the monotonic coalition summary files and creates comprehensive plots.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np

# Add the Neuron_Selection directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
neuron_selection_dir = os.path.abspath(os.path.join(script_dir, "../.."))
project_root = os.path.abspath(os.path.join(neuron_selection_dir, ".."))

# Add both directories to path if they're not already there
for path in [neuron_selection_dir, project_root]:
    if not path in sys.path:
        sys.path.insert(0, path)

def load_summary_data(agent_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load summary data for all available metrics from monotonic coalition results.
    
    Args:
        agent_path: Path to the agent directory
        
    Returns:
        Dictionary mapping metric names to their summary data
    """
    agent_path = Path(agent_path)
    monotonic_dir = agent_path / "circuit_verification" / "monotonic"
    
    if not monotonic_dir.exists():
        print(f"Monotonic results directory not found: {monotonic_dir}")
        return {}
    
    summaries = {}
    
    for metric_dir in monotonic_dir.iterdir():
        if metric_dir.is_dir():
            summary_file = metric_dir / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summaries[metric_dir.name] = json.load(f)
                    print(f"Loaded summary for metric: {metric_dir.name}")
                except Exception as e:
                    print(f"Error loading summary for {metric_dir.name}: {e}")
            else:
                print(f"No summary.json found for metric: {metric_dir.name}")
    
    return summaries

def create_score_progression_plots(summaries: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """
    Create plots showing coalition score progression for each metric.
    
    Args:
        summaries: Dictionary of metric summaries
        output_dir: Directory to save plots
    """
    if not summaries:
        print("No summary data available for score progression plots")
        return
    
    # Create individual plots for each metric
    for metric_name, summary in summaries.items():
        if 'results' not in summary or 'coalition_scores' not in summary['results']:
            print(f"No coalition scores found for metric: {metric_name}")
            continue
        
        scores = summary['results']['coalition_scores']
        iterations = list(range(len(scores)))
        
        plt.figure(figsize=(12, 6))
        
        # Plot score progression
        plt.plot(iterations, scores, 'o-', linewidth=2, markersize=4, alpha=0.8)
        
        # Add baseline and final score annotations
        initial_score = scores[0] if scores else 0
        final_score = scores[-1] if scores else 0
        improvement = final_score - initial_score
        
        plt.axhline(y=initial_score, color='red', linestyle='--', alpha=0.7, label=f'Initial: {initial_score:.6f}')
        plt.axhline(y=final_score, color='green', linestyle='--', alpha=0.7, label=f'Final: {final_score:.6f}')
        
        plt.xlabel('Coalition Position', fontsize=12)
        plt.ylabel(f'{metric_name.replace("_", " ").title()} Score', fontsize=12)
        plt.title(f'Coalition Score Progression: {metric_name.replace("_", " ").title()}\n'
                 f'Improvement: {improvement:.6f} (+{improvement/initial_score*100:.1f}%)' if initial_score > 0 else 
                 f'Improvement: {improvement:.6f}', fontsize=14)
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        safe_metric_name = metric_name.replace("/", "_").replace("\\", "_")
        output_file = output_dir / f"monotonic_progression_{safe_metric_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved progression plot for {metric_name}: {output_file}")

def create_comparison_plots(summaries: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """
    Create comparison plots across all metrics.
    
    Args:
        summaries: Dictionary of metric summaries
        output_dir: Directory to save plots
    """
    if not summaries:
        print("No summary data available for comparison plots")
        return
    
    # Extract data for comparison
    metrics = []
    initial_scores = []
    final_scores = []
    improvements = []
    improvement_percentages = []
    total_experiments = []
    
    for metric_name, summary in summaries.items():
        if 'results' not in summary:
            continue
            
        results = summary['results']
        if 'score_progression' in results:
            metrics.append(metric_name.replace("_", " ").title())
            initial_score = results['score_progression']['initial_score']
            final_score = results['score_progression']['final_score']
            improvement = results['score_progression']['total_improvement']
            
            initial_scores.append(initial_score)
            final_scores.append(final_score)
            improvements.append(improvement)
            
            # Calculate improvement as percentage of initial score
            if initial_score > 0:
                improvement_percentage = (improvement / initial_score) * 100
            else:
                improvement_percentage = 0.0 if improvement == 0 else float('inf')
            improvement_percentages.append(improvement_percentage)
            
            total_experiments.append(results.get('total_experiments_run', 0))
    
    if not metrics:
        print("No valid comparison data found")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Initial vs Final Scores
    ax1 = axes[0, 0]
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, initial_scores, width, label='Initial Score', alpha=0.7, color='lightcoral')
    bars2 = ax1.bar(x_pos + width/2, final_scores, width, label='Final Score', alpha=0.7, color='lightblue')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Initial vs Final Coalition Scores')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Percentage Improvement
    ax2 = axes[0, 1]
    bars = ax2.bar(metrics, improvement_percentages, alpha=0.7, color='lightgreen')
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Score Improvement as % of Initial Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, pct in zip(bars, improvement_percentages):
        height = bar.get_height()
        if height != float('inf'):
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., 0.1,
                    'inf%', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Experiments Run
    ax3 = axes[1, 0]
    bars = ax3.bar(metrics, total_experiments, alpha=0.7, color='lightsalmon')
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Total Experiments')
    ax3.set_title('Total Experiments Run by Metric')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Efficiency (Percentage Improvement per Experiment)
    ax4 = axes[1, 1]
    efficiency = []
    for pct, exp in zip(improvement_percentages, total_experiments):
        if exp > 0 and pct != float('inf'):
            efficiency.append(pct / exp)
        else:
            efficiency.append(0.0)
    
    bars = ax4.bar(metrics, efficiency, alpha=0.7, color='lightsteelblue')
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('% Improvement per Experiment')
    ax4.set_title('Coalition Building Efficiency')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save comparison plot
    output_file = output_dir / "monotonic_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {output_file}")

def create_coalition_composition_analysis(summaries: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """
    Create analysis plots showing coalition composition and neuron selection patterns.
    
    Args:
        summaries: Dictionary of metric summaries
        output_dir: Directory to save plots
    """
    if not summaries:
        print("No summary data available for composition analysis")
        return
    
    # Analyze coalition composition across metrics
    metrics = []
    coalition_sizes = []
    layer_distributions = {}
    
    for metric_name, summary in summaries.items():
        if 'results' not in summary or 'final_coalition_neurons' not in summary['results']:
            continue
        
        metrics.append(metric_name.replace("_", " ").title())
        final_coalition = summary['results']['final_coalition_neurons']
        coalition_sizes.append(len(final_coalition))
        
        # Analyze layer distribution
        layer_counts = {}
        for neuron_name in final_coalition:
            # Extract layer name from neuron name (e.g., "q_net.4_neuron_3" -> "q_net.4")
            parts = neuron_name.split("_")
            if len(parts) >= 3:
                layer_name = "_".join(parts[:-2])
                layer_counts[layer_name] = layer_counts.get(layer_name, 0) + 1
        
        layer_distributions[metric_name] = layer_counts
    
    if not metrics:
        print("No valid data for composition analysis")
        return
    
    # Create composition plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Coalition Sizes
    ax1 = axes[0, 0]
    bars = ax1.bar(metrics, coalition_sizes, alpha=0.7, color='lightsteelblue')
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Coalition Size')
    ax1.set_title('Final Coalition Sizes by Metric')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Layer Distribution Heatmap
    ax2 = axes[0, 1]
    
    # Get all unique layers
    all_layers = set()
    for layer_dist in layer_distributions.values():
        all_layers.update(layer_dist.keys())
    all_layers = sorted(list(all_layers))
    
    if all_layers:
        # Create matrix for heatmap
        heatmap_data = []
        for metric_name in [m.replace(" ", "_").lower() for m in metrics]:
            row = []
            for layer in all_layers:
                count = layer_distributions.get(metric_name, {}).get(layer, 0)
                row.append(count)
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        im = ax2.imshow(heatmap_data, cmap='Blues', aspect='auto')
        ax2.set_xticks(range(len(all_layers)))
        ax2.set_xticklabels(all_layers, rotation=45, ha='right')
        ax2.set_yticks(range(len(metrics)))
        ax2.set_yticklabels(metrics)
        ax2.set_title('Neuron Count by Layer and Metric')
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(all_layers)):
                if heatmap_data[i, j] > 0:
                    ax2.text(j, i, str(int(heatmap_data[i, j])),
                           ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, shrink=0.6)
    
    # Plot 3: Average Coalition Size and Range
    ax3 = axes[1, 0]
    if coalition_sizes:
        avg_size = np.mean(coalition_sizes)
        std_size = np.std(coalition_sizes)
        
        ax3.bar(['Average Coalition Size'], [avg_size], alpha=0.7, color='lightcoral')
        ax3.errorbar(['Average Coalition Size'], [avg_size], yerr=[std_size],
                    capsize=5, capthick=2, color='black')
        ax3.set_ylabel('Coalition Size')
        ax3.set_title(f'Coalition Size Statistics\nMean: {avg_size:.1f} ± {std_size:.1f}')
        ax3.grid(True, alpha=0.3)
        
        # Add individual points
        for i, size in enumerate(coalition_sizes):
            ax3.scatter([0], [size], alpha=0.6, s=50, color='red')
    
    # Plot 4: Coalition Efficiency (if we have experiment counts)
    ax4 = axes[1, 1]
    
    efficiency_data = []
    efficiency_labels = []
    
    for i, (metric_name, summary) in enumerate(summaries.items()):
        if 'results' in summary:
            coalition_size = len(summary['results'].get('final_coalition_neurons', []))
            total_experiments = summary['results'].get('total_experiments_run', 0)
            
            if total_experiments > 0:
                efficiency = coalition_size / total_experiments
                efficiency_data.append(efficiency)
                efficiency_labels.append(metrics[i] if i < len(metrics) else metric_name)
    
    if efficiency_data:
        bars = ax4.bar(efficiency_labels, efficiency_data, alpha=0.7, color='lightgreen')
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Neurons per Experiment')
        ax4.set_title('Coalition Building Efficiency')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, eff in zip(bars, efficiency_data):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No efficiency data available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Coalition Building Efficiency')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "monotonic_coalition_composition_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved composition analysis: {output_file}")

def create_monotonic_visualizations(agent_path: str, output_dir: str = None) -> None:
    """
    Create all monotonic coalition visualizations.
    
    Args:
        agent_path: Path to the agent directory
        output_dir: Directory to save plots (default: circuit_verification/monotonic/plots)
    """
    agent_path = Path(agent_path)
    
    if output_dir is None:
        output_dir = agent_path / "circuit_verification" / "monotonic" / "plots"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating monotonic coalition visualizations...")
    print(f"Agent path: {agent_path}")
    print(f"Output directory: {output_dir}")
    
    # Load summary data
    summaries = load_summary_data(str(agent_path))
    
    if not summaries:
        print("No summary data found. Cannot create visualizations.")
        return
    
    print(f"Found {len(summaries)} metrics with complete data")
    
    # Create visualizations
    create_score_progression_plots(summaries, output_dir)
    create_comparison_plots(summaries, output_dir)
    create_coalition_composition_analysis(summaries, output_dir)
    
    print(f"\nVisualization complete! All plots saved to: {output_dir}")

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Create visualizations for monotonic coalition builder results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script creates comprehensive visualizations of monotonic coalition building results:

1. Score progression plots for each metric showing how coalition scores improve
2. Comparison plots across metrics showing relative performance  
3. Coalition composition analysis showing neuron distribution by layer

Example usage:
  python -m Neuron_Selection.CircuitTooling.MonotonicTooling.visualize_monotonic_results \\
    --agent_path "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6"
        """
    )
    
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save plots (default: circuit_verification/monotonic/plots)")
    
    args = parser.parse_args()
    
    try:
        create_monotonic_visualizations(args.agent_path, args.output_dir)
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 