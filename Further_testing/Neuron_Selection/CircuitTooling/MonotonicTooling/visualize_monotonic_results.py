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
    total_experiments = []
    
    for metric_name, summary in summaries.items():
        if 'results' not in summary:
            continue
            
        results = summary['results']
        if 'score_progression' in results:
            metrics.append(metric_name.replace("_", " ").title())
            initial_scores.append(results['score_progression']['initial_score'])
            final_scores.append(results['score_progression']['final_score'])
            improvements.append(results['score_progression']['total_improvement'])
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
    
    # Plot 2: Total Improvement
    ax2 = axes[0, 1]
    bars = ax2.bar(metrics, improvements, alpha=0.7, color='lightgreen')
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Total Improvement')
    ax2.set_title('Total Score Improvement by Metric')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
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
    
    # Plot 4: Efficiency (Improvement per Experiment)
    ax4 = axes[1, 1]
    efficiency = [imp/exp if exp > 0 else 0 for imp, exp in zip(improvements, total_experiments)]
    bars = ax4.bar(metrics, efficiency, alpha=0.7, color='lightsteelblue')
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Improvement per Experiment')
    ax4.set_title('Coalition Building Efficiency')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save comparison plot
    output_file = output_dir / "monotonic_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {output_file}")

def create_coalition_composition_analysis(summaries: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """
    Create analysis of coalition composition across metrics.
    
    Args:
        summaries: Dictionary of metric summaries
        output_dir: Directory to save plots
    """
    if not summaries:
        print("No summary data available for composition analysis")
        return
    
    # Analyze layer composition
    layer_composition = {}
    metric_names = []
    
    for metric_name, summary in summaries.items():
        if 'results' not in summary or 'final_coalition_neurons' not in summary['results']:
            continue
        
        metric_names.append(metric_name.replace("_", " ").title())
        neurons = summary['results']['final_coalition_neurons']
        
        # Count neurons by layer
        layer_counts = {}
        for neuron in neurons:
            layer = neuron.split('_neuron_')[0]
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        layer_composition[metric_name] = layer_counts
    
    if not layer_composition:
        print("No coalition composition data found")
        return
    
    # Get all unique layers
    all_layers = set()
    for composition in layer_composition.values():
        all_layers.update(composition.keys())
    all_layers = sorted(list(all_layers))
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bottom = np.zeros(len(metric_names))
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_layers)))
    
    for i, layer in enumerate(all_layers):
        layer_counts = []
        for metric in layer_composition.keys():
            layer_counts.append(layer_composition[metric].get(layer, 0))
        
        bars = ax.bar(metric_names, layer_counts, bottom=bottom, 
                     label=layer, color=colors[i], alpha=0.8)
        bottom += np.array(layer_counts)
        
        # Add count labels on bars if count > 0
        for j, (bar, count) in enumerate(zip(bars, layer_counts)):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., 
                       bottom[j] - count/2,
                       str(count), ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('Coalition Composition by Layer Across Metrics')
    ax.legend(title='Layers', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save composition plot
    output_file = output_dir / "monotonic_composition.png"
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