#!/usr/bin/env python3
"""
Create Circuit-Style Plots for Monotonic Coalition Results

This script generates individual circuit-verification-style plots for each metric,
matching the exact format of the descending plots but showing coalition building progression.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add the Neuron_Selection directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
neuron_selection_dir = os.path.abspath(os.path.join(script_dir, "../.."))
project_root = os.path.abspath(os.path.join(neuron_selection_dir, ".."))

for path in [neuron_selection_dir, project_root]:
    if path not in sys.path:
        sys.path.append(path)

from Neuron_Selection.CircuitTooling.MonotonicTooling.visualize_monotonic_results import create_circuit_style_plots


def load_all_summaries(agent_path: str) -> Dict[str, Dict[str, Any]]:
    """Load summary files for all metrics."""
    agent_path = Path(agent_path)
    monotonic_dir = agent_path / "circuit_verification" / "monotonic"
    
    if not monotonic_dir.exists():
        print(f"Monotonic directory not found: {monotonic_dir}")
        return {}
    
    summaries = {}
    
    # Find all metric directories
    for metric_dir in monotonic_dir.iterdir():
        if not metric_dir.is_dir():
            continue
            
        summary_file = metric_dir / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summaries[metric_dir.name] = json.load(f)
                print(f"Loaded summary for: {metric_dir.name}")
            except Exception as e:
                print(f"Error loading summary for {metric_dir.name}: {e}")
    
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Create circuit-style plots for monotonic coalition results")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--metrics", type=str, nargs="*",
                       help="Specific metrics to plot (default: all found)")
    
    args = parser.parse_args()
    
    # Load all summaries
    summaries = load_all_summaries(args.agent_path)
    
    if not summaries:
        print("No monotonic coalition summaries found!")
        return
    
    # Filter to specific metrics if requested
    if args.metrics:
        filtered_summaries = {k: v for k, v in summaries.items() if k in args.metrics}
        if not filtered_summaries:
            print(f"None of the requested metrics found: {args.metrics}")
            print(f"Available metrics: {list(summaries.keys())}")
            return
        summaries = filtered_summaries
    
    print(f"Creating circuit-style plots for {len(summaries)} metrics...")
    
    # Create the plots
    output_dir = Path(args.agent_path) / "circuit_verification" / "monotonic"
    create_circuit_style_plots(summaries, args.agent_path, output_dir)
    
    print("Circuit-style plot generation complete!")


if __name__ == "__main__":
    main() 