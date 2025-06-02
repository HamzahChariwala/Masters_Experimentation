#!/usr/bin/env python3
"""
Script to run circuit verification experiments.

This script takes experiment JSON files and runs activation patching experiments
to verify the selected circuits.
"""

import os
import sys
import json
import argparse
import subprocess
from typing import Dict, List, Any

# Add the Neuron_Selection directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
neuron_selection_dir = os.path.dirname(script_dir)
if neuron_selection_dir not in sys.path:
    sys.path.insert(0, neuron_selection_dir)


def run_circuit_experiments(agent_path: str) -> None:
    """
    Run circuit verification experiments for all experiment files.
    
    Args:
        agent_path: Path to the agent directory
    """
    experiments_dir = os.path.join(agent_path, "circuit_verification", "experiments")
    
    if not os.path.exists(experiments_dir):
        print(f"Experiments directory not found: {experiments_dir}")
        return
    
    # Find all experiment files
    experiment_files = []
    for filename in os.listdir(experiments_dir):
        if filename.startswith('experiments_') and filename.endswith('.json'):
            experiment_files.append(os.path.join(experiments_dir, filename))
    
    if not experiment_files:
        print(f"No experiment files found in {experiments_dir}")
        return
    
    print(f"Found {len(experiment_files)} experiment files to process")
    
    # Process each experiment file
    for experiment_file in experiment_files:
        # Extract metric name from filename
        filename = os.path.basename(experiment_file)
        metric_name = filename[12:-5]  # Remove 'experiments_' prefix and '.json' suffix
        
        print(f"  Running experiments for {metric_name}...")
        
        # For now, just report that we would run the experiments
        # In the actual implementation, this would call the activation patching script
        with open(experiment_file, 'r') as f:
            experiments = json.load(f)
        
        total_experiments = len([exp for exp in experiments if exp])  # Count non-empty experiments
        print(f"    Would run {total_experiments} circuit verification experiments")
        
        # Create output directory
        results_dir = os.path.join(agent_path, "circuit_verification", "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create placeholder results file
        results_file = os.path.join(results_dir, f"results_{metric_name}.json")
        with open(results_file, 'w') as f:
            json.dump({
                "metric": metric_name,
                "total_experiments": total_experiments,
                "status": "placeholder - circuit verification not implemented"
            }, f, indent=2)
        
        print(f"    Created placeholder results file: {results_file}")


def main():
    """Main function to run circuit experiments."""
    parser = argparse.ArgumentParser(description='Run circuit verification experiments')
    parser.add_argument('--agent_path', type=str, required=True,
                       help='Path to the agent directory')
    
    args = parser.parse_args()
    
    # Validate agent path
    if not os.path.exists(args.agent_path):
        print(f"Error: Agent path '{args.agent_path}' does not exist")
        return
    
    # Run circuit experiments
    run_circuit_experiments(args.agent_path)


if __name__ == "__main__":
    main() 