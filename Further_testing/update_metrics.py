#!/usr/bin/env python3
"""
Script to update existing patching results with new metrics
"""
import os
import sys
from Neuron_Selection.AnalysisTooling import process_result_directory

def update_agent_metrics(agent_path, metrics):
    """Update patching results for the specified agent with new metrics"""
    results_dir = os.path.join(agent_path, 'patching_results')
    
    if not os.path.exists(results_dir):
        print(f"Error: Patching results directory not found at {results_dir}")
        return False
    
    print(f"Processing agent: {agent_path}")
    
    # Process noising results
    print(f"Processing noising results...")
    noising_dir = os.path.join(results_dir, 'noising')
    if os.path.exists(noising_dir):
        process_result_directory(noising_dir, metrics)
    else:
        print(f"No noising directory found at {noising_dir}")
    
    # Process denoising results
    print(f"Processing denoising results...")
    denoising_dir = os.path.join(results_dir, 'denoising')
    if os.path.exists(denoising_dir):
        process_result_directory(denoising_dir, metrics)
    else:
        print(f"No denoising directory found at {denoising_dir}")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update patching results with new metrics")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--metrics", type=str, default="confidence_margin_magnitude,reversed_pearson_correlation",
                       help="Comma-separated list of metrics to add")
    
    args = parser.parse_args()
    
    # Process arguments
    agent_path = args.agent_path
    metrics = args.metrics.split(',')
    
    print(f"Adding metrics: {', '.join(metrics)}")
    
    # Update the agent's metrics
    success = update_agent_metrics(agent_path, metrics)
    
    if success:
        print(f"Successfully updated metrics for {agent_path}")
    else:
        print(f"Failed to update metrics for {agent_path}")
        sys.exit(1) 