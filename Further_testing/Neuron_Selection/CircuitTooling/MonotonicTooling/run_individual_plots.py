#!/usr/bin/env python3
"""
Wrapper script to generate individual circuit-style plots for all monotonic coalition metrics.

This script runs the individual plot generation for all available metrics in the 
monotonic coalition results directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate individual circuit-style plots for all monotonic coalition metrics"
    )
    parser.add_argument(
        "--agent_path", 
        type=str, 
        required=True,
        help="Path to the agent directory"
    )
    parser.add_argument(
        "--metrics", 
        type=str, 
        nargs="*",
        help="Specific metrics to process (default: all found)"
    )
    
    args = parser.parse_args()
    
    # Build command to run the individual plots script
    cmd = [
        sys.executable, "-m", 
        "Neuron_Selection.CircuitTooling.MonotonicTooling.create_individual_plots",
        "--agent_path", args.agent_path
    ]
    
    if args.metrics:
        cmd.extend(["--metrics"] + args.metrics)
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the script
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\nIndividual plot generation completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nError running individual plot generation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 