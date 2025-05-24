"""
Test script for the State Corruption Tool.

This script creates a sample filtered_states.json file with synthetic data
and then launches the corruption tool for testing.
"""

import os
import json
import numpy as np
import argparse
import subprocess
import random
import sys

def create_sample_data(output_dir: str, num_states: int = 10, grid_size: int = 5):
    """Create sample filtered_states.json with synthetic data."""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filtered_states.json
    filtered_states_path = os.path.join(output_dir, 'filtered_states.json')
    
    # Generate sample data
    data = {}
    
    for i in range(num_states):
        # Create a unique key with the format used in state_filter.py
        key = f"MiniGrid-LavaCrossingS11N5-v0-{i:05d}-1,2,0-{i+1:04d}"
        
        # Create first 8 values (metadata)
        first_values = [random.randint(0, 1) for _ in range(8)]
        
        # Create grid data (two square grids)
        grid_values = []
        for _ in range(2 * grid_size * grid_size):
            grid_values.append(random.randint(0, 1))
        
        # Combine into input array
        input_array = first_values + grid_values
        
        # Add to data dictionary
        data[key] = {"input": input_array}
    
    # Write to JSON file
    with open(filtered_states_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created sample data with {num_states} states at {filtered_states_path}")
    return filtered_states_path

def main():
    parser = argparse.ArgumentParser(description='Test the State Corruption Tool')
    parser.add_argument('--output', default='test_agent', help='Output directory for test data')
    parser.add_argument('--states', type=int, default=10, help='Number of sample states to generate')
    parser.add_argument('--grid-size', type=int, default=5, help='Size of the grid (N x N)')
    
    args = parser.parse_args()
    
    # Get absolute path for output directory
    output_dir = os.path.abspath(args.output)
    
    # Create sample data
    create_sample_data(output_dir, args.states, args.grid_size)
    
    # Get directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to corruption_tool.py
    corruption_tool_path = os.path.join(script_dir, 'corruption_tool.py')
    
    # Run the corruption tool with absolute paths
    print(f"Launching corruption tool with test data...")
    print(f"Running: python {corruption_tool_path} --path {output_dir}")
    
    # Use the current Python executable
    python_exe = sys.executable
    subprocess.run([python_exe, corruption_tool_path, '--path', output_dir])

if __name__ == "__main__":
    main() 