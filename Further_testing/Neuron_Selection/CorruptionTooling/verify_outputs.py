"""
Utility script to verify and display clean and corrupted inputs.

This script loads clean_inputs.json and corrupted_inputs.json files
and shows the differences between them to verify corruptions.
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Any

def load_json_file(file_path: str) -> Dict:
    """Load and return contents of a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def display_differences(clean_data: Dict[str, Any], corrupted_data: Dict[str, Any]):
    """Display differences between clean and corrupted data."""
    print(f"Found {len(clean_data)} entries in clean inputs")
    print(f"Found {len(corrupted_data)} entries in corrupted inputs")
    
    # Check that both files have the same keys
    clean_keys = set(clean_data.keys())
    corrupted_keys = set(corrupted_data.keys())
    
    if clean_keys != corrupted_keys:
        print("Warning: Clean and corrupted inputs have different keys!")
        print(f"Keys only in clean: {clean_keys - corrupted_keys}")
        print(f"Keys only in corrupted: {corrupted_keys - clean_keys}")
    
    # Analyze differences for common keys
    common_keys = clean_keys.intersection(corrupted_keys)
    print(f"\nAnalyzing differences for {len(common_keys)} common keys:")
    
    for key in common_keys:
        clean_array = clean_data[key]['input']
        corrupted_array = corrupted_data[key]['input']
        
        # Find difference indices
        differences = []
        for i, (clean_val, corrupted_val) in enumerate(zip(clean_array, corrupted_array)):
            if clean_val != corrupted_val:
                differences.append(i)
        
        if differences:
            print(f"\nKey: {key}")
            print(f"  Differences at indices: {differences}")
            
            # Calculate grid positions for each difference
            grid_size = int(np.sqrt((len(clean_array) - 8) // 2))
            print(f"  Grid size: {grid_size}x{grid_size}")
            
            for idx in differences:
                if idx < 8:
                    print(f"  Metadata at position {idx}: {clean_array[idx]} -> {corrupted_array[idx]}")
                else:
                    rel_idx = idx - 8
                    if rel_idx < grid_size * grid_size:
                        grid = 0
                        grid_idx = rel_idx
                    else:
                        grid = 1
                        grid_idx = rel_idx - (grid_size * grid_size)
                    
                    row = grid_idx // grid_size
                    col = grid_idx % grid_size
                    
                    print(f"  Grid {grid}, position ({row},{col}): {clean_array[idx]} -> {corrupted_array[idx]}")
        else:
            print(f"Key: {key} - No differences found!")

def main():
    parser = argparse.ArgumentParser(description='Verify and display differences between clean and corrupted inputs')
    parser.add_argument('--path', required=True, help='Path to the agent directory')
    
    args = parser.parse_args()
    
    # Convert to absolute path
    agent_path = os.path.abspath(args.path)
    
    # Define file paths
    clean_path = os.path.join(agent_path, 'clean_inputs.json')
    corrupted_path = os.path.join(agent_path, 'corrupted_inputs.json')
    
    print(f"Looking for files in: {agent_path}")
    print(f"Clean inputs file: {clean_path}")
    print(f"Corrupted inputs file: {corrupted_path}")
    
    # Check if files exist
    if not os.path.exists(clean_path):
        print(f"Error: Clean inputs file not found at {clean_path}")
        return
    
    if not os.path.exists(corrupted_path):
        print(f"Error: Corrupted inputs file not found at {corrupted_path}")
        return
    
    # Load data
    clean_data = load_json_file(clean_path)
    corrupted_data = load_json_file(corrupted_path)
    
    # Display differences
    display_differences(clean_data, corrupted_data)

if __name__ == "__main__":
    main() 