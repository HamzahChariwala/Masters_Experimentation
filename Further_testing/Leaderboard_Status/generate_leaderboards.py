#!/usr/bin/env python3
"""
Script to generate leaderboards from agent performance metrics.
Extracts key metrics from performance_all_states.json files and ranks agents.
"""

import os
import json
from collections import defaultdict
import pandas as pd

# Metrics we want to track
METRICS = [
    'avg_path_length',
    'avg_lava_steps',
    'goal_reached_proportion',
    'next_cell_lava_proportion',
    'risky_diagonal_proportion'
]

def find_performance_files(root_dir):
    """Find all performance_all_states.json files and their associated agent IDs."""
    performance_files = []
    
    for root, dirs, files in os.walk(root_dir):
        if 'performance_all_states.json' in files:
            # Get the path relative to the root_dir
            rel_path = os.path.relpath(root, root_dir)
            path_parts = rel_path.split(os.sep)
            
            # We want the parent folder and current folder (excluding evaluation_summary)
            meaningful_parts = [p for p in path_parts if p and p not in ['.', '..', 'evaluation_summary']]
            if len(meaningful_parts) >= 2:
                # Take the last two parts before 'evaluation_summary'
                agent_id = f"{meaningful_parts[-2]}/{meaningful_parts[-1]}"
                performance_files.append({
                    'path': os.path.join(root, 'performance_all_states.json'),
                    'agent_id': agent_id
                })
    
    return performance_files

def extract_metrics(performance_file):
    """Extract the required metrics from a performance file."""
    try:
        with open(performance_file['path'], 'r') as f:
            data = json.load(f)
            
        metrics = {}
        if 'overall_summary' in data:
            for metric in METRICS:
                if metric in data['overall_summary']:
                    metrics[metric] = data['overall_summary'][metric]
                else:
                    print(f"Warning: Metric {metric} not found in overall_summary of {performance_file['path']}")
                    metrics[metric] = None
        else:
            print(f"Warning: No overall_summary found in {performance_file['path']}")
            metrics = {metric: None for metric in METRICS}
                
        return {
            'agent_id': performance_file['agent_id'],
            **metrics
        }
    except Exception as e:
        print(f"Error processing {performance_file['path']}: {e}")
        return None

def create_leaderboards(metrics_data):
    """Create leaderboards for each metric."""
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(metrics_data)
    
    leaderboards = {}
    for metric in METRICS:
        # Sort based on metric (higher is better for some metrics, lower for others)
        ascending = metric in ['avg_path_length', 'avg_lava_steps']  # These should be minimized
        # Drop None values and sort
        sorted_df = df[['agent_id', metric]].dropna().sort_values(by=metric, ascending=ascending)
        
        # Create leaderboard entry
        leaderboards[metric] = sorted_df.to_dict('records')
    
    return leaderboards

def save_leaderboards(leaderboards, output_file):
    """Save leaderboards to a file."""
    # Create a formatted string for each leaderboard
    output = []
    
    for metric, rankings in leaderboards.items():
        output.append(f"\n{'='*50}")
        output.append(f"Leaderboard for {metric}")
        output.append('='*50)
        
        for i, entry in enumerate(rankings, 1):
            value = entry[metric]
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            output.append(f"{i:2d}. {entry['agent_id']:<30} {formatted_value}")
    
    # Save both text and JSON formats
    with open(output_file + '.txt', 'w') as f:
        f.write('\n'.join(output))
    
    with open(output_file + '.json', 'w') as f:
        json.dump(leaderboards, f, indent=2)
    
    print(f"Leaderboards saved to {output_file}.txt and {output_file}.json")
    return output

def verify_results(leaderboards, performance_files):
    """Verify that the reported values match the original files."""
    print("\nVerifying results...")
    
    # Sample a few random entries to verify
    import random
    sample_size = min(5, len(performance_files))
    samples = random.sample(performance_files, sample_size)
    
    for sample in samples:
        print(f"\nVerifying {sample['agent_id']}...")
        with open(sample['path'], 'r') as f:
            original_data = json.load(f)
        
        # Find this agent in our leaderboards
        for metric in METRICS:
            original_value = original_data.get('overall_summary', {}).get(metric)
            reported_value = None
            
            # Find the agent in the leaderboard for this metric
            for entry in leaderboards[metric]:
                if entry['agent_id'] == sample['agent_id']:
                    reported_value = entry[metric]
                    break
            
            if original_value != reported_value:
                print(f"Mismatch in {metric}:")
                print(f"  Original: {original_value}")
                print(f"  Reported: {reported_value}")
            else:
                print(f"✓ {metric} verified")

def main():
    # Find all performance files
    performance_files = find_performance_files('.')
    if not performance_files:
        print("No performance files found. Make sure you're running this script from the correct directory.")
        return
        
    print(f"Found {len(performance_files)} performance files")
    
    # Extract metrics from each file
    metrics_data = []
    for pf in performance_files:
        metrics = extract_metrics(pf)
        if metrics:
            metrics_data.append(metrics)
    
    if not metrics_data:
        print("No valid metrics data found.")
        return
        
    print(f"Successfully processed {len(metrics_data)} files")
    
    # Create leaderboards
    leaderboards = create_leaderboards(metrics_data)
    
    # Save leaderboards
    output = save_leaderboards(leaderboards, 'leaderboard')
    
    # Verify results
    verify_results(leaderboards, performance_files)
    
    # Print leaderboards to console
    print("\nLeaderboards:")
    print('\n'.join(output))

if __name__ == "__main__":
    main() 