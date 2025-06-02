#!/usr/bin/env python3
"""
Script to generate cumulative patching experiments from filtered results.

This script takes filtered patching results JSON files and creates new JSONs
where experiments are ordered by their mean metric value (highest to lowest)
and then creates cumulative experiments starting from the highest scoring
experiment and progressively adding the next highest scoring ones.
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any


def collect_all_experiments(data: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Collect all unique experiments from the filtered data and their mean scores.
    
    Args:
        data: The loaded JSON data from the filtered file
        
    Returns:
        List of tuples (experiment_name, mean_score) sorted by mean_score descending
    """
    experiments = {}
    
    # First try to collect from 'averaged' section (new format)
    if 'averaged' in data:
        for exp_name, exp_data in data['averaged'].items():
            # Use the noising mean score for ordering (could also use denoising)
            if 'noising' in exp_data and 'mean' in exp_data['noising']:
                experiments[exp_name] = exp_data['noising']['mean']
    # Fall back to 'common' section (old format, for backward compatibility)
    elif 'common' in data:
        for exp_name, exp_data in data['common'].items():
            # Use the noising mean score for ordering (could also use denoising)
            if 'noising' in exp_data and 'mean' in exp_data['noising']:
                experiments[exp_name] = exp_data['noising']['mean']
    
    # Then collect from 'noising' section for any missing experiments
    if 'noising' in data:
        for exp_name, exp_data in data['noising'].items():
            if exp_name not in experiments and 'mean' in exp_data:
                experiments[exp_name] = exp_data['mean']
    
    # Finally collect from 'denoising' section for any remaining missing experiments
    if 'denoising' in data:
        for exp_name, exp_data in data['denoising'].items():
            if exp_name not in experiments and 'mean' in exp_data:
                experiments[exp_name] = exp_data['mean']
    
    # Sort by mean score (highest to lowest)
    sorted_experiments = sorted(experiments.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_experiments


def create_cumulative_experiments(sorted_experiments: List[Tuple[str, float]], 
                                original_data: Dict[str, Any], max_experiments: int = 30) -> Dict[str, Any]:
    """
    Create cumulative experiments where each experiment includes all previous ones.
    
    Args:
        sorted_experiments: List of (experiment_name, mean_score) sorted by score
        original_data: Original filtered data for reference
        max_experiments: Maximum number of cumulative experiments to generate (default: 30)
        
    Returns:
        New JSON structure with cumulative experiments
    """
    # Limit the number of experiments to process
    experiments_to_process = sorted_experiments[:max_experiments]
    
    cumulative_data = {
        "metric": original_data.get("metric", "unknown"),
        "threshold": original_data.get("threshold", 0.5),
        "source_file": "Generated from cumulative patching script",
        "description": f"Cumulative experiments ordered by mean metric value (highest to lowest), limited to {max_experiments} experiments",
        "original_experiment_count": len(sorted_experiments),
        "processed_experiment_count": len(experiments_to_process),
        "max_experiments": max_experiments,
        "experiments": {}
    }
    
    # Create cumulative experiments
    for i, (exp_name, mean_score) in enumerate(experiments_to_process, 1):
        # Get all experiments up to this point
        cumulative_exp_names = [exp[0] for exp in experiments_to_process[:i]]
        
        # Calculate cumulative mean (simple average for now)
        cumulative_mean = sum(exp[1] for exp in experiments_to_process[:i]) / i
        
        experiment_data = {
            "experiment_name": f"cumulative_{i:02d}",
            "description": f"Top {i} experiment(s) by mean score",
            "included_experiments": cumulative_exp_names,
            "experiment_count": i,
            "cumulative_mean_score": cumulative_mean,
            "highest_individual_score": experiments_to_process[0][1],
            "lowest_individual_score": experiments_to_process[i-1][1],
            "individual_scores": [exp[1] for exp in experiments_to_process[:i]]
        }
        
        cumulative_data["experiments"][f"cumulative_{i:02d}"] = experiment_data
    
    return cumulative_data


def process_filtered_file(input_file: Path, output_dir: Path, max_experiments: int = 30) -> bool:
    """
    Process a single filtered file and generate cumulative experiments.
    
    Args:
        input_file: Path to the input filtered file
        output_dir: Directory to save the output file
        max_experiments: Maximum number of cumulative experiments to generate (default: 30)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the input data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Collect and sort experiments
        sorted_experiments = collect_all_experiments(data)
        
        if not sorted_experiments:
            print(f"  No experiments found in {input_file.name}")
            return False
        
        # Create cumulative experiments
        cumulative_data = create_cumulative_experiments(sorted_experiments, data, max_experiments)
        
        # Determine output filename (replace 'filtered_' with 'coalitions_')
        input_name = input_file.stem
        if input_name.startswith('filtered_'):
            output_name = input_name.replace('filtered_', 'coalitions_', 1) + '.json'
        else:
            output_name = f"coalitions_{input_name}.json"
        
        output_file = output_dir / output_name
        
        # Save the output
        with open(output_file, 'w') as f:
            json.dump(cumulative_data, f, indent=2)
        
        print(f"  {input_file.name} -> {output_name} ({len(sorted_experiments)} total, {cumulative_data['processed_experiment_count']} processed)")
        return True
        
    except Exception as e:
        print(f"  Error processing {input_file.name}: {e}")
        return False


def generate_cumulative_experiments(agent_path: str, input_files: List[str] = None, max_experiments: int = 30) -> None:
    """
    Generate cumulative experiments for filtered files.
    
    Args:
        agent_path: Path to the agent directory
        input_files: Optional list of specific files to process. If None, processes all filtered files.
        max_experiments: Maximum number of cumulative experiments to generate (default: 30)
    """
    agent_path = Path(agent_path)
    filtered_dir = agent_path / "patching_results" / "filtered"
    circuit_verification_dir = agent_path / "circuit_verification" / "descending" / "descriptions"
    
    # Create circuit_verification/descending/descriptions directory if it doesn't exist
    circuit_verification_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if filtered directory exists
    if not filtered_dir.exists():
        print(f"Error: Filtered directory not found: {filtered_dir}")
        return
    
    # Determine which files to process
    if input_files:
        # Process specific files
        files_to_process = []
        for file_name in input_files:
            file_path = filtered_dir / file_name
            if file_path.exists():
                files_to_process.append(file_path)
            else:
                print(f"Warning: File not found: {file_path}")
    else:
        # Process all filtered_*.json files
        files_to_process = list(filtered_dir.glob("filtered_*.json"))
    
    if not files_to_process:
        print("No files to process")
        return
    
    print(f"Processing {len(files_to_process)} filtered files (max {max_experiments} experiments each):")
    
    successful = 0
    for file_path in sorted(files_to_process):
        if process_filtered_file(file_path, circuit_verification_dir, max_experiments):
            successful += 1
    
    print(f"\nCompleted: {successful}/{len(files_to_process)} files processed successfully")
    print(f"Output directory: {circuit_verification_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate cumulative patching experiments")
    parser.add_argument("--agent_path", type=str, 
                       help="Path to the agent directory (processes all filtered files)")
    parser.add_argument("input_file", nargs='?', 
                       help="Path to a specific filtered JSON file (alternative to --agent_path)")
    parser.add_argument("--output", "-o", 
                       help="Output file path (only used with input_file)")
    parser.add_argument("--max_experiments", type=int, default=30,
                       help="Maximum number of cumulative experiments to generate (default: 30)")
    
    args = parser.parse_args()
    
    if args.agent_path:
        # Process all filtered files in the agent directory
        generate_cumulative_experiments(args.agent_path, max_experiments=args.max_experiments)
    elif args.input_file:
        # Process a single file (legacy mode)
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: Input file {input_path} does not exist")
            return 1
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_cumulative.json"
        
        # Load and process the file
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading input file: {e}")
            return 1
        
        # Collect and sort experiments
        sorted_experiments = collect_all_experiments(data)
        
        if not sorted_experiments:
            print("No experiments found in the input file")
            return 1
        
        print(f"Found {len(sorted_experiments)} experiments:")
        for i, (exp_name, score) in enumerate(sorted_experiments, 1):
            print(f"  {i:2d}. {exp_name}: {score:.6f}")
        
        # Create cumulative experiments
        cumulative_data = create_cumulative_experiments(sorted_experiments, data, args.max_experiments)
        
        # Save the output
        try:
            with open(output_path, 'w') as f:
                json.dump(cumulative_data, f, indent=2)
            print(f"\nCumulative experiments saved to: {output_path}")
            print(f"Generated {len(cumulative_data['experiments'])} cumulative experiments (max {args.max_experiments})")
        except Exception as e:
            print(f"Error saving output file: {e}")
            return 1
    else:
        print("Error: Must provide either --agent_path or input_file")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 