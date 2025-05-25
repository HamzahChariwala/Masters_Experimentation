#!/usr/bin/env python3
"""
Script: patching_analysis.py

Analyzes the results of activation patching experiments, calculating various
metrics to quantify the effects of patching on model behavior.
"""
import argparse
import os
import sys
from typing import List, Optional
from AnalysisTooling import process_result_directory, process_result_file


def main():
    parser = argparse.ArgumentParser(
        description="Analyze activation patching experiment results.")
    
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory (e.g., Agent_Storage/SpawnTests/biased/biased-v1)")
    
    parser.add_argument("--results_dir", type=str, default=None,
                       help="Directory containing result files (defaults to <agent_path>/patching_results)")
    
    parser.add_argument("--file", type=str, default=None,
                       help="Process a specific result file instead of all files in the directory")
    
    parser.add_argument("--metrics", type=str, default="output_logit_delta",
                       help="Comma-separated list of metrics to calculate (default: output_logit_delta)")
    
    parser.add_argument("--subdirs", action="store_true", default=False,
                       help="Process subdirectories (e.g., denoising, noising) if they exist")
    
    args = parser.parse_args()
    
    # Set up the results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(args.agent_path, "patching_results")
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    # Parse the metrics list
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    
    print(f"Analyzing patching results with metrics: {', '.join(metrics)}")
    
    # Process a specific file if requested
    if args.file:
        file_path = args.file
        if not os.path.isabs(file_path):
            # If not an absolute path, assume it's relative to the results directory
            file_path = os.path.join(results_dir, file_path)
        
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return 1
        
        try:
            print(f"Processing file: {file_path}")
            process_result_file(file_path, metrics)
            print("File processing completed.")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return 1
    else:
        # Process the main results directory
        print(f"Processing results directory: {results_dir}")
        results = process_result_directory(results_dir, metrics)
        
        # Count successes and failures
        successes = sum(1 for status in results.values() if status == "success")
        failures = len(results) - successes
        
        print(f"Processed {len(results)} files: {successes} successful, {failures} failed.")
        
        # Process subdirectories if requested
        if args.subdirs:
            subdirs = [d for d in os.listdir(results_dir) 
                      if os.path.isdir(os.path.join(results_dir, d))]
            
            for subdir in subdirs:
                subdir_path = os.path.join(results_dir, subdir)
                print(f"\nProcessing subdirectory: {subdir_path}")
                
                subdir_results = process_result_directory(subdir_path, metrics)
                
                # Count successes and failures
                subdir_successes = sum(1 for status in subdir_results.values() if status == "success")
                subdir_failures = len(subdir_results) - subdir_successes
                
                print(f"Processed {len(subdir_results)} files in {subdir}: "
                     f"{subdir_successes} successful, {subdir_failures} failed.")
    
    print("\nAnalysis completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 