#!/usr/bin/env python3
"""
Script: run_circuit_experiments.py

Runs activation patching experiments on all files in the circuit_verification/experiments folder.
This script leverages the existing activation_patching.py functionality to process multiple experiment files
and saves results to circuit_verification/results.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add the Neuron_Selection directory to the path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
neuron_selection_dir = os.path.dirname(script_dir)
if neuron_selection_dir not in sys.path:
    sys.path.insert(0, neuron_selection_dir)
    print(f"Added Neuron_Selection directory to Python path: {neuron_selection_dir}")

# Import the PatchingExperiment class and related functions
from PatchingTooling.patching_experiment import PatchingExperiment
from activation_patching import (
    load_patches_from_file, 
    filter_patch_configs, 
    analyze_patching_results,
    EXCLUDED_LAYER,
    DEFAULT_CLEAN_INPUT,
    DEFAULT_CORRUPTED_INPUT,
    DEFAULT_CLEAN_ACTIVATIONS,
    DEFAULT_CORRUPTED_ACTIVATIONS
)
from AnalysisTooling import METRIC_FUNCTIONS


def run_experiments_from_file(
    agent_path: str,
    experiment_file: Path,
    output_dir: Path,
    device: str = "cpu",
    input_ids: List[str] = None
) -> Dict[str, Any]:
    """
    Run activation patching experiments from a single experiment file.
    
    Args:
        agent_path: Path to the agent directory
        experiment_file: Path to the experiment JSON file
        output_dir: Directory to save results
        device: Device to run on ('cpu' or 'cuda')
        input_ids: Optional list of input IDs to process
        
    Returns:
        Dictionary with experiment results and metadata
    """
    print(f"\n{'='*60}")
    print(f"Processing experiment file: {experiment_file.name}")
    print(f"{'='*60}")
    
    # Initialize patching experiment
    experiment = PatchingExperiment(agent_path, device=device)
    
    # Load patch configurations from file
    try:
        patch_configs = load_patches_from_file(str(experiment_file))
    except Exception as e:
        print(f"Error loading experiment file {experiment_file}: {e}")
        return {"error": str(e), "file": str(experiment_file)}
    
    # Filter out excluded layers
    patch_configs = filter_patch_configs(patch_configs)
    
    if not patch_configs:
        print(f"No valid patch configurations found in {experiment_file.name}")
        return {"error": "No valid configurations", "file": str(experiment_file)}
    
    # Set up file paths
    source_input = DEFAULT_CLEAN_INPUT
    target_input = DEFAULT_CORRUPTED_INPUT
    source_activations = DEFAULT_CLEAN_ACTIVATIONS
    target_activations = DEFAULT_CORRUPTED_ACTIVATIONS
    
    print(f"Running bidirectional patching experiments:")
    print(f"  Agent: {agent_path}")
    print(f"  Clean input: {source_input}")
    print(f"  Corrupted input: {target_input}")
    print(f"  Clean activations: {source_activations}")
    print(f"  Corrupted activations: {target_activations}")
    print(f"  Number of patch configurations: {len(patch_configs)}")
    print(f"  Excluded layer: {EXCLUDED_LAYER}")
    print("--------------------------------------------------")
    
    try:
        # Run bidirectional patching
        clean_to_corrupted, corrupted_to_clean = experiment.run_bidirectional_patching(
            source_input,
            target_input,
            source_activations,
            target_activations,
            patch_configs,
            input_ids
        )
        
        print(f"\nResults for {experiment_file.name}:")
        print(f"  Clean → Corrupted: {len(clean_to_corrupted)} result files")
        print(f"  Corrupted → Clean: {len(corrupted_to_clean)} result files")
        
        # Create experiment-specific output directory
        exp_name = experiment_file.stem.replace('experiments_', '')
        exp_output_dir = output_dir / exp_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move results to the experiment-specific directory
        moved_files = {"denoising": [], "noising": []}
        
        # Move denoising results
        if clean_to_corrupted:
            denoising_dir = exp_output_dir / "denoising"
            denoising_dir.mkdir(exist_ok=True)
            
            for input_id, result_file in clean_to_corrupted.items():
                if os.path.exists(result_file):
                    new_path = denoising_dir / os.path.basename(result_file)
                    os.rename(result_file, new_path)
                    moved_files["denoising"].append(str(new_path))
        
        # Move noising results
        if corrupted_to_clean:
            noising_dir = exp_output_dir / "noising"
            noising_dir.mkdir(exist_ok=True)
            
            for input_id, result_file in corrupted_to_clean.items():
                if os.path.exists(result_file):
                    new_path = noising_dir / os.path.basename(result_file)
                    os.rename(result_file, new_path)
                    moved_files["noising"].append(str(new_path))
        
        # Clean up the original output directory if it's empty
        original_output_dir = Path(experiment.output_dir)
        try:
            if original_output_dir.exists():
                # Remove empty subdirectories
                for subdir in ["denoising", "noising"]:
                    subdir_path = original_output_dir / subdir
                    if subdir_path.exists() and not any(subdir_path.iterdir()):
                        subdir_path.rmdir()
                
                # Remove main directory if empty
                if not any(original_output_dir.iterdir()):
                    original_output_dir.rmdir()
        except OSError:
            pass  # Directory not empty or other issue
        
        return {
            "success": True,
            "file": str(experiment_file),
            "experiment_name": exp_name,
            "output_directory": str(exp_output_dir),
            "patch_count": len(patch_configs),
            "results": moved_files,
            "clean_to_corrupted_count": len(clean_to_corrupted),
            "corrupted_to_clean_count": len(corrupted_to_clean)
        }
        
    except Exception as e:
        print(f"Error running experiments from {experiment_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "file": str(experiment_file)}


def run_all_circuit_experiments(
    agent_path: str,
    device: str = "cpu",
    input_ids: List[str] = None,
    analyze_results: bool = True,
    subfolder: str = "results"
) -> Dict[str, Any]:
    """
    Run activation patching experiments on all files in circuit_verification/experiments.
    
    Args:
        agent_path: Path to the agent directory
        device: Device to run on ('cpu' or 'cuda')
        input_ids: Optional list of input IDs to process
        analyze_results: Whether to analyze results after running experiments
        subfolder: Subfolder name within circuit_verification for results (default: "results")
        
    Returns:
        Dictionary with summary of all experiments
    """
    agent_path = Path(agent_path)
    experiments_dir = agent_path / "circuit_verification" / "experiments"
    results_dir = agent_path / "circuit_verification" / subfolder
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if experiments directory exists
    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found: {experiments_dir}")
        return {"error": f"Experiments directory not found: {experiments_dir}"}
    
    # Find all experiment files
    experiment_files = list(experiments_dir.glob("experiments_*.json"))
    
    if not experiment_files:
        print(f"No experiment files found in {experiments_dir}")
        return {"error": f"No experiment files found in {experiments_dir}"}
    
    print(f"Found {len(experiment_files)} experiment files:")
    for exp_file in sorted(experiment_files):
        print(f"  - {exp_file.name}")
    
    # Run experiments for each file
    all_results = []
    successful = 0
    failed = 0
    
    for exp_file in sorted(experiment_files):
        result = run_experiments_from_file(
            str(agent_path),
            exp_file,
            results_dir,
            device,
            input_ids
        )
        
        all_results.append(result)
        
        if result.get("success", False):
            successful += 1
        else:
            failed += 1
    
    # Save summary
    summary = {
        "agent_path": str(agent_path),
        "experiments_processed": len(experiment_files),
        "successful": successful,
        "failed": failed,
        "results_directory": str(results_dir),
        "experiment_results": all_results
    }
    
    summary_file = results_dir / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiment files processed: {len(experiment_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {results_dir}")
    print(f"Summary saved to: {summary_file}")
    
    # Analyze results if requested
    if analyze_results and successful > 0:
        print(f"\nAnalyzing results...")
        metrics = list(METRIC_FUNCTIONS.keys())
        
        for result in all_results:
            if result.get("success", False):
                exp_output_dir = result["output_directory"]
                print(f"\nAnalyzing results for {result['experiment_name']}...")
                try:
                    analyze_patching_results(exp_output_dir, metrics, analyze_subdirs=True)
                except Exception as e:
                    print(f"Error analyzing results for {result['experiment_name']}: {e}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run circuit verification experiments")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on ('cpu' or 'cuda'). Default: cpu")
    parser.add_argument("--input_ids", type=str,
                       help="Comma-separated list of input IDs to process (optional)")
    parser.add_argument("--no_analyze", action="store_true",
                       help="Skip analysis of results after running experiments")
    parser.add_argument("--subfolder", type=str, default="results",
                       help="Subfolder name within circuit_verification for results (default: results)")
    
    args = parser.parse_args()
    
    # Parse input IDs if provided
    input_ids = None
    if args.input_ids:
        input_ids = [id.strip() for id in args.input_ids.split(',')]
    
    # Run all circuit experiments
    summary = run_all_circuit_experiments(
        args.agent_path,
        args.device,
        input_ids,
        analyze_results=not args.no_analyze,
        subfolder=args.subfolder
    )
    
    # Exit with appropriate code
    if summary.get("failed", 0) > 0:
        print(f"\nWarning: {summary['failed']} experiment(s) failed")
        return 1
    
    print(f"\nAll experiments completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main()) 