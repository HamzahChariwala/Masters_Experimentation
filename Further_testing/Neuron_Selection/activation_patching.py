#!/usr/bin/env python3
"""
Script: activation_patching.py

Performs activation patching experiments on trained agents.
Allows for patching activations from clean runs into corrupted runs
and other customizable patching experiments.
"""
import argparse
import sys
import os
import json
from typing import Dict, List, Union, Optional

# Add the Neuron_Selection directory to the path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
if not script_dir in sys.path:
    sys.path.insert(0, script_dir)
    print(f"Added Neuron_Selection directory to Python path: {script_dir}")

# Make sure the project root is in the path
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if not project_root in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to Python path: {project_root}")

from Neuron_Selection.PatchingTooling.patching_experiment import PatchingExperiment
# Import functions from AnalysisTooling for result analysis
from Neuron_Selection.AnalysisTooling import (
    process_result_file,
    process_result_directory,
    METRIC_FUNCTIONS
)

# Hardcoded constants that don't change between runs
DEFAULT_DEVICE = "cpu"
DEFAULT_ORGANIZE_BY_INPUT = True
DEFAULT_BIDIRECTIONAL = True  # Changed to True to make bidirectional the default
DEFAULT_OUTPUT_PREFIX = ""
DEFAULT_ANALYZE_RESULTS = True  # Default to analyzing results
DEFAULT_METRICS = ",".join(METRIC_FUNCTIONS.keys())  # All available metrics

# Default file names and paths
DEFAULT_CLEAN_INPUT = "clean_inputs.json"
DEFAULT_CORRUPTED_INPUT = "corrupted_inputs.json"
DEFAULT_CLEAN_ACTIVATIONS = "clean_activations.npz"
DEFAULT_CORRUPTED_ACTIVATIONS = "corrupted_activations.npz"
DEFAULT_RESULT_FILENAME = "patching_results.json"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run activation patching experiments.")
    
    # Required arguments
    parser.add_argument("--agent_path", required=True,
                        help="Path to the agent directory")
    
    # Optional arguments with reasonable defaults
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help=f"Device to run on ('cpu' or 'cuda'). Default: {DEFAULT_DEVICE}")
    
    # Patching experiment mode
    parser.add_argument("--patches_file", 
                        help="Path to JSON file with patch specifications")
    parser.add_argument("--layer", 
                        help="Layer to patch (e.g., 'q_net.2')")
    parser.add_argument("--neurons", 
                        help="Comma-separated list of neuron indices to patch")
    
    # Bidirectional mode flag - now defaults to True
    parser.add_argument("--unidirectional", action="store_true", 
                        help="Run unidirectional patching experiment (only clean→corrupted). Default is bidirectional.")
    parser.add_argument("--bidirectional", action="store_true", default=DEFAULT_BIDIRECTIONAL,
                        help=f"Run bidirectional patching experiments (clean→corrupted and corrupted→clean). Default: {DEFAULT_BIDIRECTIONAL}")
    
    # Input file paths
    parser.add_argument("--source_input", 
                        help=f"Source input file (default: {DEFAULT_CLEAN_INPUT})")
    parser.add_argument("--target_input", 
                        help=f"Target input file (default: {DEFAULT_CORRUPTED_INPUT})")
    
    # Activation file paths
    parser.add_argument("--source_activations", 
                        help=f"Source activations file (default: {DEFAULT_CLEAN_ACTIVATIONS})")
    parser.add_argument("--target_activations", 
                        help=f"Target activations file (default: {DEFAULT_CORRUPTED_ACTIVATIONS})")
    
    # Output organization
    parser.add_argument("--organize_by_input", action="store_true", default=DEFAULT_ORGANIZE_BY_INPUT,
                        help=f"Organize results by input ID. Default: {DEFAULT_ORGANIZE_BY_INPUT}")
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX,
                        help=f"Prefix for output files. Default: {DEFAULT_OUTPUT_PREFIX}")
    parser.add_argument("--output_file", 
                        help=f"Output file name (default: {DEFAULT_RESULT_FILENAME})")
    
    # Input IDs to process
    parser.add_argument("--input_ids", 
                        help="Comma-separated list of input IDs to process (if not specified, all inputs are processed)")
    
    # Analysis options
    parser.add_argument("--analyze", action="store_true", default=DEFAULT_ANALYZE_RESULTS,
                        help=f"Analyze results after patching. Default: {DEFAULT_ANALYZE_RESULTS}")
    parser.add_argument("--no_analyze", action="store_false", dest="analyze",
                        help="Don't analyze results after patching.")
    parser.add_argument("--metrics", default=DEFAULT_METRICS,
                        help=f"Comma-separated list of metrics to calculate. Default: {DEFAULT_METRICS}")
    
    return parser.parse_args()

def load_patches_from_file(file_path: str) -> List[Dict[str, Union[List[int], str]]]:
    """
    Load patch configurations from a JSON file.
    
    Args:
        file_path: Path to JSON file with patch configurations
        
    Returns:
        List of patch configurations
    """
    with open(file_path, 'r') as f:
        patches = json.load(f)
    
    print(f"Loaded {len(patches)} patch configurations from {file_path}")
    return patches

def create_patch_config_from_args(layer: str, neurons_str: str) -> Dict[str, Union[List[int], str]]:
    """
    Create a patch configuration from command-line arguments.
    
    Args:
        layer: Layer name
        neurons_str: Comma-separated list of neuron indices
        
    Returns:
        Patch configuration dictionary
    """
    neurons = [int(n.strip()) for n in neurons_str.split(',')]
    return {layer: neurons}

def analyze_patching_results(results_dir: str, metrics: List[str], analyze_subdirs: bool = True) -> None:
    """
    Analyze patching experiment results.
    
    Args:
        results_dir: Directory containing result files
        metrics: List of metrics to calculate
        analyze_subdirs: Whether to analyze subdirectories (denoising, noising)
    """
    print(f"\nAnalyzing patching results in {results_dir}")
    print(f"Calculating metrics: {', '.join(metrics)}")
    
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory not found: {results_dir}")
        return
    
    # Check for denoising and noising subdirectories first
    denoising_dir = os.path.join(results_dir, "denoising")
    noising_dir = os.path.join(results_dir, "noising")
    
    # If bidirectional subdirectories exist, analyze them directly
    if os.path.exists(denoising_dir) and os.path.exists(noising_dir) and analyze_subdirs:
        print("\nDetected bidirectional patching structure. Analyzing subdirectories directly:")
        
        # Process denoising directory
        print(f"\nAnalyzing denoising results in: {denoising_dir}")
        denoising_results = process_result_directory(denoising_dir, metrics)
        denoising_successes = sum(1 for status in denoising_results.values() if status == "success")
        denoising_failures = len(denoising_results) - denoising_successes
        print(f"Processed {len(denoising_results)} files in denoising: "
              f"{denoising_successes} successful, {denoising_failures} failed.")
        
        # Process noising directory
        print(f"\nAnalyzing noising results in: {noising_dir}")
        noising_results = process_result_directory(noising_dir, metrics)
        noising_successes = sum(1 for status in noising_results.values() if status == "success")
        noising_failures = len(noising_results) - noising_successes
        print(f"Processed {len(noising_results)} files in noising: "
              f"{noising_successes} successful, {noising_failures} failed.")
        
        return
    
    # If no bidirectional structure, process the main directory
    main_results = process_result_directory(results_dir, metrics)
    
    # Count successes and failures
    successes = sum(1 for status in main_results.values() if status == "success")
    failures = len(main_results) - successes
    
    print(f"Processed {len(main_results)} files: {successes} successful, {failures} failed.")
    
    # Process subdirectories if requested and not already processed
    if analyze_subdirs:
        subdirs = [d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(results_dir, subdir)
            print(f"\nAnalyzing subdirectory: {subdir_path}")
            
            subdir_results = process_result_directory(subdir_path, metrics)
            
            # Count successes and failures
            subdir_successes = sum(1 for status in subdir_results.values() if status == "success")
            subdir_failures = len(subdir_results) - subdir_successes
            
            print(f"Processed {len(subdir_results)} files in {subdir}: "
                 f"{subdir_successes} successful, {subdir_failures} failed.")
    
    print("\nAnalysis completed.")

def main():
    args = parse_args()
    
    # Initialize patching experiment
    experiment = PatchingExperiment(args.agent_path, device=args.device)
    
    # Set default paths if not provided
    source_input = args.source_input or DEFAULT_CLEAN_INPUT
    target_input = args.target_input or DEFAULT_CORRUPTED_INPUT
    source_activations = args.source_activations or DEFAULT_CLEAN_ACTIVATIONS
    target_activations = args.target_activations or DEFAULT_CORRUPTED_ACTIVATIONS
    output_file = args.output_file or DEFAULT_RESULT_FILENAME
    
    # Parse input IDs if provided
    input_ids = None
    if args.input_ids:
        input_ids = [id.strip() for id in args.input_ids.split(',')]
    
    # Parse metrics for analysis
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    
    # Track if we're doing bidirectional patching
    bidirectional_patching = False
    
    # Determine patching mode and configurations
    if args.patches_file:
        # Multiple patch configurations from file
        patch_configs = load_patches_from_file(args.patches_file)
        
        # Default to bidirectional unless explicitly set to unidirectional
        run_bidirectional = not args.unidirectional
        
        if run_bidirectional:
            # Run bidirectional patching (default behavior)
            print("\nRunning bidirectional patching experiments:")
            print(f"  Agent: {args.agent_path}")
            print(f"  Clean input: {source_input}")
            print(f"  Corrupted input: {target_input}")
            print(f"  Clean activations: {source_activations}")
            print(f"  Corrupted activations: {target_activations}")
            print(f"  Number of patch configurations: {len(patch_configs)}")
            print("--------------------------------------------------")
            
            clean_to_corrupted, corrupted_to_clean = experiment.run_bidirectional_patching(
                source_input,
                target_input,
                source_activations,
                target_activations,
                patch_configs,
                input_ids
            )
            
            # Mark that we did bidirectional patching
            bidirectional_patching = True
            
            print("\nResults:")
            print(f"  Clean → Corrupted: {len(clean_to_corrupted)} result files")
            print(f"  Corrupted → Clean: {len(corrupted_to_clean)} result files")
            
        else:
            # Run unidirectional patching
            experiment_names = [f"patch_{i}" for i in range(len(patch_configs))]
            
            # Run experiments for each patch configuration
            all_results = []
            for i, patch_spec in enumerate(patch_configs):
                print(f"\nRunning patching experiment {i+1}/{len(patch_configs)}:")
                print(f"  Agent: {args.agent_path}")
                print(f"  Target input: {target_input}")
                print(f"  Source activations: {source_activations}")
                print(f"  Patching: {patch_spec}")
                print("--------------------------------------------------")
                
                try:
                    results = experiment.run_patching_experiment(
                        target_input,
                        source_activations,
                        patch_spec,
                        input_ids
                    )
                    all_results.append(results)
                    
                    # Count action changes
                    action_changes = sum(1 for input_id, result in results.items() if result["action_changed"])
                    total_inputs = len(results)
                    print(f"Experiment {i+1} completed.")
                    print(f"Summary: {action_changes}/{total_inputs} actions changed due to patching\n")
                    
                except Exception as e:
                    print(f"Error running experiment {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({})  # Add empty results to maintain index correspondence
            
            # Save results
            if args.organize_by_input:
                output_paths = experiment.save_results_by_input(
                    all_results,
                    patch_configs,
                    experiment_names,
                    args.output_prefix
                )
                print("\nSaving results organized by input ID...")
                for input_id, path in output_paths.items():
                    print(f"Results for input {input_id} saved to {path}")
                print(f"Saved results for {len(output_paths)} inputs")
            else:
                # Save all results in a single file
                all_experiment_results = {}
                for i, (results, patch_spec) in enumerate(zip(all_results, patch_configs)):
                    experiment_name = f"patch_{i}"
                    all_experiment_results[experiment_name] = {
                        "patch_config": patch_spec,
                        "results": results
                    }
                
                output_path = os.path.join(experiment.output_dir, output_file)
                with open(output_path, 'w') as f:
                    json.dump(all_experiment_results, f, indent=2)
                print(f"\nAll results saved to {output_path}")
    
    elif args.layer and args.neurons:
        # Single patch configuration from command-line arguments
        patch_spec = create_patch_config_from_args(args.layer, args.neurons)
        
        # Default to bidirectional unless explicitly set to unidirectional
        run_bidirectional = not args.unidirectional
        
        if run_bidirectional:
            # Bidirectional patching with single patch configuration
            patch_configs = [patch_spec]
            
            print("\nRunning bidirectional patching experiments:")
            print(f"  Agent: {args.agent_path}")
            print(f"  Clean input: {source_input}")
            print(f"  Corrupted input: {target_input}")
            print(f"  Clean activations: {source_activations}")
            print(f"  Corrupted activations: {target_activations}")
            print(f"  Patch configuration: {patch_spec}")
            print("--------------------------------------------------")
            
            clean_to_corrupted, corrupted_to_clean = experiment.run_bidirectional_patching(
                source_input,
                target_input,
                source_activations,
                target_activations,
                patch_configs,
                input_ids
            )
            
            # Mark that we did bidirectional patching
            bidirectional_patching = True
            
            print("\nResults:")
            print(f"  Clean → Corrupted: {len(clean_to_corrupted)} result files")
            print(f"  Corrupted → Clean: {len(corrupted_to_clean)} result files")
        else:
            # Single patch configuration, unidirectional
            print("\nRunning patching experiment with command-line patch configuration:")
            print(f"  Agent: {args.agent_path}")
            print(f"  Target input: {target_input}")
            print(f"  Source activations: {source_activations}")
            print(f"  Patching: {patch_spec}")
            print("--------------------------------------------------")
            
            results = experiment.run_patching_experiment(
                target_input,
                source_activations,
                patch_spec,
                input_ids
            )
            
            # Save results
            output_path = experiment.save_results(results, output_file)
            print(f"Results saved to {output_path}")
    
    else:
        print("Error: You must specify either --patches_file or --layer and --neurons")
        sys.exit(1)
    
    print("\nAll experiments completed.")
    
    # Analyze results if requested
    if args.analyze:
        results_dir = experiment.output_dir
        analyze_patching_results(results_dir, metrics)

if __name__ == "__main__":
    main()
