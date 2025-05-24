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

from Neuron_Selection.PatchingTooling import PatchingExperiment


def parse_patch_spec(patch_spec_str: str) -> Dict[str, Union[List[int], str]]:
    """
    Parse the patch specification string.
    
    Format: "layer1:neuron1,neuron2;layer2:all"
    
    Args:
        patch_spec_str: String specification of layers and neurons to patch
        
    Returns:
        Dictionary mapping layer names to patching specifications
    """
    patch_spec = {}
    
    if not patch_spec_str:
        return patch_spec
    
    # Split by semicolon to get layer specifications
    layer_specs = patch_spec_str.split(';')
    
    for layer_spec in layer_specs:
        if not layer_spec:
            continue
            
        # Split by colon to get layer name and neuron indices
        parts = layer_spec.split(':')
        if len(parts) != 2:
            print(f"Warning: Invalid layer specification: {layer_spec}")
            continue
            
        layer_name = parts[0].strip()
        neuron_spec = parts[1].strip()
        
        # Check if patching all neurons
        if neuron_spec.lower() == 'all':
            patch_spec[layer_name] = 'all'
        else:
            # Parse neuron indices
            try:
                neuron_indices = [int(idx) for idx in neuron_spec.split(',')]
                patch_spec[layer_name] = neuron_indices
            except ValueError:
                print(f"Warning: Invalid neuron indices in {layer_spec}")
                continue
    
    return patch_spec


def load_patches_from_file(file_path: str) -> List[Dict[str, Union[List[int], str]]]:
    """
    Load patch configurations from a JSON file.
    
    Args:
        file_path: Path to the JSON file with patch configurations
        
    Returns:
        List of patch specifications
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Patches file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        patches = json.load(f)
    
    # Validate the loaded patches
    if not isinstance(patches, list):
        raise ValueError("Patches file must contain a JSON array of patch configurations")
    
    return patches


def generate_output_filename(output_prefix: str, patch_index: int, patch_spec: Dict[str, Union[List[int], str]]) -> str:
    """
    Generate a descriptive output filename based on the patch configuration.
    
    Args:
        output_prefix: Base prefix for the output file
        patch_index: Index of the current patch in the batch
        patch_spec: Current patch specification
        
    Returns:
        Descriptive output filename
    """
    # Create a string representation of the patched layers
    patched_layers = []
    
    for layer_name, neurons in patch_spec.items():
        if neurons == "all":
            patched_layers.append(f"{layer_name}_all")
        else:
            neuron_count = len(neurons)
            patched_layers.append(f"{layer_name}_{neuron_count}n")
    
    layer_str = "-".join(patched_layers)
    
    return f"{output_prefix}_exp{patch_index+1}_{layer_str}.json"


def main():
    parser = argparse.ArgumentParser(
        description="Perform activation patching experiments on trained agents.")
    
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory (e.g., Agent_Storage/SpawnTests/biased/biased-v1)")
    
    parser.add_argument("--target_input", type=str, required=True,
                       help="Target input file to patch (e.g., corrupted_inputs.json)")
    
    parser.add_argument("--source_activations", type=str, required=True,
                       help="Source activation file to patch from (e.g., clean_activations.npz)")
    
    # Make patch_spec optional since we now have patches_file as an alternative
    parser.add_argument("--patch_spec", type=str, default=None,
                       help="Specification of layers and neurons to patch. Format: 'layer1:neuron1,neuron2;layer2:all'")
    
    # Add new argument for patches file
    parser.add_argument("--patches_file", type=str, default=None,
                       help="Path to JSON file containing multiple patch configurations")
    
    parser.add_argument("--input_ids", type=str, default=None,
                       help="Comma-separated list of input IDs to process (if not specified, process all)")
    
    parser.add_argument("--output_prefix", type=str, default="patching_result",
                       help="Prefix for output files")
    
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Ensure either patch_spec or patches_file is provided
    if not args.patch_spec and not args.patches_file:
        print("Error: Either --patch_spec or --patches_file must be provided.")
        return 1
    
    # Load patches from file or parse patch_spec
    if args.patches_file:
        try:
            patches = load_patches_from_file(args.patches_file)
            print(f"Loaded {len(patches)} patch configurations from {args.patches_file}")
        except Exception as e:
            print(f"Error loading patches file: {e}")
            return 1
    else:
        # Use single patch specification
        patches = [parse_patch_spec(args.patch_spec)]
    
    # Parse input IDs if provided
    input_ids = None
    if args.input_ids:
        input_ids = [id.strip() for id in args.input_ids.split(',')]
    
    # Create experiment runner
    experiment = PatchingExperiment(args.agent_path, device=args.device)
    
    # Process each patch configuration
    for i, patch_spec in enumerate(patches):
        if not patch_spec:
            print(f"Warning: Empty patch specification at index {i}. Skipping.")
            continue
        
        # Create descriptive output filename
        output_file = generate_output_filename(args.output_prefix, i, patch_spec)
        
        # Print experiment details
        print(f"\nRunning patching experiment {i+1}/{len(patches)}:")
        print(f"  Agent: {args.agent_path}")
        print(f"  Target input: {args.target_input}")
        print(f"  Source activations: {args.source_activations}")
        print(f"  Patching: {patch_spec}")
        print(f"  Output file: {output_file}")
        print("-" * 50)
        
        try:
            # Run experiment
            results = experiment.run_patching_experiment(
                args.target_input,
                args.source_activations,
                patch_spec,
                input_ids
            )
            
            # Add patch configuration to results for reference
            for input_id in results:
                results[input_id]["patch_configuration"] = patch_spec
            
            # Save results
            experiment.save_results(results, output_file)
            
            # Print summary
            changed_count = sum(1 for result in results.values() if result.get("action_changed", False))
            total_count = len(results)
            
            print(f"\nExperiment {i+1} completed. Results saved to {experiment.output_dir}/{output_file}")
            print(f"Summary: {changed_count}/{total_count} actions changed due to patching")
            
        except Exception as e:
            print(f"Error running experiment {i+1}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next patch
            continue
    
    print(f"\nAll experiments completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
