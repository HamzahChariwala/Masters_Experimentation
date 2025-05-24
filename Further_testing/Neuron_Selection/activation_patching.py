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


def main():
    parser = argparse.ArgumentParser(
        description="Perform activation patching experiments on trained agents.")
    
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory (e.g., Agent_Storage/SpawnTests/biased/biased-v1)")
    
    parser.add_argument("--target_input", type=str, required=True,
                       help="Target input file to patch (e.g., corrupted_inputs.json)")
    
    parser.add_argument("--source_activations", type=str, required=True,
                       help="Source activation file to patch from (e.g., clean_activations.npz)")
    
    parser.add_argument("--patch_spec", type=str, required=True,
                       help="Specification of layers and neurons to patch. Format: 'layer1:neuron1,neuron2;layer2:all'")
    
    parser.add_argument("--input_ids", type=str, default=None,
                       help="Comma-separated list of input IDs to process (if not specified, process all)")
    
    parser.add_argument("--output_prefix", type=str, default="patching_result",
                       help="Prefix for output files")
    
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Parse patch specification
    patch_spec = parse_patch_spec(args.patch_spec)
    if not patch_spec:
        print("Error: No valid patch specification provided.")
        return 1
    
    # Parse input IDs if provided
    input_ids = None
    if args.input_ids:
        input_ids = [id.strip() for id in args.input_ids.split(',')]
    
    # Create experiment runner
    experiment = PatchingExperiment(args.agent_path, device=args.device)
    
    # Create descriptive output filename
    # Extract base names without extensions
    target_base = os.path.splitext(args.target_input)[0]
    source_base = os.path.splitext(args.source_activations)[0]
    
    # Create list of patched layers for filename
    patched_layers = '-'.join(patch_spec.keys())
    
    output_file = f"{args.output_prefix}_{target_base}_from_{source_base}_{patched_layers}.json"
    
    # Run the experiment
    print(f"\nRunning patching experiment:")
    print(f"  Agent: {args.agent_path}")
    print(f"  Target input: {args.target_input}")
    print(f"  Source activations: {args.source_activations}")
    print(f"  Patching: {args.patch_spec}")
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
        
        # Save results
        experiment.save_results(results, output_file)
        
        # Print summary
        changed_count = sum(1 for result in results.values() if result.get("action_changed", False))
        total_count = len(results)
        
        print(f"\nExperiment completed. Results saved to {experiment.output_dir}/{output_file}")
        print(f"Summary: {changed_count}/{total_count} actions changed due to patching")
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
