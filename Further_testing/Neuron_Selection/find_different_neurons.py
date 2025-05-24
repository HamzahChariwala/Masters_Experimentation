#!/usr/bin/env python3
"""
Script: find_different_neurons.py

Analyzes activation differences between clean and corrupted inputs to identify 
neurons that show significant changes in their activations.
"""
import argparse
import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Set, Optional


def load_activations(file_path: str) -> Dict:
    """
    Load activations from a JSON file.
    
    Args:
        file_path: Path to the JSON file with activations
        
    Returns:
        Dictionary of activations
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Activation file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def find_different_neurons(
    clean_activations: Dict, 
    corrupted_activations: Dict,
    layer_name: str,
    input_id: Optional[str] = None,
    top_n: int = 10,
    threshold: float = 0.0
) -> Dict[str, List]:
    """
    Find neurons with the largest activation differences between clean and corrupted inputs.
    
    Args:
        clean_activations: Dictionary of clean activations
        corrupted_activations: Dictionary of corrupted activations
        layer_name: Name of the layer to analyze
        input_id: Specific input ID to analyze (if None, analyze all inputs)
        top_n: Number of top different neurons to return
        threshold: Minimum absolute difference to consider
        
    Returns:
        Dictionary mapping input_id -> list of neuron indices with largest differences
    """
    results = {}
    
    # If a specific input ID is provided, only analyze that one
    input_ids = [input_id] if input_id else clean_activations.keys()
    
    for current_id in input_ids:
        if current_id not in clean_activations:
            print(f"Warning: Input ID {current_id} not found in clean activations. Skipping.")
            continue
            
        if current_id not in corrupted_activations:
            print(f"Warning: Input ID {current_id} not found in corrupted activations. Skipping.")
            continue
            
        if layer_name not in clean_activations[current_id]:
            print(f"Warning: Layer {layer_name} not found in clean activations for {current_id}. Skipping.")
            continue
            
        if layer_name not in corrupted_activations[current_id]:
            print(f"Warning: Layer {layer_name} not found in corrupted activations for {current_id}. Skipping.")
            continue
        
        # Get activations for this input and layer
        clean_values = np.array(clean_activations[current_id][layer_name])
        corrupted_values = np.array(corrupted_activations[current_id][layer_name])
        
        # Calculate absolute differences
        differences = np.abs(clean_values - corrupted_values)
        
        # Get indices of neurons with largest differences
        # Using flatten() and argsort() to handle multi-dimensional arrays
        flat_differences = differences.flatten()
        sorted_indices = np.argsort(flat_differences)
        
        # Get the top N indices with differences above threshold
        top_indices = sorted_indices[-(min(top_n, len(sorted_indices))):]
        
        # Convert flat indices back to original array indices
        # For 2D activations (batch_size, n_neurons), we just need the neuron index
        neuron_indices = []
        for idx in reversed(top_indices):  # Reverse to get largest first
            if flat_differences[idx] > threshold:
                # For 2D array with shape (1, n_neurons), we want the second dimension index
                if clean_values.shape[0] == 1:
                    neuron_idx = idx % clean_values.shape[1]
                else:
                    # For more complex shapes, keep the flat index
                    neuron_idx = idx
                neuron_indices.append(int(neuron_idx))
        
        results[current_id] = neuron_indices
        
        # Also identify neurons that are active in one condition but dead in another
        clean_active = clean_values > 0
        corrupted_active = corrupted_values > 0
        
        # Neurons active in clean but dead in corrupted
        clean_only = np.where(clean_active & ~corrupted_active)
        
        # Neurons active in corrupted but dead in clean
        corrupted_only = np.where(~clean_active & corrupted_active)
        
        # Convert to lists of neuron indices (for 2D arrays with batch size 1)
        if clean_values.shape[0] == 1:
            clean_only_neurons = clean_only[1].tolist() if len(clean_only) > 1 else []
            corrupted_only_neurons = corrupted_only[1].tolist() if len(corrupted_only) > 1 else []
        else:
            clean_only_neurons = clean_only[0].tolist() if len(clean_only) > 0 else []
            corrupted_only_neurons = corrupted_only[0].tolist() if len(corrupted_only) > 0 else []
        
        # Store the ReLU-specific analysis
        results[f"{current_id}_active_in_clean_only"] = clean_only_neurons
        results[f"{current_id}_active_in_corrupted_only"] = corrupted_only_neurons
    
    return results


def print_results(results: Dict[str, List], layer_name: str) -> None:
    """
    Print the analysis results in a readable format.
    
    Args:
        results: Dictionary of results from find_different_neurons
        layer_name: Name of the layer that was analyzed
    """
    print(f"\nAnalysis results for layer: {layer_name}")
    print("=" * 60)
    
    for input_id, neurons in results.items():
        if "_active_in_" in input_id:
            # This is a ReLU analysis result
            continue
            
        print(f"\nInput: {input_id}")
        print("-" * 40)
        
        print(f"Top different neurons: {neurons}")
        
        # Print ReLU analysis
        clean_only_key = f"{input_id}_active_in_clean_only"
        corrupted_only_key = f"{input_id}_active_in_corrupted_only"
        
        if clean_only_key in results:
            print(f"Neurons active in clean but dead in corrupted: {results[clean_only_key]}")
            
        if corrupted_only_key in results:
            print(f"Neurons active in corrupted but dead in clean: {results[corrupted_only_key]}")


def save_results(results: Dict[str, List], output_file: str) -> None:
    """
    Save the analysis results to a JSON file.
    
    Args:
        results: Dictionary of results from find_different_neurons
        output_file: Path to the output file
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


def generate_patch_spec(results: Dict[str, List], layer_name: str) -> str:
    """
    Generate a patch specification string for the most significant neurons.
    
    Args:
        results: Dictionary of results from find_different_neurons
        layer_name: Name of the layer to generate the patch spec for
        
    Returns:
        A patch specification string that can be used with activation_patching.py
    """
    all_neurons = set()
    
    # Collect unique neurons across all inputs
    for input_id, neurons in results.items():
        if "_active_in_" not in input_id:
            all_neurons.update(neurons)
    
    # Generate the patch spec
    if all_neurons:
        neuron_list = ",".join(str(n) for n in sorted(all_neurons))
        return f"{layer_name}:{neuron_list}"
    else:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Analyze activation differences between clean and corrupted inputs.")
    
    parser.add_argument("--clean", type=str, required=True,
                      help="Path to the clean activations JSON file")
    
    parser.add_argument("--corrupted", type=str, required=True,
                      help="Path to the corrupted activations JSON file")
    
    parser.add_argument("--layer", type=str, required=True,
                      help="Name of the layer to analyze (e.g., q_net.2)")
    
    parser.add_argument("--input_id", type=str, default=None,
                      help="Specific input ID to analyze (if not specified, analyze all)")
    
    parser.add_argument("--top_n", type=int, default=10,
                      help="Number of top different neurons to return")
    
    parser.add_argument("--threshold", type=float, default=0.1,
                      help="Minimum absolute difference to consider")
    
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save the results (if not specified, just print to console)")
    
    parser.add_argument("--generate_patch_spec", action="store_true",
                      help="Generate a patch specification for the most significant neurons")
    
    args = parser.parse_args()
    
    try:
        # Load activations
        clean_activations = load_activations(args.clean)
        corrupted_activations = load_activations(args.corrupted)
        
        # Find different neurons
        results = find_different_neurons(
            clean_activations,
            corrupted_activations,
            args.layer,
            args.input_id,
            args.top_n,
            args.threshold
        )
        
        # Print results
        print_results(results, args.layer)
        
        # Save results if an output file is specified
        if args.output:
            save_results(results, args.output)
        
        # Generate patch spec if requested
        if args.generate_patch_spec:
            patch_spec = generate_patch_spec(results, args.layer)
            print(f"\nSuggested patch specification for significant neurons:")
            print(f"--patch_spec \"{patch_spec}\"")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 