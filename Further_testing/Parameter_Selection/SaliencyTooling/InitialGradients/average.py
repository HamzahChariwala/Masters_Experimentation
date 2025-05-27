#!/usr/bin/env python3
"""
Average gradient magnitude computation tool.
Computes average magnitude of gradients across all input examples for each neuron.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Any


def compute_average_gradients(gradients_path: str) -> Dict[str, Any]:
    """
    Compute average gradient magnitudes across all input examples.
    
    Args:
        gradients_path: Path to weight_gradients.json file
        
    Returns:
        Dictionary containing average gradient magnitudes for each layer
    """
    # Load gradient data
    with open(gradients_path, 'r') as f:
        gradient_data = json.load(f)
    
    print(f"Processing gradients from {len(gradient_data)} input examples...")
    
    # Initialize storage for accumulated gradients
    accumulated_gradients = {}
    example_count = len(gradient_data)
    
    # Process each input example
    for input_id, input_result in gradient_data.items():
        gradients = input_result['gradients']
        
        for layer_name, layer_gradients in gradients.items():
            if layer_name not in accumulated_gradients:
                accumulated_gradients[layer_name] = []
            
            # Convert to numpy array and take absolute value
            grad_array = np.abs(np.array(layer_gradients))
            accumulated_gradients[layer_name].append(grad_array)
    
    # Compute averages
    average_gradients = {}
    
    for layer_name, gradient_list in accumulated_gradients.items():
        print(f"  Processing layer: {layer_name}")
        
        # Stack all gradients for this layer and compute mean across examples
        stacked_gradients = np.stack(gradient_list, axis=0)  # Shape: (n_examples, ...)
        average_magnitude = np.mean(stacked_gradients, axis=0)  # Average across examples
        
        # Convert back to list for JSON serialization
        average_gradients[layer_name] = average_magnitude.tolist()
    
    return {
        'metadata': {
            'num_examples': example_count,
            'description': 'Average gradient magnitudes across all input examples'
        },
        'average_gradients': average_gradients
    }


def save_average_gradients(results: Dict[str, Any], output_path: str) -> None:
    """Save average gradient results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Average gradients saved to: {output_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute average gradient magnitudes")
    parser.add_argument("--gradients_path", type=str, required=True,
                       help="Path to weight_gradients.json file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for average gradients")
    
    args = parser.parse_args()
    
    # Set default output directory (same as input file directory)
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.gradients_path)
    
    print(f"Computing average gradients from: {args.gradients_path}")
    
    # Compute average gradients
    results = compute_average_gradients(args.gradients_path)
    
    # Save results
    output_path = os.path.join(args.output_dir, "average_gradients.json")
    save_average_gradients(results, output_path)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Processed {results['metadata']['num_examples']} input examples")
    print(f"  Computed averages for {len(results['average_gradients'])} layers")
    print(f"  Output saved to: {output_path}")
    
    # Show layer shapes
    for layer_name, avg_grads in results['average_gradients'].items():
        avg_array = np.array(avg_grads)
        print(f"    {layer_name}: {avg_array.shape}")


if __name__ == "__main__":
    main() 