#!/usr/bin/env python3
"""
Extract top weights from candidate neurons based on gradient magnitudes.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Any, List, Tuple


def load_useful_neurons(definitions_path: str) -> List[Dict[str, List[int]]]:
    """Load the useful_neurons.json mapping."""
    with open(definitions_path, 'r') as f:
        return json.load(f)


def parse_exp_id(exp_id: str, useful_neurons: List[Dict[str, List[int]]]) -> Tuple[str, int]:
    """
    Parse exp_n_layer format to extract layer name and neuron index.
    
    Args:
        exp_id: e.g., "exp_75_q_net.2"
        useful_neurons: List of experiment definitions
        
    Returns:
        Tuple of (layer_name, neuron_index)
    """
    # Extract experiment number from exp_id
    parts = exp_id.split('_')
    exp_num = int(parts[1])  # exp_75 -> 75
    layer_suffix = '_'.join(parts[2:])  # q_net.2
    
    # Convert to 0-based indexing (useful_neurons uses 1-based in the name)
    exp_idx = exp_num - 1
    
    if exp_idx < 0 or exp_idx >= len(useful_neurons):
        raise ValueError(f"Experiment index {exp_num} out of range (1-{len(useful_neurons)})")
    
    # Get the experiment definition
    exp_def = useful_neurons[exp_idx]
    
    # Find the layer that matches the suffix
    layer_name = None
    neuron_indices = None
    
    for layer, indices in exp_def.items():
        if layer == layer_suffix:
            layer_name = layer
            neuron_indices = indices
            break
    
    if layer_name is None:
        raise ValueError(f"Layer {layer_suffix} not found in experiment {exp_num}")
    
    # For weight layers, we need to convert to the weight parameter name
    if layer_name.startswith('q_net.'):
        weight_layer_name = f"{layer_name}.weight"
    elif layer_name.startswith('features_extractor.'):
        weight_layer_name = f"{layer_name}.weight"
    else:
        weight_layer_name = f"{layer_name}.weight"
    
    # Assuming single neuron per experiment for now
    neuron_idx = neuron_indices[0] if neuron_indices else 0
    
    return weight_layer_name, neuron_idx


def extract_candidate_weights(cross_metric_path: str, average_gradients_path: str, 
                            useful_neurons_path: str, k: int = 1, m: int = 10) -> Dict[str, Any]:
    """
    Extract candidate weights based on gradient magnitudes.
    
    Args:
        cross_metric_path: Path to cross_metric_summary.json
        average_gradients_path: Path to average_gradients.json
        useful_neurons_path: Path to useful_neurons.json
        k: Top k weights per neuron
        m: Additional weights to extract globally (not total)
        
    Returns:
        Dictionary with candidate weights and their indices
    """
    # Load data
    with open(cross_metric_path, 'r') as f:
        cross_metric = json.load(f)
    
    with open(average_gradients_path, 'r') as f:
        avg_gradients = json.load(f)
    
    useful_neurons = load_useful_neurons(useful_neurons_path)
    
    print(f"Processing {len(cross_metric['common_neurons'])} candidate neurons...")
    
    # Extract candidate neurons and their gradients
    candidate_neurons = []
    neuron_weights = {}
    
    for exp_id in cross_metric['common_neurons'].keys():
        try:
            layer_name, neuron_idx = parse_exp_id(exp_id, useful_neurons)
            
            # Get gradients for this neuron
            if layer_name in avg_gradients['average_gradients']:
                neuron_grads = np.array(avg_gradients['average_gradients'][layer_name][neuron_idx])
                
                candidate_neurons.append({
                    'exp_id': exp_id,
                    'layer_name': layer_name,
                    'neuron_index': neuron_idx,
                    'gradients': neuron_grads
                })
                
                neuron_weights[exp_id] = {
                    'layer_name': layer_name,
                    'neuron_index': neuron_idx,
                    'weights': neuron_grads.tolist()
                }
                
                print(f"  {exp_id} -> {layer_name}[{neuron_idx}] (shape: {neuron_grads.shape})")
                
        except Exception as e:
            print(f"  Warning: Could not process {exp_id}: {e}")
    
    # Find top k weights per neuron
    top_k_per_neuron = {}
    remaining_weights = []  # (magnitude, exp_id, weight_idx)
    weight_targets = []  # Final targeting information
    
    for neuron in candidate_neurons:
        exp_id = neuron['exp_id']
        gradients = neuron['gradients']
        layer_name = neuron['layer_name']
        neuron_idx = neuron['neuron_index']
        
        # Get top k indices for this neuron
        top_k_indices = np.argsort(gradients)[-k:][::-1]  # Descending order
        top_k_values = gradients[top_k_indices]
        
        top_k_per_neuron[exp_id] = {
            'indices': top_k_indices.tolist(),
            'values': top_k_values.tolist()
        }
        
        # Add to targeting information
        for i, (weight_idx, magnitude) in enumerate(zip(top_k_indices, top_k_values)):
            weight_targets.append({
                'layer_name': layer_name,
                'neuron_index': neuron_idx,
                'weight_index': int(weight_idx),
                'gradient_magnitude': float(magnitude),
                'source': f'top_{i+1}_of_{exp_id}',
                'priority': f'guaranteed_k_{i+1}'
            })
        
        # Add remaining weights to global pool
        remaining_indices = np.argsort(gradients)[:-k]  # All except top k
        for idx in remaining_indices:
            remaining_weights.append((gradients[idx], exp_id, int(idx), layer_name, neuron_idx))
    
    # Sort remaining weights globally and take top m additional weights
    remaining_weights.sort(reverse=True, key=lambda x: x[0])
    top_remaining = remaining_weights[:m]
    
    # Organize top remaining weights by neuron
    additional_weights = {}
    for magnitude, exp_id, weight_idx, layer_name, neuron_idx in top_remaining:
        if exp_id not in additional_weights:
            additional_weights[exp_id] = {'indices': [], 'values': []}
        additional_weights[exp_id]['indices'].append(weight_idx)
        additional_weights[exp_id]['values'].append(magnitude)
        
        # Add to targeting information
        weight_targets.append({
            'layer_name': layer_name,
            'neuron_index': neuron_idx,
            'weight_index': weight_idx,
            'gradient_magnitude': magnitude,
            'source': f'additional_from_{exp_id}',
            'priority': 'additional_global'
        })
    
    # Sort weight targets by magnitude for easier interpretation
    weight_targets.sort(key=lambda x: x['gradient_magnitude'], reverse=True)
    
    return {
        'weight_targets': weight_targets,
        'metadata': {
            'k_per_neuron': k,
            'additional_m': m,
            'num_candidate_neurons': len(candidate_neurons),
            'total_weights_selected': len(weight_targets),
            'guaranteed_per_neuron': k * len(candidate_neurons),
            'additional_selected': len(top_remaining),
            'description': f'Top {k} weights per neuron plus top {m} additional weights globally'
        },
        'neuron_weights': neuron_weights,
        'top_k_per_neuron': top_k_per_neuron,
        'additional_weights': additional_weights
    }


def save_candidate_weights(results: Dict[str, Any], output_path: str) -> None:
    """Save candidate weights to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Candidate weights saved to: {output_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract candidate weights from gradient analysis")
    parser.add_argument("--cross_metric_path", type=str, required=True,
                       help="Path to cross_metric_summary.json")
    parser.add_argument("--average_gradients_path", type=str, required=True,
                       help="Path to average_gradients.json")
    parser.add_argument("--useful_neurons_path", type=str, 
                       default="Neuron_Selection/ExperimentTooling/Definitions/useful_neurons.json",
                       help="Path to useful_neurons.json")
    parser.add_argument("--k", type=int, default=1,
                       help="Top k weights per neuron")
    parser.add_argument("--m", type=int, default=10,
                       help="Additional weights to extract globally")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.average_gradients_path)
    
    print(f"Extracting candidate weights...")
    print(f"  Cross metric: {args.cross_metric_path}")
    print(f"  Average gradients: {args.average_gradients_path}")
    print(f"  Useful neurons: {args.useful_neurons_path}")
    print(f"  k={args.k}, m={args.m}")
    
    # Extract candidate weights
    results = extract_candidate_weights(
        args.cross_metric_path, 
        args.average_gradients_path,
        args.useful_neurons_path,
        args.k, 
        args.m
    )
    
    # Save results
    output_path = os.path.join(args.output_dir, "candidate_weights.json")
    save_candidate_weights(results, output_path)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Candidate neurons: {results['metadata']['num_candidate_neurons']}")
    print(f"  Top {args.k} weights per neuron: {results['metadata']['guaranteed_per_neuron']}")
    print(f"  Additional top weights: {results['metadata']['additional_selected']}")
    print(f"  Total weights selected: {results['metadata']['total_weights_selected']}")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    main() 