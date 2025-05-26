#!/usr/bin/env python3
"""
Script to filter patching results based on specified criteria.
Currently implements filtering for non-zero chebyshev ratio.

Usage:
    python filter_patching_results.py --agent_path <path_to_agent> --criterion chebyshev_ratio_nonzero

Example:
    python filter_patching_results.py --agent_path Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6 --criterion chebyshev_ratio_nonzero
"""

import os
import json
import argparse
import datetime
from typing import Dict, List, Callable, Any, Tuple
import re

# Define available criteria functions
def chebyshev_ratio_nonzero(experiment_data: Dict[str, Any], experiment_type: str) -> bool:
    """
    Check if the chebyshev ratio is non-zero.
    
    Args:
        experiment_data: Data for a single experiment
        experiment_type: Either 'noising' or 'denoising'
        
    Returns:
        True if the chebyshev ratio is non-zero, False otherwise
    """
    if 'metrics' not in experiment_data:
        return False
    
    if 'chebyshev_ratio' not in experiment_data['metrics']:
        return False
    
    chebyshev_data = experiment_data['metrics']['chebyshev_ratio']
    
    # Check if the mean value is non-zero
    if 'mean' in chebyshev_data and chebyshev_data['mean'] != 0:
        return True
    
    return False

def chebyshev_ratio_significant(experiment_data: Dict[str, Any], experiment_type: str) -> bool:
    """
    Check if the chebyshev ratio is non-zero and has a magnitude of at least 1
    in either its max or min value.
    
    Args:
        experiment_data: Data for a single experiment
        experiment_type: Either 'noising' or 'denoising'
        
    Returns:
        True if the chebyshev ratio is non-zero and has significant magnitude, False otherwise
    """
    if 'metrics' not in experiment_data:
        return False
    
    if 'chebyshev_ratio' not in experiment_data['metrics']:
        return False
    
    chebyshev_data = experiment_data['metrics']['chebyshev_ratio']
    
    # Check if the mean value is non-zero
    if 'mean' not in chebyshev_data or chebyshev_data['mean'] == 0:
        return False
    
    # Check if at least one of max or min has magnitude >= 1
    has_significant_magnitude = False
    
    if 'max' in chebyshev_data and abs(chebyshev_data['max']) >= 1:
        has_significant_magnitude = True
    
    if 'min' in chebyshev_data and abs(chebyshev_data['min']) >= 1:
        has_significant_magnitude = True
    
    return has_significant_magnitude

# Dictionary of available criteria functions
CRITERIA_FUNCTIONS = {
    'chebyshev_ratio_nonzero': chebyshev_ratio_nonzero,
    'chebyshev_ratio_significant': chebyshev_ratio_significant
}

def extract_neuron_info(experiment_id: str) -> Tuple[str, int]:
    """
    Extract layer name and neuron index from an experiment ID.
    
    Args:
        experiment_id: ID in the format 'exp_N_layer_name'
        
    Returns:
        Tuple of (layer_name, neuron_index_in_layer)
    """
    # Extract the experiment number and layer name
    match = re.match(r'exp_(\d+)_(.+)', experiment_id)
    if not match:
        return None, None
    
    exp_num = int(match.group(1))
    layer_name = match.group(2)
    
    # Find the first experiment with this layer
    # We'll need to scan all experiments to find this
    return layer_name, exp_num

def find_layer_start_indices(experiments: Dict[str, Any]) -> Dict[str, int]:
    """
    Find the starting experiment index for each layer.
    
    Args:
        experiments: Dictionary of all experiments
        
    Returns:
        Dictionary mapping layer names to their first experiment index
    """
    layer_starts = {}
    
    for exp_id in experiments.keys():
        match = re.match(r'exp_(\d+)_(.+)', exp_id)
        if match:
            exp_num = int(match.group(1))
            layer_name = match.group(2)
            
            if layer_name not in layer_starts or exp_num < layer_starts[layer_name]:
                layer_starts[layer_name] = exp_num
    
    return layer_starts

def calculate_neuron_indices(experiments: Dict[str, Any]) -> Dict[str, Tuple[str, int]]:
    """
    Calculate the neuron index within each layer for all experiments.
    
    Args:
        experiments: Dictionary of all experiments
        
    Returns:
        Dictionary mapping experiment IDs to tuples of (layer_name, neuron_index)
    """
    # First find the starting index for each layer
    layer_starts = find_layer_start_indices(experiments)
    
    # Then calculate the neuron index for each experiment
    neuron_indices = {}
    
    for exp_id in experiments.keys():
        match = re.match(r'exp_(\d+)_(.+)', exp_id)
        if match:
            exp_num = int(match.group(1))
            layer_name = match.group(2)
            
            # Calculate the neuron index (0-based)
            neuron_index = exp_num - layer_starts[layer_name]
            neuron_indices[exp_id] = (layer_name, neuron_index)
    
    return neuron_indices

def count_neurons_per_layer(experiments: Dict[str, Any]) -> Dict[str, int]:
    """
    Count the total number of neurons in each layer based on experiment IDs.
    
    Args:
        experiments: Dictionary of all experiments
        
    Returns:
        Dictionary mapping layer names to neuron counts
    """
    # Calculate neuron indices
    neuron_indices = calculate_neuron_indices(experiments)
    
    # Count maximum neuron index for each layer
    layer_counts = {}
    
    for exp_id, (layer_name, neuron_index) in neuron_indices.items():
        if layer_name not in layer_counts or neuron_index > layer_counts[layer_name]:
            layer_counts[layer_name] = neuron_index
    
    # Add 1 to each count to convert from 0-based index to count
    return {layer: count + 1 for layer, count in layer_counts.items()}

def filter_experiments(
    noising_data: Dict[str, Any],
    denoising_data: Dict[str, Any],
    criterion_func: Callable[[Dict[str, Any], str], bool],
    same_criterion: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter experiments based on the provided criterion function.
    
    Args:
        noising_data: Noising experiments data
        denoising_data: Denoising experiments data
        criterion_func: Function to apply to each experiment
        same_criterion: Whether to apply the same criterion to both noising and denoising
        
    Returns:
        Dictionary with 'noising' and 'denoising' keys, each containing a list of passing experiments
    """
    # Calculate neuron indices for each experiment
    noising_indices = calculate_neuron_indices(noising_data)
    denoising_indices = calculate_neuron_indices(denoising_data)
    
    # Filter the experiments
    noising_passed = []
    denoising_passed = []
    
    for exp_id, exp_data in noising_data.items():
        if criterion_func(exp_data, 'noising'):
            if exp_id in noising_indices:
                layer_name, neuron_idx = noising_indices[exp_id]
                noising_passed.append({
                    'experiment_id': exp_id,
                    'layer': layer_name,
                    'neuron_index': neuron_idx
                })
    
    for exp_id, exp_data in denoising_data.items():
        if criterion_func(exp_data, 'denoising'):
            if exp_id in denoising_indices:
                layer_name, neuron_idx = denoising_indices[exp_id]
                denoising_passed.append({
                    'experiment_id': exp_id,
                    'layer': layer_name,
                    'neuron_index': neuron_idx
                })
    
    return {
        'noising': noising_passed,
        'denoising': denoising_passed
    }

def save_filtered_results(
    results: Dict[str, List[Dict[str, Any]]], 
    output_path: str,
    noising_data: Dict[str, Any],
    denoising_data: Dict[str, Any]
):
    """
    Save the filtered results to a JSON file.
    
    Args:
        results: Filtered results to save
        output_path: Path to save the results to
        noising_data: Complete noising data for summary
        denoising_data: Complete denoising data for summary
    """
    # Count neurons per layer for both studies
    noising_counts = count_neurons_per_layer(noising_data)
    denoising_counts = count_neurons_per_layer(denoising_data)
    
    # Add summary to the results
    output_results = {
        'summary': {
            'noising': {
                'total_neurons': sum(noising_counts.values()),
                'neurons_per_layer': noising_counts,
                'passing_neurons': len(results['noising'])
            },
            'denoising': {
                'total_neurons': sum(denoising_counts.values()),
                'neurons_per_layer': denoising_counts,
                'passing_neurons': len(results['denoising'])
            }
        },
        'noising': results['noising'],
        'denoising': results['denoising']
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_results, f, indent=2)
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Filter patching results based on criteria')
    parser.add_argument('--agent_path', type=str, required=True,
                      help='Path to the agent directory')
    parser.add_argument('--criterion', type=str, default='chebyshev_ratio_nonzero',
                      choices=list(CRITERIA_FUNCTIONS.keys()),
                      help='Criterion to use for filtering')
    parser.add_argument('--different_criteria', action='store_true',
                      help='Apply different criteria to noising and denoising')
    args = parser.parse_args()
    
    # Construct paths
    agent_path = args.agent_path
    patching_results_dir = os.path.join(agent_path, 'patching_results')
    noising_path = os.path.join(patching_results_dir, 'patching_summary_noising.json')
    denoising_path = os.path.join(patching_results_dir, 'patching_summary_denoising.json')
    output_dir = os.path.join(agent_path, 'patching_selection')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamped output file name
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'filtered_results_{args.criterion}_{timestamp}.json')
    
    # Load the patching summaries
    print(f"Loading patching summaries from {patching_results_dir}...")
    with open(noising_path, 'r') as f:
        noising_data = json.load(f)
    
    with open(denoising_path, 'r') as f:
        denoising_data = json.load(f)
    
    # Get the criterion function
    criterion_func = CRITERIA_FUNCTIONS[args.criterion]
    
    # Filter the experiments
    print(f"Filtering experiments using criterion: {args.criterion}")
    results = filter_experiments(
        noising_data, 
        denoising_data, 
        criterion_func, 
        not args.different_criteria
    )
    
    # Print summary
    print(f"Found {len(results['noising'])} passing noising experiments")
    print(f"Found {len(results['denoising'])} passing denoising experiments")
    
    # Save the results
    save_filtered_results(results, output_path, noising_data, denoising_data)

if __name__ == "__main__":
    main() 