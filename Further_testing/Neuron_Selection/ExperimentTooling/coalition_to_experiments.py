#!/usr/bin/env python3
"""
Script to convert coalition JSON files to experiment format compatible with analyze_metric.py.

This script processes coalition JSON files from circuit_verification/descriptions and converts
them into the format expected by analyze_metric.py, mapping neuron names to actual neuron
definitions using the all_neurons.json file.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_all_neurons() -> List[Dict[str, Any]]:
    """
    Load the all_neurons.json file to get the neuron mapping.
    
    Returns:
        List of neuron definitions
    """
    all_neurons_path = Path(__file__).parent / "Definitions" / "all_neurons.json"
    
    try:
        with open(all_neurons_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading all_neurons.json: {e}")
        return []


def neuron_name_to_neuron_def(neuron_name: str, all_neurons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert a neuron name to a neuron definition.
    
    Args:
        neuron_name: String like "q_net.4_neuron_3" or "features_extractor.mlp.0_neuron_15"
        all_neurons: List of neuron definitions from all_neurons.json
        
    Returns:
        Neuron definition dictionary
    """
    # For the new naming convention, we need to find the matching neuron definition
    # by searching through all_neurons for a definition that would generate this name
    try:
        # Parse the neuron name to extract layer and index
        if "_neuron_" in neuron_name:
            layer_part, index_part = neuron_name.rsplit("_neuron_", 1)
            neuron_index = int(index_part)
            
            # Search for a matching definition in all_neurons
            for neuron_def in all_neurons:
                # neuron_def is like {"q_net.0": [35]} or {"features_extractor.mlp.0": [15]}
                for layer_name, neuron_list in neuron_def.items():
                    if layer_name == layer_part and neuron_index in neuron_list:
                        return neuron_def
            
            # If not found in existing definitions, create a new one
            # This handles the case where we have individual neuron names
            return {layer_part: [neuron_index]}
        else:
            # Handle old exp_N format if present (for backward compatibility)
            if neuron_name.startswith('exp_'):
                parts = neuron_name.split('_')
                if len(parts) >= 2:
                    exp_number = int(parts[1])
                    # Convert to 0-based index (exp_1 -> index 0)
                    index = exp_number - 1
                    
                    if 0 <= index < len(all_neurons):
                        return all_neurons[index]
                    else:
                        print(f"Warning: Index {index} out of range for all_neurons (length: {len(all_neurons)})")
                        return {}
            
            print(f"Warning: Could not parse experiment string: {neuron_name}")
            return {}
    except (ValueError, IndexError) as e:
        print(f"Warning: Error parsing experiment string {neuron_name}: {e}")
        return {}


def convert_coalition_to_experiments(coalition_file: Path, all_neurons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a coalition JSON file to experiment format.
    
    Args:
        coalition_file: Path to the coalition JSON file
        all_neurons: List of neuron definitions
        
    Returns:
        List of patch configurations (each is a dict mapping layer names to neuron lists)
    """
    try:
        with open(coalition_file, 'r') as f:
            coalition_data = json.load(f)
    except Exception as e:
        print(f"Error loading coalition file {coalition_file}: {e}")
        return []
    
    experiments = []
    
    # Process each cumulative experiment in order
    if 'experiments' in coalition_data:
        # Sort by experiment name to ensure proper order (cumulative_01, cumulative_02, etc.)
        sorted_experiments = sorted(coalition_data['experiments'].items(), 
                                   key=lambda x: x[0])
        
        for exp_name, exp_data in sorted_experiments:
            if 'included_experiments' in exp_data:
                # Convert each neuron name to neuron definition and group by layer
                layer_neurons = {}
                
                for neuron_name in exp_data['included_experiments']:
                    neuron_def = neuron_name_to_neuron_def(neuron_name, all_neurons)
                    if neuron_def:
                        # neuron_def is like {"q_net.0": [35]} or {"features_extractor.mlp.0": [15]}
                        for layer_name, neuron_list in neuron_def.items():
                            if layer_name not in layer_neurons:
                                layer_neurons[layer_name] = []
                            layer_neurons[layer_name].extend(neuron_list)
                
                # Remove duplicates and sort neuron indices for each layer
                for layer_name in layer_neurons:
                    layer_neurons[layer_name] = sorted(list(set(layer_neurons[layer_name])))
                
                if layer_neurons:
                    experiments.append(layer_neurons)
    
    return experiments


def process_coalition_files(descriptions_dir: Path, output_dir: Path, all_neurons: List[Dict[str, Any]]) -> None:
    """
    Process all coalition files in the descriptions directory.
    
    Args:
        descriptions_dir: Directory containing coalition JSON files
        output_dir: Directory to save experiment files
        all_neurons: List of neuron definitions
    """
    # Find all coalition files
    coalition_files = list(descriptions_dir.glob("coalitions_*.json"))
    
    if not coalition_files:
        print(f"No coalition files found in {descriptions_dir}")
        return
    
    print(f"Processing {len(coalition_files)} coalition files:")
    
    successful = 0
    for coalition_file in sorted(coalition_files):
        print(f"  Processing {coalition_file.name}...")
        
        # Convert coalition to experiments
        experiments = convert_coalition_to_experiments(coalition_file, all_neurons)
        
        if experiments:
            # Create output filename (replace 'coalitions_' with 'experiments_')
            output_name = coalition_file.name.replace('coalitions_', 'experiments_')
            output_file = output_dir / output_name
            
            # Save experiments
            try:
                with open(output_file, 'w') as f:
                    json.dump(experiments, f, indent=2)
                print(f"    -> {output_name} ({len(experiments)} experiments)")
                successful += 1
            except Exception as e:
                print(f"    Error saving {output_name}: {e}")
        else:
            print(f"    No valid experiments found in {coalition_file.name}")
    
    print(f"\nCompleted: {successful}/{len(coalition_files)} files processed successfully")


def main():
    parser = argparse.ArgumentParser(description="Convert coalition JSONs to experiment format")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    
    args = parser.parse_args()
    
    # Set up paths
    agent_path = Path(args.agent_path)
    descriptions_dir = agent_path / "circuit_verification" / "descending" / "descriptions"
    experiments_dir = agent_path / "circuit_verification" / "descending" / "experiments"
    
    # Create experiments directory
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if descriptions directory exists
    if not descriptions_dir.exists():
        print(f"Error: Descriptions directory not found: {descriptions_dir}")
        return 1
    
    # Load all neurons mapping
    print("Loading all neurons mapping...")
    all_neurons = load_all_neurons()
    
    if not all_neurons:
        print("Error: Could not load all neurons mapping")
        return 1
    
    print(f"Loaded {len(all_neurons)} neuron definitions")
    
    # Process coalition files
    process_coalition_files(descriptions_dir, experiments_dir, all_neurons)
    
    print(f"Output directory: {experiments_dir}")
    return 0


if __name__ == "__main__":
    exit(main()) 