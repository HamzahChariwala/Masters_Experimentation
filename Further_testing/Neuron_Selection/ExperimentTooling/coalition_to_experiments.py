#!/usr/bin/env python3
"""
Script to convert coalition JSON files to experiment format compatible with analyze_metric.py.

This script processes coalition JSON files from circuit_verification/descriptions and converts
them into the format expected by analyze_metric.py, mapping exp_N strings to actual neuron
definitions using the useful_neurons.json file.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_useful_neurons() -> List[Dict[str, Any]]:
    """
    Load the useful_neurons.json file to get the neuron mapping.
    
    Returns:
        List of neuron definitions
    """
    useful_neurons_path = Path(__file__).parent / "Definitions" / "useful_neurons.json"
    
    try:
        with open(useful_neurons_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading useful_neurons.json: {e}")
        return []


def exp_string_to_neuron_def(exp_string: str, useful_neurons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert an exp_N string to a neuron definition.
    
    Args:
        exp_string: String like "exp_36_q_net.0"
        useful_neurons: List of neuron definitions from useful_neurons.json
        
    Returns:
        Neuron definition dictionary
    """
    # Extract the number from the exp string (e.g., "exp_36_q_net.0" -> 36)
    try:
        # Split by underscore and get the number part
        parts = exp_string.split('_')
        if len(parts) >= 2 and parts[0] == 'exp':
            exp_number = int(parts[1])
            # Convert to 0-based index (exp_1 -> index 0)
            index = exp_number - 1
            
            if 0 <= index < len(useful_neurons):
                return useful_neurons[index]
            else:
                print(f"Warning: Index {index} out of range for useful_neurons (length: {len(useful_neurons)})")
                return {}
        else:
            print(f"Warning: Could not parse experiment string: {exp_string}")
            return {}
    except (ValueError, IndexError) as e:
        print(f"Warning: Error parsing experiment string {exp_string}: {e}")
        return {}


def convert_coalition_to_experiments(coalition_file: Path, useful_neurons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a coalition JSON file to experiment format.
    
    Args:
        coalition_file: Path to the coalition JSON file
        useful_neurons: List of neuron definitions
        
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
                # Convert each exp_N string to neuron definition and group by layer
                layer_neurons = {}
                
                for exp_string in exp_data['included_experiments']:
                    neuron_def = exp_string_to_neuron_def(exp_string, useful_neurons)
                    if neuron_def:
                        # neuron_def is like {"q_net.0": [35]}
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


def process_coalition_files(descriptions_dir: Path, output_dir: Path, useful_neurons: List[Dict[str, Any]]) -> None:
    """
    Process all coalition files in the descriptions directory.
    
    Args:
        descriptions_dir: Directory containing coalition JSON files
        output_dir: Directory to save experiment files
        useful_neurons: List of neuron definitions
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
        experiments = convert_coalition_to_experiments(coalition_file, useful_neurons)
        
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
    descriptions_dir = agent_path / "circuit_verification" / "descriptions"
    experiments_dir = agent_path / "circuit_verification" / "experiments"
    
    # Create experiments directory
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if descriptions directory exists
    if not descriptions_dir.exists():
        print(f"Error: Descriptions directory not found: {descriptions_dir}")
        return 1
    
    # Load useful neurons mapping
    print("Loading useful neurons mapping...")
    useful_neurons = load_useful_neurons()
    
    if not useful_neurons:
        print("Error: Could not load useful neurons mapping")
        return 1
    
    print(f"Loaded {len(useful_neurons)} neuron definitions")
    
    # Process coalition files
    process_coalition_files(descriptions_dir, experiments_dir, useful_neurons)
    
    print(f"Output directory: {experiments_dir}")
    return 0


if __name__ == "__main__":
    exit(main()) 