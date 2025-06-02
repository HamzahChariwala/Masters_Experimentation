#!/usr/bin/env python3
"""
Script: generate_experiment.py

Generates experiment definitions for activation patching in a structured format.
This script allows for parametric definition of patching experiments by generating
appropriately formatted JSON files for use with the activation_patching.py script.
"""
import os
import json
import argparse
import sys
import pickle
import numpy as np
import torch
from typing import Dict, List, Union, Any, Optional, Tuple

# Default directory for saving experiment definitions
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Definitions")

# Default model layer structure for MiniGrid agents
# Format: {layer_name: neuron_count}
DEFAULT_LAYER_STRUCTURE = {
    "features_extractor.mlp.0": 64,
    "q_net.0": 64,
    "q_net.2": 64,
    "q_net.4": 5  # Output layer with 5 actions
}


def generate_single_neuron_experiments(
    layer_structure: Dict[str, int],
    output_file: str,
    include_output_logits: bool = True
) -> None:
    """
    Generate an experiment definition that patches each neuron individually.
    
    This creates a separate experiment for each neuron in each layer,
    allowing for a systematic analysis of individual neuron effects.
    
    Args:
        layer_structure: Dictionary mapping layer names to their neuron counts
        output_file: Path to save the generated experiment definition
        include_output_logits: Whether to include output logits patching
    """
    experiments = []
    
    # For each layer
    for layer_name, neuron_count in layer_structure.items():
        # Generate an experiment for each neuron in the layer
        for neuron_idx in range(neuron_count):
            # Create an experiment that patches just this single neuron
            experiment = {
                layer_name: [neuron_idx]
            }
            experiments.append(experiment)
    
    # Add output logits patching if requested
    if include_output_logits and "output_logits" not in layer_structure:
        # Find the output layer (typically the last layer)
        output_layer_name = list(layer_structure.keys())[-1]
        output_size = layer_structure[output_layer_name]
        
        # Create experiments for patching each output logit
        for logit_idx in range(output_size):
            experiment = {
                "output_logits": [logit_idx]
            }
            experiments.append(experiment)
    
    # Write the experiments to a JSON file
    with open(output_file, 'w') as f:
        json.dump(experiments, f, indent=2)
    
    print(f"Generated {len(experiments)} single-neuron patching experiments")
    print(f"Output saved to {output_file}")


def generate_layer_experiments(
    layer_structure: Dict[str, int],
    output_file: str,
    include_output_logits: bool = True
) -> None:
    """
    Generate an experiment definition that patches each layer completely.
    
    This creates a separate experiment for each layer, patching all neurons
    in that layer simultaneously.
    
    Args:
        layer_structure: Dictionary mapping layer names to their neuron counts
        output_file: Path to save the generated experiment definition
        include_output_logits: Whether to include output logits patching
    """
    experiments = []
    
    # For each layer, create an experiment that patches the entire layer
    for layer_name in layer_structure.keys():
        experiment = {
            layer_name: "all"
        }
        experiments.append(experiment)
    
    # Add output logits patching if requested
    if include_output_logits and "output_logits" not in layer_structure:
        experiment = {
            "output_logits": "all"
        }
        experiments.append(experiment)
    
    # Write the experiments to a JSON file
    with open(output_file, 'w') as f:
        json.dump(experiments, f, indent=2)
    
    print(f"Generated {len(experiments)} layer-wise patching experiments")
    print(f"Output saved to {output_file}")


def generate_neuron_groups_experiments(
    layer_structure: Dict[str, int],
    group_size: int,
    output_file: str,
    include_output_logits: bool = True
) -> None:
    """
    Generate an experiment definition that patches neurons in groups.
    
    This creates experiments where neurons are patched in groups of
    the specified size, allowing for a more efficient search process.
    
    Args:
        layer_structure: Dictionary mapping layer names to their neuron counts
        group_size: Number of neurons to patch together in each experiment
        output_file: Path to save the generated experiment definition
        include_output_logits: Whether to include output logits patching
    """
    experiments = []
    
    # For each layer
    for layer_name, neuron_count in layer_structure.items():
        # Generate groups of neurons
        for start_idx in range(0, neuron_count, group_size):
            # Calculate the end index (cap at neuron_count)
            end_idx = min(start_idx + group_size, neuron_count)
            
            # Create the neuron group
            neuron_group = list(range(start_idx, end_idx))
            
            # Create an experiment for this group
            experiment = {
                layer_name: neuron_group
            }
            experiments.append(experiment)
    
    # Add output logits patching if requested
    if include_output_logits and "output_logits" not in layer_structure:
        # Find the output layer (typically the last layer)
        output_layer_name = list(layer_structure.keys())[-1]
        output_size = layer_structure[output_layer_name]
        
        # Create experiments for patching groups of output logits
        for start_idx in range(0, output_size, group_size):
            # Calculate the end index (cap at output_size)
            end_idx = min(start_idx + group_size, output_size)
            
            # Create the logit group
            logit_group = list(range(start_idx, end_idx))
            
            # Create an experiment for this group
            experiment = {
                "output_logits": logit_group
            }
            experiments.append(experiment)
    
    # Write the experiments to a JSON file
    with open(output_file, 'w') as f:
        json.dump(experiments, f, indent=2)
    
    print(f"Generated {len(experiments)} neuron group patching experiments (group size: {group_size})")
    print(f"Output saved to {output_file}")


def generate_custom_experiment(
    experiment_specs: List[Dict[str, Union[List[int], str]]],
    output_file: str
) -> None:
    """
    Generate a custom experiment definition based on provided specifications.
    
    Args:
        experiment_specs: List of experiment specifications
        output_file: Path to save the generated experiment definition
    """
    # Write the experiments to a JSON file
    with open(output_file, 'w') as f:
        json.dump(experiment_specs, f, indent=2)
    
    print(f"Generated {len(experiment_specs)} custom patching experiments")
    print(f"Output saved to {output_file}")


def get_layer_structure_from_agent(agent_path: str) -> Dict[str, int]:
    """
    Extract layer structure from an agent model.
    
    This function analyzes the agent model to determine the structure
    of its layers, including their names and neuron counts.
    
    Args:
        agent_path: Path to the agent directory
        
    Returns:
        Dictionary mapping layer names to their neuron counts
    """
    try:
        # Try to import the required libraries
        import torch
        
        # Set up path for model and configuration files
        model_path = os.path.join(agent_path, "pytorch_model.pt")
        
        # Check if the PyTorch model file exists
        if not os.path.exists(model_path):
            # Try to find other model files
            model_files = [f for f in os.listdir(agent_path) if f.endswith('.pt') or f.endswith('.pth')]
            if model_files:
                model_path = os.path.join(agent_path, model_files[0])
                print(f"Using model file: {model_path}")
            else:
                print(f"No model file found in {agent_path}. Using default layer structure.")
                return DEFAULT_LAYER_STRUCTURE
        
        # Load the model
        model = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Initialize the layer structure dictionary
        layer_structure = {}
        
        # Check if the loaded object is a dictionary with a 'state_dict' key (common format)
        if isinstance(model, dict) and 'state_dict' in model:
            state_dict = model['state_dict']
        elif hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
        else:
            state_dict = model  # Assume it's already a state dict
        
        # Extract layer names and sizes from the state dictionary
        for key, tensor in state_dict.items():
            # We're interested in weight tensors, not bias or other parameters
            if 'weight' in key and len(tensor.shape) > 1:
                # Extract the base layer name (remove '.weight' suffix)
                layer_name = key.replace('.weight', '')
                
                # Get the output dimension (number of neurons)
                num_neurons = tensor.shape[0]
                
                # Store in the layer structure dictionary
                layer_structure[layer_name] = num_neurons
                print(f"Found layer: {layer_name} with {num_neurons} neurons")
        
        # Check if we found any layers
        if not layer_structure:
            print("No layers found in the model. Using default layer structure.")
            return DEFAULT_LAYER_STRUCTURE
        
        return layer_structure
        
    except (ImportError, FileNotFoundError, KeyError, AttributeError) as e:
        print(f"Error extracting layer structure: {str(e)}")
        print("Using default layer structure.")
        return DEFAULT_LAYER_STRUCTURE


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment definitions for activation patching.")
    
    parser.add_argument("--type", type=str, required=True, choices=["single_neuron", "layer", "neuron_groups", "custom"],
                        help="Type of experiment definition to generate")
    
    parser.add_argument("--agent_path", type=str, default=None,
                        help="Path to the agent directory (for extracting layer structure)")
    
    parser.add_argument("--layer_structure", type=str, default=None,
                        help="JSON string or file path defining the layer structure (overrides agent_path)")
    
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for the generated experiment definition")
    
    parser.add_argument("--group_size", type=int, default=5,
                        help="Number of neurons per group for neuron_groups experiment type")
    
    parser.add_argument("--custom_spec", type=str, default=None,
                        help="JSON string or file path defining custom experiment specifications")
    
    parser.add_argument("--include_output_logits", action="store_true", default=True,
                        help="Include patching of output logits in the experiments")
    
    parser.add_argument("--exclude_output_logits", dest="include_output_logits", action="store_false",
                        help="Exclude patching of output logits from the experiments")
    
    args = parser.parse_args()
    
    # Determine layer structure
    layer_structure = DEFAULT_LAYER_STRUCTURE
    
    if args.agent_path:
        # Extract layer structure from agent model
        layer_structure = get_layer_structure_from_agent(args.agent_path)
    
    if args.layer_structure:
        # Parse layer structure from JSON string or file
        if os.path.exists(args.layer_structure):
            with open(args.layer_structure, 'r') as f:
                layer_structure = json.load(f)
        else:
            try:
                layer_structure = json.loads(args.layer_structure)
            except json.JSONDecodeError:
                print("Error: Invalid JSON string for layer_structure")
                return 1
    
    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        # Create default output file name based on experiment type
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(DEFAULT_OUTPUT_DIR, f"{args.type}_experiment.json")
    
    # Generate experiment definition based on type
    if args.type == "single_neuron":
        generate_single_neuron_experiments(
            layer_structure, 
            output_file,
            include_output_logits=args.include_output_logits
        )
    
    elif args.type == "layer":
        generate_layer_experiments(
            layer_structure, 
            output_file,
            include_output_logits=args.include_output_logits
        )
    
    elif args.type == "neuron_groups":
        generate_neuron_groups_experiments(
            layer_structure, 
            args.group_size, 
            output_file,
            include_output_logits=args.include_output_logits
        )
    
    elif args.type == "custom":
        # Parse custom experiment specifications
        if not args.custom_spec:
            print("Error: custom_spec is required for custom experiment type")
            return 1
        
        if os.path.exists(args.custom_spec):
            with open(args.custom_spec, 'r') as f:
                custom_specs = json.load(f)
        else:
            try:
                custom_specs = json.loads(args.custom_spec)
            except json.JSONDecodeError:
                print("Error: Invalid JSON string for custom_spec")
                return 1
        
        generate_custom_experiment(custom_specs, output_file)
    
    return 0


if __name__ == "__main__":
    main() 