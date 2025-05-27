#!/usr/bin/env python3
"""
Script to generate a patches file for activation patching.
This will create a JSON file with patch specifications for all neurons in all layers except q_net.4.
"""
import os
import json
import argparse

def generate_patches_file(output_path, layers, neurons_per_layer):
    """
    Generate a JSON file with patch specifications for all neurons in the specified layers.
    
    Args:
        output_path: Path to save the patches file
        layers: List of layer names to include
        neurons_per_layer: Dictionary mapping layer names to the number of neurons in each layer
    """
    patches = []
    
    for layer in layers:
        # Skip the excluded layer (q_net.4)
        if layer == "q_net.4":
            continue
        
        # Get the number of neurons in this layer
        num_neurons = neurons_per_layer.get(layer, 0)
        
        if num_neurons > 0:
            # Create a patch for each neuron
            for neuron_idx in range(num_neurons):
                patch = {layer: [neuron_idx]}
                patches.append(patch)
    
    # Ensure directory exists if there's a directory path
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(patches, f, indent=2)
    
    print(f"Generated patches file with {len(patches)} patch configurations at {output_path}")
    
def main():
    parser = argparse.ArgumentParser(description="Generate patches file for activation patching")
    parser.add_argument("--output", default="patches.json", help="Path to save the patches file")
    args = parser.parse_args()
    
    # Define the layer structure of the model
    # These are the typical layers in the model used in the codebase
    layers = [
        "q_net.0",
        "q_net.2",
        "q_net.4",  # This will be excluded
        "features_extractor.mlp.0"
    ]
    
    # Define the number of neurons in each layer
    # These values might need to be adjusted based on the actual model
    neurons_per_layer = {
        "q_net.0": 64,
        "q_net.2": 64,
        "q_net.4": 4,  # This will be excluded
        "features_extractor.mlp.0": 64
    }
    
    generate_patches_file(args.output, layers, neurons_per_layer)

if __name__ == "__main__":
    main() 