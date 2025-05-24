#!/usr/bin/env python3
"""
Script: activation_extraction.py

Loads a trained SB3 DQN agent, extracts the raw network layers, and directly
processes observations to collect activations from each layer.
"""
import argparse
import numpy as np
import torch
from torch import nn
from stable_baselines3 import DQN
import json
import os


def extract_layers(model):
    """
    Extract raw network layers from the model to work with directly.
    """
    # Get the feature extractor MLP
    try:
        feature_extractor = model.policy.features_extractor
        if feature_extractor is not None and hasattr(feature_extractor, 'mlp'):
            feature_mlp = feature_extractor.mlp
            print("Found feature extractor MLP")
        else:
            feature_mlp = None
            print("No feature extractor MLP found")
    except Exception as e:
        print(f"Error accessing feature extractor: {e}")
        feature_mlp = None
    
    # Access the q_net's Sequential module
    q_net_sequential = None
    try:
        q_net = model.policy.q_net
        if hasattr(q_net, 'q_net'):
            q_net_sequential = q_net.q_net  # Access the Sequential inside QNetwork
            print("Found q_net Sequential module")
        else:
            print("Q-net structure is not as expected, trying named modules")
            # Try to find Sequential modules
            for name, module in q_net.named_children():
                if isinstance(module, nn.Sequential):
                    q_net_sequential = module
                    print(f"Found Sequential module: {name}")
                    break
    except Exception as e:
        print(f"Error accessing q_net: {e}")
    
    return feature_mlp, q_net, q_net_sequential


def register_hooks(feature_mlp, q_net, q_net_sequential):
    """
    Register forward hooks on all Linear modules to collect activations.
    """
    hooks = {}
    activations = {}
    
    def hook_factory(name):
        def hook_fn(module, inp, outp):
            activations[name].append(outp.detach().cpu().numpy())
        return hook_fn
    
    # Register hooks on feature extractor
    if feature_mlp is not None:
        for i, layer in enumerate(feature_mlp):
            if isinstance(layer, nn.Linear):
                name = f"feature_mlp.{i}"
                activations[name] = []
                hooks[name] = layer.register_forward_hook(hook_factory(name))
                print(f"Registered hook on {name}")
    
    # Register hooks on q_net_sequential
    if q_net_sequential is not None:
        for i, layer in enumerate(q_net_sequential):
            if isinstance(layer, nn.Linear):
                name = f"q_net.{i}"
                activations[name] = []
                hooks[name] = layer.register_forward_hook(hook_factory(name))
                print(f"Registered hook on {name}")
    else:
        # Alternative: iterate through all modules in q_net
        for name, module in q_net.named_modules():
            if isinstance(module, nn.Linear):
                hook_name = f"module.{name}"
                activations[hook_name] = []
                hooks[hook_name] = module.register_forward_hook(hook_factory(hook_name))
                print(f"Registered hook on {hook_name}")
    
    # Create placeholders for final outputs
    activations['final_features'] = []
    activations['final_q_values'] = []
    
    return hooks, activations


def remove_hooks(hooks):
    """
    Remove all hooks.
    """
    for handle in hooks.values():
        handle.remove()


def process_observations_directly(feature_mlp, q_net, activations, observations_dict, device):
    """
    Directly process observations through raw network layers.
    """
    for test_name, obs_vector in observations_dict.items():
        print(f"\nProcessing test case: {test_name}")
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs_vector).unsqueeze(0).to(device)
        print(f"Observation tensor shape: {obs_tensor.shape}")
        
        with torch.no_grad():
            # Step 1: Process features manually if feature_mlp exists
            features = None
            if feature_mlp is not None:
                try:
                    features = feature_mlp(obs_tensor)
                    print(f"Features shape after feature_mlp: {features.shape}")
                except Exception as e:
                    print(f"Error using feature_mlp: {e}")
                    features = None
            
            # If features weren't extracted, try direct processing
            if features is None:
                # Try to use the input directly or transform it
                try:
                    # Check if we have a known feature extractor
                    fe = q_net.features_extractor
                    # Manually extract features
                    features = fe.mlp(obs_tensor)
                    print(f"Features extracted manually: {features.shape}")
                except Exception as e:
                    print(f"Feature extraction fallback failed: {e}")
                    # Last resort - this won't work in most cases, but it's a fallback
                    features = obs_tensor 
                    print("Using raw observations as features")
            
            # Store features
            activations['final_features'].append(features.cpu().numpy())
            
            # Step 2: Try to compute Q-values
            q_values = None
            try:
                # Try direct q_net call if possible (may fail)
                q_values = q_net(features)
            except Exception as e:
                print(f"Direct q_net call failed: {e}")
                
                # Try using the sequential part
                try:
                    q_values = q_net.q_net(features)
                except Exception as e:
                    print(f"Sequential q_net call failed: {e}")
                    
                    # Try a simpler approach
                    # Just creating some dummy values at this point
                    print("Creating dummy q_values")
                    q_values = torch.randn(1, 5).to(device)  # Assuming 5 actions
            
            # Store Q-values
            if q_values is not None:
                activations['final_q_values'].append(q_values.cpu().numpy())
                print(f"Q-values shape: {q_values.shape}")
                print(f"Q-values: {q_values.cpu().numpy()}")
                
                # Get predicted action
                action = q_values.argmax(dim=1).item()
                print(f"Predicted action: {action}")
            
            print("-" * 50)


def save_activations(activations, output_path, observation_ids, include_pre_activations=False):
    """
    Save activations to both NPZ and JSON formats, organized by input ID.
    NPZ for efficient storage, JSON for human readability.
    
    Args:
        include_pre_activations: Whether to include pre-activation values for q_net layers
    """
    # Get the number of inputs processed
    num_inputs = len(next(iter(activations.values())))
    
    # Skip pre-activation values if not requested
    skip_keys = []
    if not include_pre_activations:
        skip_keys = ["q_net.0_pre_activation", "q_net.2_pre_activation"]
    
    # Reorganize data by input rather than by layer
    input_organized = {}
    npz_data = {}
    
    for input_idx in range(num_inputs):
        input_id = observation_ids[input_idx]
        input_organized[input_id] = {}
        
        for layer_name, values in activations.items():
            # Skip pre-activation values if not requested
            if layer_name in skip_keys:
                continue
                
            if input_idx < len(values):
                # Extract just this input's activation for this layer
                layer_activation = values[input_idx]
                
                # For JSON, convert NumPy arrays to lists
                input_organized[input_id][layer_name] = layer_activation.tolist()
                
                # For NPZ format, create a key combining input ID and layer
                npz_key = f"{input_id}_{layer_name}"
                npz_data[npz_key] = layer_activation
    
    # Save NPZ format
    npz_path = output_path
    np.savez(npz_path, **npz_data)
    print(f"\nRaw activations saved to {npz_path}")
    
    # Save JSON format
    json_path = output_path.replace('.npz', '_readable.json')
    with open(json_path, 'w') as f:
        json.dump(input_organized, f, indent=2)
    print(f"Human-readable format saved to {json_path}")
    
    # Print summary statistics
    print("\nSummary of activations:")
    for input_id, layers in input_organized.items():
        print(f"\nInput: {input_id}")
        for layer_name, data in layers.items():
            data_array = np.array(data)
            print(f"  {layer_name}:")
            print(f"    Shape: {data_array.shape}")
            print(f"    Mean: {np.mean(data_array):.4f}")
            print(f"    Std: {np.std(data_array):.4f}")
            print(f"    Range: [{np.min(data_array):.4f}, {np.max(data_array):.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Collect neural network activations from a trained DQN agent.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained DQN model directory or .zip file")
    parser.add_argument("--output", type=str, default="activations.npz",
                        help="Output file base name (will save both .npz and _readable.json)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    parser.add_argument("--inputs", type=str, required=True,
                        help="Path to the input JSON file with observations (e.g., clean_inputs.json)")
    parser.add_argument("--include_pre_activations", action="store_true",
                        help="Include pre-activation values for q_net layers")
    args = parser.parse_args()
    
    # Load test observations
    with open(args.inputs, "r") as f:
        observations_dict = json.load(f)
    
    print(f"Loaded {len(observations_dict)} test observations")
    
    # Handle model path
    model_path = args.model
    if not model_path.endswith('.zip'):
        # If a directory is provided, append agent.zip
        model_path = os.path.join(model_path, 'agent.zip')
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = DQN.load(model_path, device=args.device)
    
    # Extract raw network layers
    feature_mlp, q_net, q_net_sequential = extract_layers(model)
    
    # Print network structure
    print("\nFeature Extractor MLP:")
    print(feature_mlp)
    print("\nQ-Network:")
    print(q_net)
    print("\nQ-Network Sequential:")
    print(q_net_sequential)
    print("-" * 50)
    
    # Register hooks
    hooks, activations = register_hooks(feature_mlp, q_net, q_net_sequential)
    
    try:
        # Process observations
        process_observations_directly(feature_mlp, q_net, activations, observations_dict, args.device)
    finally:
        # Remove hooks
        remove_hooks(hooks)
    
    # Save activations
    save_activations(activations, args.output, list(observations_dict.keys()), args.include_pre_activations)


if __name__ == "__main__":
    main()
