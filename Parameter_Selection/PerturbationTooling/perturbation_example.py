#!/usr/bin/env python3
"""
Example script showing how to use the WeightPerturbationTool programmatically.

This demonstrates:
1. Loading a model
2. Creating perturbation configurations
3. Running experiments
4. Analyzing results
"""

import os
import sys
import torch

# Add the parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from model_loader import DQNModelLoader
from weight_perturbation import WeightPerturbationTool


def create_custom_perturbations():
    """
    Create custom perturbation configurations.
    """
    perturbations = {
        "test_feature_extractor": {
            "features_extractor.mlp.0.weight": {
                0: {5: 0.1, 10: -0.1},  # Modify two weights in neuron 0
                1: {0: 0.2}             # Modify one weight in neuron 1
            }
        },
        "test_q_network": {
            "q_net.0.weight": {
                5: {11: 5.3}  # Modify one weight in the first Q-network layer
            }
        },
        "test_bias": {
            "features_extractor.mlp.0.bias": {
                0: {0: 0.5},  # Increase bias of neuron 0
                1: {0: -0.3}  # Decrease bias of neuron 1
            }
        },
        "combined_changes": {
            "features_extractor.mlp.0.weight": {
                0: {5: 0.1}
            },
            "features_extractor.mlp.0.bias": {
                0: {0: 0.2}
            },
            "q_net.0.weight": {
                10: {20: -0.15}
            }
        }
    }
    
    return perturbations


def main():
    """
    Main example function.
    """
    # Configuration
    agent_path = "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6"
    device = "cpu"
    
    print("=== Weight Perturbation Tool Example ===")
    
    # Step 1: Load the model
    print("\n1. Loading DQN model...")
    try:
        loader = DQNModelLoader(agent_path, device=device)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("Make sure the agent path is correct and the agent exists.")
        return
    
    # Step 2: Create perturbation tool
    print("\n2. Creating perturbation tool...")
    tool = WeightPerturbationTool(loader)
    print("✓ Perturbation tool ready")
    
    # Step 3: Show layer information (optional)
    print("\n3. Model layer information:")
    layer_info = tool.get_layer_info()
    for layer_name, info in layer_info.items():
        print(f"  {layer_name}: {info['shape']} ({info['numel']:,} params)")
    
    # Step 4: Create input data
    print("\n4. Creating input data...")
    # Create a specific input (you can replace this with real data)
    input_data = {'MLP_input': torch.randn(1, 106, device=device)}
    print(f"✓ Input shape: {input_data['MLP_input'].shape}")
    
    # Step 5: Create perturbations
    print("\n5. Creating perturbation configurations...")
    perturbations = create_custom_perturbations()
    print(f"✓ Created {len(perturbations)} perturbation experiments:")
    for exp_id, config in perturbations.items():
        num_layers = len(config)
        total_changes = sum(len(neuron_changes) for layer_config in config.values() 
                           for neuron_changes in layer_config.values())
        print(f"  {exp_id}: {num_layers} layers, {total_changes} weight changes")
    
    # Step 6: Run experiments
    print("\n6. Running perturbation experiments...")
    results = tool.run_perturbation_experiments(perturbations, input_data)
    
    # Step 7: Analyze results
    print("\n7. Analyzing results...")
    tool.analyze_results(results)
    
    # Step 8: Custom analysis
    print("\n8. Custom analysis...")
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    print("Summary of logit changes:")
    for exp_id, result in valid_results.items():
        original = result['original_logits'].squeeze()
        perturbed = result['perturbed_logits'].squeeze()
        difference = result['logit_difference'].squeeze()
        
        print(f"\n{exp_id}:")
        print(f"  Original:  {original.numpy()}")
        print(f"  Perturbed: {perturbed.numpy()}")
        print(f"  Change:    {difference.numpy()}")
        print(f"  Max abs change: {result['max_change']:.6f}")
        
        # Check if any action preferences changed significantly
        original_action = torch.argmax(original).item()
        perturbed_action = torch.argmax(perturbed).item()
        
        if original_action != perturbed_action:
            print(f"  ⚠️  Action changed: {original_action} → {perturbed_action}")
        else:
            print(f"  ✓ Action unchanged: {original_action}")
    
    print("\n=== Example Complete ===")
    print("You can now:")
    print("- Modify the perturbations dictionary to test different weight changes")
    print("- Use real input data instead of random data")
    print("- Create more complex perturbation patterns")
    print("- Analyze the relationship between weight changes and output changes")


if __name__ == "__main__":
    main() 