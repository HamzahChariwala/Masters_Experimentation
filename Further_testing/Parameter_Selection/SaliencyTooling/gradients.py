#!/usr/bin/env python3
"""
Gradient computation tool for DQN models.
Computes gradients of all weights with respect to the highest output logit.
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, Any

# Add parent directory for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from model_loader import DQNModelLoader


def compute_gradients_for_agent(agent_path: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Compute gradients for all inputs in clean_inputs.json.
    
    Args:
        agent_path: Path to the agent directory
        device: Device to run on
        
    Returns:
        Dictionary containing gradients for each input
    """
    # Load model
    loader = DQNModelLoader(agent_path, device=device)
    model = loader.get_model()
    model.train()  # Enable gradients
    
    # Load clean inputs
    clean_inputs_path = os.path.join(agent_path, "activation_inputs", "clean_inputs.json")
    if not os.path.exists(clean_inputs_path):
        raise FileNotFoundError(f"clean_inputs.json not found at {clean_inputs_path}")
    
    with open(clean_inputs_path, 'r') as f:
        clean_inputs = json.load(f)
    
    results = {}
    
    print(f"Processing {len(clean_inputs)} inputs...")
    
    for i, (input_id, input_data) in enumerate(clean_inputs.items()):
        print(f"  {i+1}/{len(clean_inputs)}: {input_id}")
        
        # Prepare input tensor
        input_tensor = torch.tensor(input_data["input"], dtype=torch.float32, device=device)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        input_dict = {'MLP_input': input_tensor}
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        outputs = model(input_dict)
        
        # Get highest logit and compute gradients
        max_logit_idx = torch.argmax(outputs, dim=1)
        highest_logit = outputs[0, max_logit_idx]
        
        # Backward pass
        highest_logit.backward()
        
        # Collect gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().cpu().numpy().tolist()
        
        results[input_id] = {
            'max_logit_action': max_logit_idx.item(),
            'max_logit_value': highest_logit.item(),
            'gradients': gradients
        }
    
    return results


def save_gradients(results: Dict[str, Any], output_path: str) -> None:
    """Save gradient results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Gradients saved to: {output_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute gradients for DQN model")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for gradients")
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.agent_path, "gradient_analysis")
    
    print(f"Computing gradients for agent: {args.agent_path}")
    
    # Compute gradients
    results = compute_gradients_for_agent(args.agent_path, args.device)
    
    # Save results
    output_path = os.path.join(args.output_dir, "weight_gradients.json")
    save_gradients(results, output_path)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Processed {len(results)} inputs")
    print(f"  Output saved to: {output_path}")
    
    # Show sample gradient statistics
    if results:
        sample_key = next(iter(results))
        sample_gradients = results[sample_key]['gradients']
        print(f"  Gradient shapes:")
        for layer_name, grad in sample_gradients.items():
            grad_array = np.array(grad)
            print(f"    {layer_name}: {grad_array.shape}")


if __name__ == "__main__":
    main() 