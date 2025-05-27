#!/usr/bin/env python3
"""
Script: weight_perturbation.py

A tool for applying targeted weight perturbations to DQN models and comparing the resulting logits.
This script allows you to specify precise weight changes and observe their effects on model outputs.

The perturbation dictionary structure is:
{
    experiment_id: {
        layer_name: {
            neuron_index: {
                weight_index: change_value
            }
        }
    }
}

Usage:
    from weight_perturbation import WeightPerturbationTool
    
    tool = WeightPerturbationTool(model_loader)
    perturbations = {
        "exp1": {
            "features_extractor.mlp.0.weight": {
                0: {5: 0.1, 10: -0.05}  # Change weights at positions [0,5] and [0,10]
            }
        }
    }
    results = tool.run_perturbation_experiments(perturbations, input_data)
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
import copy

# Add the parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from model_loader import DQNModelLoader


class WeightPerturbationTool:
    """
    A tool for applying targeted weight perturbations to DQN models.
    """
    
    def __init__(self, model_loader: DQNModelLoader):
        """
        Initialize the perturbation tool.
        
        Args:
            model_loader: An initialized DQNModelLoader instance
        """
        self.model_loader = model_loader
        self.model = model_loader.get_model()
        self.original_state_dict = self.model.state_dict()
        
        # Store original state dict as a deep copy for restoration
        self.backup_state_dict = {}
        for key, value in self.original_state_dict.items():
            self.backup_state_dict[key] = value.clone()
    
    def apply_perturbations(self, perturbation_spec: Dict[str, Dict[int, Dict[int, float]]], 
                           layer_name: str) -> None:
        """
        Apply perturbations to a specific layer based on the perturbation specification.
        
        Args:
            perturbation_spec: Dictionary mapping neuron_index -> weight_index -> change_value
            layer_name: Name of the layer to modify (e.g., "features_extractor.mlp.0.weight")
        """
        state_dict = self.model.state_dict()
        
        if layer_name not in state_dict:
            raise ValueError(f"Layer '{layer_name}' not found in model. Available layers: {list(state_dict.keys())}")
        
        # Get the parameter tensor
        param_tensor = state_dict[layer_name]
        
        # Apply each perturbation
        with torch.no_grad():
            for neuron_idx, weight_changes in perturbation_spec.items():
                for weight_idx, change_value in weight_changes.items():
                    if len(param_tensor.shape) == 2:  # Weight matrix
                        if neuron_idx >= param_tensor.shape[0] or weight_idx >= param_tensor.shape[1]:
                            raise IndexError(f"Index [{neuron_idx}, {weight_idx}] out of bounds for layer {layer_name} with shape {param_tensor.shape}")
                        param_tensor[neuron_idx, weight_idx] += change_value
                    elif len(param_tensor.shape) == 1:  # Bias vector
                        if neuron_idx >= param_tensor.shape[0]:
                            raise IndexError(f"Index [{neuron_idx}] out of bounds for layer {layer_name} with shape {param_tensor.shape}")
                        param_tensor[neuron_idx] += change_value
                    else:
                        raise ValueError(f"Unsupported parameter shape {param_tensor.shape} for layer {layer_name}")
        
        # Load the modified state dict back into the model
        self.model.load_state_dict(state_dict)
    
    def restore_original_weights(self) -> None:
        """
        Restore the model to its original state.
        """
        self.model.load_state_dict(self.backup_state_dict)
    
    def run_forward_pass(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Run a forward pass through the model with the given input.
        
        Args:
            input_data: Input tensor or dictionary of tensors
            
        Returns:
            Model output (logits)
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(input_data, dict):
                return self.model(input_data)
            else:
                # For non-dict inputs, we need to format them properly
                if len(input_data.shape) == 1:
                    input_data = input_data.unsqueeze(0)  # Add batch dimension
                return self.model({'MLP_input': input_data})
    
    def run_single_perturbation_experiment(self, 
                                          perturbation_config: Dict[str, Dict[int, Dict[int, float]]],
                                          input_data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run a single perturbation experiment.
        
        Args:
            perturbation_config: Dictionary mapping layer_name -> neuron_index -> weight_index -> change_value
            input_data: Input data for the forward pass
            
        Returns:
            Tuple of (original_logits, perturbed_logits)
        """
        # Get original logits
        self.restore_original_weights()
        original_logits = self.run_forward_pass(input_data)
        
        # Apply perturbations
        for layer_name, perturbation_spec in perturbation_config.items():
            self.apply_perturbations(perturbation_spec, layer_name)
        
        # Get perturbed logits
        perturbed_logits = self.run_forward_pass(input_data)
        
        # Restore original weights for next experiment
        self.restore_original_weights()
        
        return original_logits.clone(), perturbed_logits.clone()
    
    def run_perturbation_experiments(self, 
                                   perturbations: Dict[str, Dict[str, Dict[int, Dict[int, float]]]],
                                   input_data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Run multiple perturbation experiments.
        
        Args:
            perturbations: Dictionary with structure:
                {
                    experiment_id: {
                        layer_name: {
                            neuron_index: {
                                weight_index: change_value
                            }
                        }
                    }
                }
            input_data: Input data for the forward passes
            
        Returns:
            Dictionary with structure:
                {
                    experiment_id: {
                        'original_logits': torch.Tensor,
                        'perturbed_logits': torch.Tensor,
                        'logit_difference': torch.Tensor,
                        'max_change': float,
                        'mean_abs_change': float
                    }
                }
        """
        results = {}
        
        print(f"Running {len(perturbations)} perturbation experiments...")
        
        for experiment_id, perturbation_config in perturbations.items():
            print(f"  Running experiment: {experiment_id}")
            
            try:
                original_logits, perturbed_logits = self.run_single_perturbation_experiment(
                    perturbation_config, input_data
                )
                
                # Calculate difference metrics
                logit_difference = perturbed_logits - original_logits
                max_change = torch.max(torch.abs(logit_difference)).item()
                mean_abs_change = torch.mean(torch.abs(logit_difference)).item()
                
                results[experiment_id] = {
                    'original_logits': original_logits,
                    'perturbed_logits': perturbed_logits,
                    'logit_difference': logit_difference,
                    'max_change': max_change,
                    'mean_abs_change': mean_abs_change,
                    'perturbation_config': perturbation_config  # Store for reference
                }
                
                print(f"    Max logit change: {max_change:.6f}")
                print(f"    Mean abs logit change: {mean_abs_change:.6f}")
                
            except Exception as e:
                print(f"    Error in experiment {experiment_id}: {e}")
                results[experiment_id] = {
                    'error': str(e),
                    'perturbation_config': perturbation_config
                }
        
        # Ensure model is restored to original state
        self.restore_original_weights()
        
        return results
    
    def analyze_results(self, results: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        Print a summary analysis of the perturbation results.
        
        Args:
            results: Results from run_perturbation_experiments
        """
        print("\n" + "="*60)
        print("PERTURBATION EXPERIMENT RESULTS")
        print("="*60)
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        error_results = {k: v for k, v in results.items() if 'error' in v}
        
        if valid_results:
            print(f"Successful experiments: {len(valid_results)}")
            print(f"Failed experiments: {len(error_results)}")
            
            # Summary statistics
            max_changes = [result['max_change'] for result in valid_results.values()]
            mean_changes = [result['mean_abs_change'] for result in valid_results.values()]
            
            print(f"\nLogit Change Summary:")
            print(f"  Max change range: [{min(max_changes):.6f}, {max(max_changes):.6f}]")
            print(f"  Mean abs change range: [{min(mean_changes):.6f}, {max(mean_changes):.6f}]")
            print(f"  Average max change: {np.mean(max_changes):.6f}")
            print(f"  Average mean abs change: {np.mean(mean_changes):.6f}")
            
            # Individual experiment details
            print(f"\nIndividual Experiment Results:")
            for exp_id, result in valid_results.items():
                print(f"  {exp_id}:")
                print(f"    Original logits: {result['original_logits'].squeeze().numpy()}")
                print(f"    Perturbed logits: {result['perturbed_logits'].squeeze().numpy()}")
                print(f"    Difference: {result['logit_difference'].squeeze().numpy()}")
                print(f"    Max change: {result['max_change']:.6f}")
                print()
        
        if error_results:
            print(f"\nFailed Experiments:")
            for exp_id, result in error_results.items():
                print(f"  {exp_id}: {result['error']}")
    
    def get_layer_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all layers in the model for reference.
        
        Returns:
            Dictionary with layer information
        """
        layer_info = {}
        state_dict = self.model.state_dict()
        
        for layer_name, param in state_dict.items():
            layer_info[layer_name] = {
                'shape': list(param.shape),
                'numel': param.numel(),
                'dtype': str(param.dtype),
                'device': str(param.device)
            }
        
        return layer_info
    
    def print_layer_info(self) -> None:
        """
        Print information about all layers for easy reference when creating perturbations.
        """
        print("\n" + "="*60)
        print("MODEL LAYER INFORMATION")
        print("="*60)
        
        layer_info = self.get_layer_info()
        
        for layer_name, info in layer_info.items():
            print(f"{layer_name}:")
            print(f"  Shape: {info['shape']}")
            print(f"  Parameters: {info['numel']:,}")
            print(f"  Type: {info['dtype']}")
            print()


def create_example_perturbations() -> Dict[str, Dict[str, Dict[int, Dict[int, float]]]]:
    """
    Create example perturbation configurations for demonstration.
    
    Returns:
        Example perturbations dictionary
    """
    perturbations = {
        "small_feature_change": {
            "features_extractor.mlp.0.weight": {
                0: {5: 0.1}  # Change weight at position [0, 5] by +0.1
            }
        },
        "multiple_feature_changes": {
            "features_extractor.mlp.0.weight": {
                0: {5: 0.1, 10: -0.05},  # Change two weights in neuron 0
                1: {3: 0.2}              # Change one weight in neuron 1
            }
        },
        "bias_change": {
            "features_extractor.mlp.0.bias": {
                0: {0: 0.5}  # For bias, weight_index is typically 0
            }
        },
        "q_network_change": {
            "q_net.0.weight": {
                10: {15: -0.3}  # Change weight in first Q-network layer
            }
        },
        "multi_layer_change": {
            "features_extractor.mlp.0.weight": {
                0: {5: 0.1}
            },
            "q_net.0.weight": {
                10: {15: -0.2}
            }
        }
    }
    
    return perturbations


def main():
    """
    Demonstration of the weight perturbation tool.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Weight Perturbation Tool for DQN Models")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory or zip file")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to load the model on (cpu or cuda)")
    parser.add_argument("--show_layers", action="store_true",
                       help="Show layer information for creating perturbations")
    
    args = parser.parse_args()
    
    # Load the model
    print("Loading DQN model...")
    loader = DQNModelLoader(args.agent_path, device=args.device)
    
    # Create perturbation tool
    tool = WeightPerturbationTool(loader)
    
    if args.show_layers:
        tool.print_layer_info()
        return
    
    # Create dummy input data (matching the model's expected input)
    dummy_input = {'MLP_input': torch.randn(1, 106, device=args.device)}
    
    # Create example perturbations
    perturbations = create_example_perturbations()
    
    print(f"\nCreated {len(perturbations)} example perturbations:")
    for exp_id, config in perturbations.items():
        print(f"  {exp_id}: {len(config)} layers affected")
    
    # Run perturbation experiments
    results = tool.run_perturbation_experiments(perturbations, dummy_input)
    
    # Analyze results
    tool.analyze_results(results)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("You can now use this tool to:")
    print("- Apply precise weight changes to specific neurons")
    print("- Compare original vs perturbed model outputs")
    print("- Analyze the effect of weight changes on logits")
    print("- Run batch experiments with multiple perturbations")


if __name__ == "__main__":
    main() 