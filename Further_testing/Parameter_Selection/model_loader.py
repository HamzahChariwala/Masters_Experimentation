#!/usr/bin/env python3
"""
Script: model_loader.py

Loads a DQN agent and prepares it as a standard PyTorch model for inspection and manipulation.
This script provides functionality to:
1. Load a DQN agent with the correct custom feature extractor
2. Extract the model components as standard PyTorch modules
3. Inspect model parameters, shapes, and architecture
4. Provide easy access to state_dict() for parameter analysis

Usage:
    python model_loader.py --agent_path Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Add the root directory to sys.path to ensure proper imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")

# Import the custom feature extractor
from Environment_Tooling.BespokeEdits.FeatureExtractor import CustomCombinedExtractor

# Import stable_baselines3
from stable_baselines3 import DQN


class DQNModelLoader:
    """
    A class to load DQN agents and prepare them as standard PyTorch models.
    """
    
    def __init__(self, agent_path: str, device: str = "cpu"):
        """
        Initialize the model loader.
        
        Args:
            agent_path: Path to the agent directory or zip file
            device: Device to load the model on ("cpu" or "cuda")
        """
        self.agent_path = agent_path
        self.device = device
        self.agent = None
        self.model = None
        
        # Load the agent
        self._load_agent()
        
        # Extract the model components
        self._extract_model()
    
    def _load_agent(self) -> None:
        """Load the DQN agent from the specified path."""
        # Determine the correct path to the agent file
        if os.path.isfile(self.agent_path) and self.agent_path.endswith('.zip'):
            # Direct path to zip file
            model_path = self.agent_path
        elif os.path.isdir(self.agent_path):
            # Directory path - look for agent.zip or similar
            possible_files = ['agent.zip', 'model.zip', 'dqn.zip']
            model_path = None
            
            for filename in possible_files:
                candidate_path = os.path.join(self.agent_path, filename)
                if os.path.exists(candidate_path):
                    model_path = candidate_path
                    break
            
            if model_path is None:
                # Look for any .zip file in the directory
                for file in os.listdir(self.agent_path):
                    if file.endswith('.zip'):
                        model_path = os.path.join(self.agent_path, file)
                        break
                        
            if model_path is None:
                raise FileNotFoundError(f"No agent file found in directory: {self.agent_path}")
        else:
            raise FileNotFoundError(f"Agent path not found: {self.agent_path}")
        
        print(f"Loading agent from: {model_path}")
        
        # Load the agent with custom objects
        custom_objects = {"features_extractor_class": CustomCombinedExtractor}
        
        try:
            self.agent = DQN.load(model_path, device=self.device, custom_objects=custom_objects)
            print("✓ Agent loaded successfully")
            print(f"  Policy type: {type(self.agent.policy)}")
            print(f"  Q-network type: {type(self.agent.q_net)}")
            
            # Check feature extractor
            if hasattr(self.agent.policy, 'features_extractor') and self.agent.policy.features_extractor is not None:
                print(f"  Feature extractor type: {type(self.agent.policy.features_extractor)}")
                if hasattr(self.agent.policy.features_extractor, '_features_dim'):
                    print(f"  Feature dimension: {self.agent.policy.features_extractor._features_dim}")
            else:
                print("  Warning: No feature extractor found")
                
        except Exception as e:
            print(f"Error loading agent: {e}")
            raise
    
    def _extract_model(self) -> None:
        """Extract the model components as a standard PyTorch model."""
        # The main model we want to work with is the policy's Q-network
        # This contains both the feature extractor and the Q-value head
        self.model = self.agent.policy.q_net
        
        # Set to evaluation mode
        self.model.eval()
        
        print(f"✓ Model extracted: {type(self.model)}")
        print(f"  Model device: {next(self.model.parameters()).device}")
        print(f"  Model in eval mode: {not self.model.training}")
    
    def get_model(self) -> nn.Module:
        """
        Get the main model as a standard PyTorch module.
        
        Returns:
            The Q-network as a PyTorch module
        """
        return self.model
    
    def get_feature_extractor(self) -> Optional[nn.Module]:
        """
        Get the feature extractor as a separate PyTorch module.
        
        Returns:
            The feature extractor module, or None if not available
        """
        if hasattr(self.agent.policy, 'features_extractor'):
            return self.agent.policy.features_extractor
        return None
    
    def get_q_head(self) -> Optional[nn.Module]:
        """
        Get the Q-value head (the part after feature extraction).
        
        Returns:
            The Q-value head module, or None if not separable
        """
        # For DQN, the q_net typically contains both feature extractor and Q-head
        # The Q-head is usually the 'q_net' part after feature extraction
        if hasattr(self.model, 'q_net'):
            return self.model.q_net
        return None
    
    def print_model_summary(self) -> None:
        """Print a comprehensive summary of the model architecture."""
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        
        # Basic info
        print(f"Agent type: DQN")
        print(f"Device: {next(self.model.parameters()).device}")
        print(f"Total parameters: {self.count_parameters():,}")
        print(f"Trainable parameters: {self.count_parameters(trainable_only=True):,}")
        
        # Observation and action spaces
        print(f"\nObservation space: {self.agent.observation_space}")
        print(f"Action space: {self.agent.action_space}")
        
        # Model architecture
        print(f"\nModel architecture:")
        print(self.model)
        
        print("\n" + "="*60)
        print("STATE_DICT SUMMARY")
        print("="*60)
        
        # Print all parameter names and shapes
        state_dict = self.model.state_dict()
        for name, param in state_dict.items():
            print(f"{name:50} {str(param.shape):20} {param.numel():>10,} params")
        
        print(f"\nTotal parameters in state_dict: {sum(p.numel() for p in state_dict.values()):,}")
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """
        Count the number of parameters in the model.
        
        Args:
            trainable_only: If True, only count trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())
    
    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get the model's state dictionary.
        
        Returns:
            Dictionary mapping parameter names to tensors
        """
        return self.model.state_dict()
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about each parameter.
        
        Returns:
            Dictionary with parameter information
        """
        param_info = {}
        
        for name, param in self.model.named_parameters():
            param_info[name] = {
                'shape': list(param.shape),
                'numel': param.numel(),
                'dtype': str(param.dtype),
                'requires_grad': param.requires_grad,
                'device': str(param.device),
                'mean': float(param.data.mean().item()),
                'std': float(param.data.std().item()),
                'min': float(param.data.min().item()),
                'max': float(param.data.max().item())
            }
        
        return param_info
    
    def save_model_components(self, output_dir: str) -> None:
        """
        Save model components for later use.
        
        Args:
            output_dir: Directory to save the components
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the full model
        torch.save(self.model, os.path.join(output_dir, "full_model.pt"))
        
        # Save the state dict
        torch.save(self.model.state_dict(), os.path.join(output_dir, "state_dict.pt"))
        
        # Save feature extractor if available
        feature_extractor = self.get_feature_extractor()
        if feature_extractor is not None:
            torch.save(feature_extractor, os.path.join(output_dir, "feature_extractor.pt"))
            torch.save(feature_extractor.state_dict(), os.path.join(output_dir, "feature_extractor_state_dict.pt"))
        
        # Save parameter info as JSON
        import json
        param_info = self.get_parameter_info()
        with open(os.path.join(output_dir, "parameter_info.json"), 'w') as f:
            json.dump(param_info, f, indent=2)
        
        print(f"✓ Model components saved to: {output_dir}")
    
    def test_forward_pass(self, batch_size: int = 1) -> None:
        """
        Test a forward pass through the model with dummy data.
        
        Args:
            batch_size: Batch size for the test
        """
        print(f"\nTesting forward pass with batch_size={batch_size}...")
        
        try:
            # Create dummy observation based on the observation space
            obs_space = self.agent.observation_space
            
            if hasattr(obs_space, 'spaces'):  # Dict space
                dummy_obs = {}
                for key, space in obs_space.spaces.items():
                    if hasattr(space, 'shape'):
                        dummy_obs[key] = torch.randn(batch_size, *space.shape, device=self.device)
                    else:
                        dummy_obs[key] = torch.randn(batch_size, space.n, device=self.device)
            else:  # Box space
                dummy_obs = torch.randn(batch_size, *obs_space.shape, device=self.device)
            
            # Forward pass
            with torch.no_grad():
                if isinstance(dummy_obs, dict):
                    # For dict observations, we need to use the policy's forward method
                    q_values = self.agent.policy.q_net(dummy_obs)
                else:
                    q_values = self.model(dummy_obs)
                
                print(f"✓ Forward pass successful!")
                print(f"  Input shape: {dummy_obs if not isinstance(dummy_obs, dict) else {k: v.shape for k, v in dummy_obs.items()}}")
                print(f"  Output shape: {q_values.shape}")
                print(f"  Output range: [{q_values.min().item():.4f}, {q_values.max().item():.4f}]")
                
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")


def main():
    """Main function to demonstrate the model loader."""
    parser = argparse.ArgumentParser(description="Load and inspect a DQN agent as a PyTorch model")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory or zip file")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to load the model on (cpu or cuda)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save model components (optional)")
    parser.add_argument("--test_forward", action="store_true",
                       help="Test a forward pass through the model")
    
    args = parser.parse_args()
    
    # Load the model
    print("Loading DQN agent...")
    loader = DQNModelLoader(args.agent_path, device=args.device)
    
    # Print model summary
    loader.print_model_summary()
    
    # Get the model for further use
    model = loader.get_model()
    print(f"\nModel ready for use: {type(model)}")
    print(f"You can now call model.state_dict() to inspect parameters")
    
    # Test forward pass if requested
    if args.test_forward:
        loader.test_forward_pass()
    
    # Save components if output directory is specified
    if args.output_dir:
        loader.save_model_components(args.output_dir)
    
    # Example usage
    print("\n" + "="*60)
    print("EXAMPLE USAGE")
    print("="*60)
    print("# Get the model")
    print("model = loader.get_model()")
    print()
    print("# Inspect state dict")
    print("state_dict = model.state_dict()")
    print("for name, param in state_dict.items():")
    print("    print(f'{name}: {param.shape}')")
    print()
    print("# Get specific parameters")
    print("first_layer_weights = state_dict['features_extractor.mlp.0.weight']")
    print("print(f'First layer weights shape: {first_layer_weights.shape}')")
    print()
    print("# Modify parameters")
    print("with torch.no_grad():")
    print("    state_dict['features_extractor.mlp.0.weight'][0, 0] = 1.0")
    print()
    print("# Load modified state dict back")
    print("model.load_state_dict(state_dict)")


if __name__ == "__main__":
    main() 