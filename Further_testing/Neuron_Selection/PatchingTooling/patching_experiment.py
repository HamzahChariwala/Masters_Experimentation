import os
import torch
import numpy as np
import json
from typing import Dict, List, Union, Optional, Tuple, Any, Set
from stable_baselines3 import DQN
from .activation_loader import ActivationLoader
from .advanced_patcher import AdvancedPatcher

class PatchingExperiment:
    """
    Manages patching experiments, including loading activations,
    running patched forward passes, and saving results.
    """
    
    def __init__(self, agent_path: str, device: str = "cpu"):
        """
        Initialize the patching experiment.
        
        Args:
            agent_path: Path to the agent directory
            device: Device to run the experiment on ("cpu" or "cuda")
        """
        self.agent_path = agent_path
        self.device = device
        self.agent = None
        self.q_net = None
        self.patcher = None
        self.activation_dir = os.path.join(agent_path, "activation_logging")
        self.output_dir = os.path.join(agent_path, "patching_results")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # List of layers to monitor for activations
        self.target_layers = [
            "q_net.0",
            "q_net.2", 
            "q_net.4",
            "features_extractor.mlp.0"
        ]
    
    def load_agent(self) -> None:
        """Load the agent model from the agent path."""
        model_path = os.path.join(self.agent_path, "agent.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        print(f"Loading agent from {model_path}")
        self.agent = DQN.load(model_path, device=self.device)
        self.q_net = self.agent.policy.q_net
        
        # Initialize the patcher
        self.patcher = AdvancedPatcher(self.q_net)
        
        # Register activation hooks for target layers
        self.patcher.register_activation_hooks(self.target_layers)
    
    def load_activations(self, file_name: str) -> Dict[str, Dict[str, Union[np.ndarray, List]]]:
        """
        Load activations from a file in the activation_logging directory.
        
        Args:
            file_name: Name of the activation file (e.g., "clean_activations.npz")
            
        Returns:
            Dictionary of activations organized by input_id and layer_name
        """
        file_path = os.path.join(self.activation_dir, file_name)
        return ActivationLoader.load_activations(file_path)
    
    def run_patching_experiment(self, 
                              target_input_file: str,
                              source_activation_file: str,
                              patch_spec: Dict[str, Union[List[int], str]],
                              input_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run a patching experiment by patching activations from one source into a target input.
        
        Args:
            target_input_file: Path to the JSON file with target inputs
            source_activation_file: Path to the file with source activations to patch in
            patch_spec: Dictionary mapping layer_name -> patching specification
            input_ids: Optional list of input IDs to process (if None, process all)
            
        Returns:
            Dictionary of results for each input ID
        """
        # Ensure agent is loaded
        if self.agent is None:
            self.load_agent()
        
        # Load target inputs
        input_file_path = os.path.join(self.agent_path, "activation_inputs", target_input_file)
        with open(input_file_path, "r") as f:
            target_inputs = json.load(f)
        
        # Load source activations
        source_activations_all = self.load_activations(source_activation_file)
        
        # Determine which input IDs to process
        if input_ids is None:
            input_ids = list(target_inputs.keys())
        
        # Results will store the experiment outcomes
        results = {}
        
        # Process each input
        for input_id in input_ids:
            print(f"\nProcessing input: {input_id}")
            
            # Check if input ID exists in target inputs
            if input_id not in target_inputs:
                print(f"Warning: Input ID {input_id} not found in target inputs. Skipping.")
                continue
            
            # Check if input ID exists in source activations
            if input_id not in source_activations_all:
                print(f"Warning: Input ID {input_id} not found in source activations. Skipping.")
                continue
            
            # Extract input vector
            if isinstance(target_inputs[input_id], dict) and "input" in target_inputs[input_id]:
                input_vector = target_inputs[input_id]["input"]
            else:
                input_vector = target_inputs[input_id]
            
            # Convert to tensor
            obs_tensor = torch.FloatTensor(input_vector).unsqueeze(0).to(self.device)
            
            # Get source activations for this input
            source_activations = ActivationLoader.get_activation_for_input(source_activations_all, input_id)
            
            # Convert source activations to PyTorch tensors
            source_tensors = ActivationLoader.convert_to_torch(source_activations)
            
            # Run baseline forward pass (without patching)
            with torch.no_grad():
                # For this specific model, we need to manually process through each component
                # First through feature extractor
                features = None
                try:
                    # Try using the feature extractor directly (may not work due to dict input expectation)
                    features = self.agent.policy.q_net.features_extractor.mlp(obs_tensor)
                except Exception as e:
                    print(f"Direct feature extraction failed: {e}")
                    print("Attempting manual forward pass...")
                    # If that fails, process manually through each layer
                    for name, module in self.agent.policy.q_net.features_extractor.mlp.named_modules():
                        if isinstance(module, torch.nn.Linear) and name == '0':
                            # First linear layer
                            features = module(obs_tensor)
                        elif isinstance(module, torch.nn.ReLU) and name == '1' and features is not None:
                            # ReLU activation
                            features = module(features)
                
                if features is None:
                    raise ValueError("Failed to extract features")
                
                # Then through the q-network
                x = features
                for i, layer in enumerate(self.agent.policy.q_net.q_net):
                    x = layer(x)
                
                baseline_output = x.cpu().numpy()
                baseline_action = np.argmax(baseline_output)
                baseline_activations = self.patcher.get_activations()
            
            # Apply patching
            self.patcher.patch_layers(source_tensors, patch_spec)
            
            # Run patched forward pass
            with torch.no_grad():
                # Same process as baseline
                features = None
                try:
                    features = self.agent.policy.q_net.features_extractor.mlp(obs_tensor)
                except Exception as e:
                    # Process manually through each layer
                    for name, module in self.agent.policy.q_net.features_extractor.mlp.named_modules():
                        if isinstance(module, torch.nn.Linear) and name == '0':
                            features = module(obs_tensor)
                        elif isinstance(module, torch.nn.ReLU) and name == '1' and features is not None:
                            features = module(features)
                
                if features is None:
                    raise ValueError("Failed to extract features")
                
                # Then through the q-network
                x = features
                for i, layer in enumerate(self.agent.policy.q_net.q_net):
                    x = layer(x)
                
                patched_output = x.cpu().numpy()
                patched_action = np.argmax(patched_output)
                patched_activations = self.patcher.get_activations()
            
            # Restore original model
            self.patcher.restore_all()
            
            # Store results for this input
            results[input_id] = {
                "baseline_output": baseline_output.tolist(),
                "baseline_action": int(baseline_action),
                "patched_output": patched_output.tolist(),
                "patched_action": int(patched_action),
                "patched_layers": list(patch_spec.keys()),
                "action_changed": baseline_action != patched_action
            }
            
            # Print summary
            print(f"  Baseline action: {baseline_action}")
            print(f"  Patched action: {patched_action}")
            print(f"  Action changed: {baseline_action != patched_action}")
        
        return results
    
    def save_results(self, results: Dict[str, Dict[str, Any]], output_file: str) -> str:
        """
        Save experiment results to a JSON file.
        
        Args:
            results: Dictionary of results from run_patching_experiment
            output_file: Name of the output file
            
        Returns:
            Path to the saved file
        """
        output_path = os.path.join(self.output_dir, output_file)
        
        # Deep copy the results to avoid modifying the original
        serializable_results = {}
        
        # Ensure all values are JSON serializable
        for input_id, result_dict in results.items():
            serializable_results[input_id] = {}
            for key, value in result_dict.items():
                # Convert any non-serializable types
                if isinstance(value, (np.ndarray, list, tuple)):
                    serializable_results[input_id][key] = np.array(value).tolist()
                elif isinstance(value, bool):
                    serializable_results[input_id][key] = bool(value)
                elif isinstance(value, (int, float, str, type(None))):
                    serializable_results[input_id][key] = value
                else:
                    serializable_results[input_id][key] = str(value)
        
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        return output_path
    
    def save_all_activations(self, 
                           input_id: str, 
                           baseline_activations: Dict[str, torch.Tensor],
                           patched_activations: Dict[str, torch.Tensor],
                           output_file: str) -> str:
        """
        Save all activations (baseline and patched) to an NPZ file.
        
        Args:
            input_id: Input ID
            baseline_activations: Dictionary of baseline activations
            patched_activations: Dictionary of patched activations
            output_file: Name of the output file
            
        Returns:
            Path to the saved file
        """
        output_path = os.path.join(self.output_dir, output_file)
        
        # Convert torch tensors to numpy arrays
        save_dict = {}
        
        # Save baseline activations
        for layer_name, activation in baseline_activations.items():
            save_dict[f"{input_id}_baseline_{layer_name}"] = activation.cpu().numpy()
        
        # Save patched activations
        for layer_name, activation in patched_activations.items():
            save_dict[f"{input_id}_patched_{layer_name}"] = activation.cpu().numpy()
        
        # Save to NPZ file
        np.savez(output_path, **save_dict)
        
        print(f"Activations saved to {output_path}")
        return output_path 