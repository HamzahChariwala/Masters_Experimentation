import os
import sys
import torch
import numpy as np
import json
from typing import Dict, List, Union, Optional, Tuple, Any, Set

# Add the project root to the Python path to ensure Environment_Tooling is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
neuron_selection_dir = os.path.dirname(script_dir)
project_root = os.path.abspath(os.path.join(neuron_selection_dir, ".."))

# Add both directories to path if they're not already there
for path in [neuron_selection_dir, project_root]:
    if not path in sys.path:
        sys.path.insert(0, path)
        print(f"Added to Python path: {path}")

from stable_baselines3 import DQN
from .activation_loader import ActivationLoader
from .advanced_patcher import AdvancedPatcher

# Import custom feature extractor - now accessible since we've added project_root to sys.path
from Environment_Tooling.BespokeEdits.FeatureExtractor import CustomCombinedExtractor

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
        # Ensure agent_path is absolute
        if not os.path.isabs(agent_path):
            # Convert relative path to absolute based on current working directory
            agent_path = os.path.abspath(agent_path)
            print(f"Using absolute agent path: {agent_path}")
            
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
        # Use the project root to find the agent if it's a relative path from the project root
        print(f"Agent path: {self.agent_path}")
        print(f"Agent path exists: {os.path.exists(self.agent_path)}")
        
        if not os.path.isabs(self.agent_path):
            # First try assuming the path is relative to the project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            print(f"Project root: {project_root}")
            model_path = os.path.join(project_root, self.agent_path, "agent.zip")
            print(f"Trying model path: {model_path}")
            print(f"Model path exists: {os.path.exists(model_path)}")
            
            if not os.path.exists(model_path):
                # If not found, try the path as provided
                model_path = os.path.join(self.agent_path, "agent.zip")
                print(f"Trying alternative model path: {model_path}")
                print(f"Model path exists: {os.path.exists(model_path)}")
        else:
            # Absolute path
            model_path = os.path.join(self.agent_path, "agent.zip")
            print(f"Using absolute model path: {model_path}")
            print(f"Model path exists: {os.path.exists(model_path)}")
            
        if not os.path.exists(model_path):
            # Try to list the contents of the agent directory to see what's there
            try:
                agent_dir_contents = os.listdir(self.agent_path)
                print(f"Agent directory contents: {agent_dir_contents}")
            except Exception as e:
                print(f"Error listing agent directory: {e}")
                
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        print(f"Loading agent from {model_path}")
        
        # Load the agent with the custom feature extractor
        self.agent = DQN.load(
            model_path, 
            device=self.device,
            custom_objects={"features_extractor_class": CustomCombinedExtractor}
        )
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
        # Handle various path formats for source activation file
        if os.path.isabs(file_name):
            # Absolute path is provided directly
            file_path = file_name
        elif file_name.startswith("../"):
            # Path is relative to the current script's location
            file_path = os.path.abspath(file_name)
        else:
            # Path is relative to the agent_path/activation_logging directory
            # Check if "activation_logging" is already in the path
            if "activation_logging" in file_name:
                file_path = os.path.join(self.agent_path, file_name)
            else:
                file_path = os.path.join(self.activation_dir, file_name)
        
        # Verify the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Activation file not found: {file_path}")
            
        # Use the activation loader to load the activations
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
        
        # Handle various path formats for target input file
        if os.path.isabs(target_input_file):
            # Absolute path is provided directly
            input_file_path = target_input_file
        elif target_input_file.startswith("../"):
            # Path is relative to the current script's location
            input_file_path = os.path.abspath(target_input_file)
        else:
            # Path is relative to the agent_path/activation_inputs directory
            # Check if "activation_inputs" is already in the path
            if "activation_inputs" in target_input_file:
                input_file_path = os.path.join(self.agent_path, target_input_file)
            else:
                input_file_path = os.path.join(self.agent_path, "activation_inputs", target_input_file)
        
        # Verify the input file exists
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Target input file not found: {input_file_path}")
        
        # Load target inputs
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
    
    def save_results_by_input(self, all_results: List[Dict[str, Dict[str, Any]]], 
                             patch_configs: List[Dict[str, Union[List[int], str]]],
                             experiment_names: List[str],
                             output_prefix: str,
                             output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save results organized by input ID rather than by patch configuration.
        Each input gets its own file containing results from all patch experiments.
        
        Args:
            all_results: List of results from multiple patching experiments
            patch_configs: List of patch configurations used for each experiment
            experiment_names: List of names for each experiment
            output_prefix: Prefix for output files
            output_dir: Optional directory to save results in (defaults to self.output_dir)
            
        Returns:
            Dictionary mapping input IDs to their output file paths
        """
        # First, collect all unique input IDs across all experiments
        all_input_ids = set()
        for results in all_results:
            all_input_ids.update(results.keys())
        
        # Create a new structure organized by input ID
        input_organized_results = {}
        output_files = {}
        
        # Use the specified output directory or default
        save_dir = output_dir if output_dir else self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        for input_id in all_input_ids:
            input_organized_results[input_id] = {}
            
            # Add results from each experiment for this input
            for i, (results, patch_config, exp_name) in enumerate(zip(all_results, patch_configs, experiment_names)):
                if input_id in results:
                    # Store the result with experiment metadata
                    input_organized_results[input_id][exp_name] = {
                        "patch_configuration": patch_config,
                        "results": results[input_id]
                    }
            
            # Create a sanitized input ID for the filename (remove special chars)
            safe_input_id = "".join(c if c.isalnum() else "_" for c in input_id)
            # Directly use the sanitized input ID as the filename without prefix
            if output_prefix:
                output_file = f"{safe_input_id}.json"
            else:
                output_file = f"{safe_input_id}.json"
            
            output_path = os.path.join(save_dir, output_file)
            
            # Save this input's results to its own file
            with open(output_path, "w") as f:
                # Convert any non-serializable types
                serializable_results = self._make_serializable(input_organized_results[input_id])
                json.dump(serializable_results, f, indent=2)
            
            output_files[input_id] = output_path
            print(f"Results for input {input_id} saved to {output_path}")
        
        return output_files
    
    def _make_serializable(self, data: Any) -> Any:
        """
        Recursively convert data to JSON-serializable types.
        
        Args:
            data: Data to convert
            
        Returns:
            JSON-serializable version of the data
        """
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, bool):
            return bool(data)
        elif isinstance(data, (int, float, str, type(None))):
            return data
        else:
            return str(data)
    
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
    
    def run_bidirectional_patching(self,
                                  clean_input_file: str,
                                  corrupted_input_file: str,
                                  clean_activations_file: str,
                                  corrupted_activations_file: str,
                                  patch_configs: List[Dict[str, Union[List[int], str]]],
                                  input_ids: Optional[List[str]] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Run patching experiments in both directions:
        1. Denoising: Clean activations patched into corrupted inputs
        2. Noising: Corrupted activations patched into clean inputs
        
        Args:
            clean_input_file: Path to the clean inputs JSON file
            corrupted_input_file: Path to the corrupted inputs JSON file
            clean_activations_file: Path to the clean activations NPZ file
            corrupted_activations_file: Path to the corrupted activations NPZ file
            patch_configs: List of patch configurations to test
            input_ids: Optional list of input IDs to process (if None, process all)
            
        Returns:
            Tuple of (denoising_output_files, noising_output_files)
        """
        # Create denoising and noising directories
        denoising_dir = os.path.join(self.output_dir, "denoising")
        noising_dir = os.path.join(self.output_dir, "noising")
        os.makedirs(denoising_dir, exist_ok=True)
        os.makedirs(noising_dir, exist_ok=True)
        
        print("\n==== Running DENOISING experiments (clean activations → corrupted inputs) ====")
        # Run denoising experiments (clean activations → corrupted inputs)
        denoising_results = []
        patch_configs_list = []
        experiment_names = []
        
        for i, patch_spec in enumerate(patch_configs):
            if not patch_spec:
                print(f"Warning: Empty patch specification at index {i}. Skipping.")
                continue
                
            # Create descriptive experiment name using new naming system (0-based indexing)
            exp_name = self.generate_experiment_name(i, patch_spec)
            experiment_names.append(exp_name)
            
            # Print experiment details
            print(f"\nRunning denoising experiment {i+1}/{len(patch_configs)} ({exp_name}):")
            print(f"  Target input: {corrupted_input_file}")
            print(f"  Source activations: {clean_activations_file}")
            print(f"  Patching: {patch_spec}")
            print("-" * 50)
            
            try:
                # Run experiment
                results = self.run_patching_experiment(
                    corrupted_input_file,
                    clean_activations_file,
                    patch_spec,
                    input_ids
                )
                
                # Add patch configuration to results for reference
                for input_id in results:
                    results[input_id]["patch_configuration"] = patch_spec
                
                # Collect results for later organization by input
                denoising_results.append(results)
                patch_configs_list.append(patch_spec)
                
                # Print summary
                changed_count = sum(1 for result in results.values() if result.get("action_changed", False))
                total_count = len(results)
                print(f"\nDenoising experiment {i+1} completed.")
                print(f"Summary: {changed_count}/{total_count} actions changed due to patching")
                
            except Exception as e:
                print(f"Error running denoising experiment {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save denoising results
        if denoising_results:
            print("\nSaving denoising results organized by input ID...")
            denoising_output_files = self.save_results_by_input(
                denoising_results,
                patch_configs_list,
                experiment_names,
                "",  # No prefix
                denoising_dir
            )
            print(f"Saved denoising results for {len(denoising_output_files)} inputs")
        else:
            denoising_output_files = {}
        
        # Reset experiment names for noising experiments
        experiment_names = []
        patch_configs_list = []
        
        print("\n==== Running NOISING experiments (corrupted activations → clean inputs) ====")
        # Run noising experiments (corrupted activations → clean inputs)
        noising_results = []
        
        for i, patch_spec in enumerate(patch_configs):
            if not patch_spec:
                print(f"Warning: Empty patch specification at index {i}. Skipping.")
                continue
                
            # Create descriptive experiment name using new naming system (0-based indexing)
            exp_name = self.generate_experiment_name(i, patch_spec)
            experiment_names.append(exp_name)
            
            # Print experiment details
            print(f"\nRunning noising experiment {i+1}/{len(patch_configs)} ({exp_name}):")
            print(f"  Target input: {clean_input_file}")
            print(f"  Source activations: {corrupted_activations_file}")
            print(f"  Patching: {patch_spec}")
            print("-" * 50)
            
            try:
                # Run experiment
                results = self.run_patching_experiment(
                    clean_input_file,
                    corrupted_activations_file,
                    patch_spec,
                    input_ids
                )
                
                # Add patch configuration to results for reference
                for input_id in results:
                    results[input_id]["patch_configuration"] = patch_spec
                
                # Collect results for later organization by input
                noising_results.append(results)
                patch_configs_list.append(patch_spec)
                
                # Print summary
                changed_count = sum(1 for result in results.values() if result.get("action_changed", False))
                total_count = len(results)
                print(f"\nNoising experiment {i+1} completed.")
                print(f"Summary: {changed_count}/{total_count} actions changed due to patching")
                
            except Exception as e:
                print(f"Error running noising experiment {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save noising results
        if noising_results:
            print("\nSaving noising results organized by input ID...")
            noising_output_files = self.save_results_by_input(
                noising_results,
                patch_configs_list,
                experiment_names,
                "",  # No prefix
                noising_dir
            )
            print(f"Saved noising results for {len(noising_output_files)} inputs")
        else:
            noising_output_files = {}
        
        return denoising_output_files, noising_output_files
    
    def generate_experiment_name(self, experiment_index: int, patch_spec: Dict[str, Union[List[int], str]]) -> str:
        """
        Generate a descriptive experiment name based on the patch specification.
        Uses actual neuron indices within layers instead of arbitrary counters.
        
        Args:
            experiment_index: The 0-based index of the experiment
            patch_spec: Dictionary specifying which neurons to patch
            
        Returns:
            Descriptive experiment name (e.g., "q_net.0_neuron_5" or "q_net.2_neurons_3_7_12")
        """
        if not patch_spec:
            return f"exp_{experiment_index}"
        
        # Handle single layer, single neuron case
        if len(patch_spec) == 1:
            layer_name = list(patch_spec.keys())[0]
            neurons = patch_spec[layer_name]
            
            if isinstance(neurons, list) and len(neurons) == 1:
                # Single neuron: "q_net.0_neuron_5"
                return f"{layer_name}_neuron_{neurons[0]}"
            elif isinstance(neurons, list) and len(neurons) <= 5:
                # Few neurons: "q_net.0_neurons_3_7_12"
                neuron_str = "_".join(map(str, sorted(neurons)))
                return f"{layer_name}_neurons_{neuron_str}"
            elif isinstance(neurons, str) and neurons == "all":
                # All neurons: "q_net.0_all"
                return f"{layer_name}_all"
            else:
                # Many neurons: "q_net.0_group_25"
                return f"{layer_name}_group_{len(neurons)}"
        
        # Handle multiple layers
        else:
            layer_parts = []
            for layer_name, neurons in patch_spec.items():
                if isinstance(neurons, list) and len(neurons) == 1:
                    layer_parts.append(f"{layer_name}_{neurons[0]}")
                elif isinstance(neurons, list) and len(neurons) <= 3:
                    neuron_str = "_".join(map(str, sorted(neurons)))
                    layer_parts.append(f"{layer_name}_{neuron_str}")
                elif isinstance(neurons, str) and neurons == "all":
                    layer_parts.append(f"{layer_name}_all")
                else:
                    layer_parts.append(f"{layer_name}_{len(neurons)}n")
            
            # Combine all layer parts
            combined_name = "_".join(layer_parts)
            
            # If the name gets too long, use a fallback
            if len(combined_name) > 100:
                layer_names = "_".join(patch_spec.keys())
                return f"exp_{experiment_index}_{layer_names}"
            
            return combined_name 