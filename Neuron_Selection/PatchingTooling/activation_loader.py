import os
import numpy as np
import json
import torch
from typing import Dict, List, Union, Optional, Tuple, Any

class ActivationLoader:
    """
    Utility class for loading activation data from NPZ and JSON files
    produced by the activation_extraction.py script.
    """
    
    @staticmethod
    def load_activations_from_npz(file_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load activations from an NPZ file.
        
        Args:
            file_path: Path to the NPZ file containing activations
            
        Returns:
            Dictionary mapping input_id -> layer_name -> activation tensor
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Activation file not found: {file_path}")
        
        # Load the NPZ file
        data = np.load(file_path)
        
        # Organize by input_id
        result = {}
        
        # NPZ files from activation_extraction.py use keys like "input_id_layer_name"
        for key in data.keys():
            # Skip any non-standard keys
            if "_" not in key:
                continue
                
            # Extract input_id and layer_name
            parts = key.split("_", 1)  # Split only on the first underscore
            input_id = parts[0]
            layer_name = parts[1]
            
            # Initialize the input dictionary if it doesn't exist
            if input_id not in result:
                result[input_id] = {}
                
            # Store the activation
            result[input_id][layer_name] = data[key]
        
        return result
    
    @staticmethod
    def load_activations_from_json(file_path: str) -> Dict[str, Dict[str, List]]:
        """
        Load activations from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing activations
            
        Returns:
            Dictionary mapping input_id -> layer_name -> activation tensor
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Activation file not found: {file_path}")
        
        # Load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    @staticmethod
    def load_activations(file_path: str) -> Dict[str, Dict[str, Union[np.ndarray, List]]]:
        """
        Load activations from either NPZ or JSON file based on file extension.
        
        Args:
            file_path: Path to the file containing activations
            
        Returns:
            Dictionary mapping input_id -> layer_name -> activation tensor
        """
        if file_path.endswith('.npz'):
            return ActivationLoader.load_activations_from_npz(file_path)
        elif file_path.endswith('.json'):
            return ActivationLoader.load_activations_from_json(file_path)
        else:
            raise ValueError(f"Unsupported file extension. Must be .npz or .json: {file_path}")
    
    @staticmethod
    def get_activation_for_input(activations: Dict, input_id: str) -> Dict[str, Union[np.ndarray, List]]:
        """
        Extract activations for a specific input ID.
        
        Args:
            activations: Dictionary of activations as loaded by load_activations
            input_id: The input ID to extract
            
        Returns:
            Dictionary mapping layer_name -> activation tensor
        """
        if input_id not in activations:
            raise KeyError(f"Input ID not found in activations: {input_id}")
        
        return activations[input_id]
    
    @staticmethod
    def convert_to_torch(activations: Dict[str, Union[np.ndarray, List]]) -> Dict[str, torch.Tensor]:
        """
        Convert NumPy arrays or lists to PyTorch tensors.
        
        Args:
            activations: Dictionary mapping layer_name -> activation data
            
        Returns:
            Dictionary mapping layer_name -> PyTorch tensor
        """
        result = {}
        
        for layer_name, activation in activations.items():
            # Handle both NumPy arrays and lists
            if isinstance(activation, np.ndarray):
                result[layer_name] = torch.from_numpy(activation)
            elif isinstance(activation, list):
                # Convert nested lists to numpy array first
                result[layer_name] = torch.tensor(activation, dtype=torch.float32)
            else:
                # Try generic conversion
                result[layer_name] = torch.tensor(activation, dtype=torch.float32)
        
        return result 