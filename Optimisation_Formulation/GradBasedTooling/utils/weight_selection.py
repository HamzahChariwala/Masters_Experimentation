"""
Weight selection utilities for neuron-level optimization.

Handles conversion from neuron indices to weight indices and manages
the scope of optimization variables.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import random


class WeightSelector:
    """
    Utility class for selecting and managing optimization variables.
    
    Handles conversion from neuron-level specifications to weight-level
    indices for optimization.
    """
    
    def __init__(self, 
                 target_layers: Optional[List[str]] = None,
                 weights_per_neuron: int = 64):
        """
        Initialize the weight selector.
        
        Args:
            target_layers: List of layer names to consider for selection
            weights_per_neuron: Number of weights per neuron (default 64 for DQN)
        """
        self.target_layers = target_layers or ['q_net.features_extractor.mlp.0', 'q_net.q_net.0', 'q_net.q_net.2']
        self.weights_per_neuron = weights_per_neuron
        self.layer_info = {}
        
    def analyze_model_structure(self, model: torch.nn.Module) -> Dict[str, Dict]:
        """
        Analyze the model structure to understand layer dimensions.
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            Dictionary mapping layer names to their structure information
        """
        layer_info = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if any(target in name for target in self.target_layers):
                    layer_info[name] = {
                        'shape': tuple(module.weight.shape),
                        'num_neurons': module.weight.shape[0],
                        'weights_per_neuron': module.weight.shape[1] if len(module.weight.shape) > 1 else 1,
                        'total_weights': module.weight.numel()
                    }
        
        self.layer_info = layer_info
        return layer_info
    
    def select_random_neurons(self, 
                             model: torch.nn.Module,
                             num_neurons: int,
                             seed: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Randomly select neurons from target layers.
        
        Args:
            model: PyTorch model
            num_neurons: Number of neurons to select
            seed: Random seed for reproducibility
            
        Returns:
            List of (layer_name, neuron_index) tuples
        """
        if seed is not None:
            random.seed(seed)
            
        # Analyze model if not done already
        if not self.layer_info:
            self.analyze_model_structure(model)
        
        # Collect all available neurons
        available_neurons = []
        for layer_name, info in self.layer_info.items():
            for neuron_idx in range(info['num_neurons']):
                available_neurons.append((layer_name, neuron_idx))
        
        # Randomly select neurons
        if num_neurons > len(available_neurons):
            raise ValueError(f"Requested {num_neurons} neurons but only {len(available_neurons)} available")
            
        selected_neurons = random.sample(available_neurons, num_neurons)
        return selected_neurons
    
    def select_neurons_by_config(self, 
                                model: torch.nn.Module,
                                num_neurons: int,
                                config: Dict,
                                seed: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Select neurons based on configuration parameters.
        
        Args:
            model: PyTorch model
            num_neurons: Number of neurons to select (for random/layer_balanced methods)
            config: Configuration dictionary with neuron_selection parameters
            seed: Random seed for reproducibility
            
        Returns:
            List of (layer_name, neuron_index) tuples
        """
        neuron_selection = config.get('neuron_selection', {})
        method = neuron_selection.get('method', 'random')
        
        if seed is not None:
            random.seed(seed)
            
        # Analyze model if not done already
        if not self.layer_info:
            self.analyze_model_structure(model)
        
        if method == 'specific':
            # Use specifically provided neurons
            specific_neurons = neuron_selection.get('specific_neurons', [])
            if not specific_neurons:
                raise ValueError("Method 'specific' requires 'specific_neurons' list in config")
            return specific_neurons
            
        elif method == 'metric':
            # Load neurons from circuit verification metric results
            from .integration_utils import load_neurons_from_metric
            
            metric = neuron_selection.get('metric')
            if not metric:
                raise ValueError("Method 'metric' requires 'metric' parameter in neuron_selection config")
            
            agent_path = config.get('agent_path')
            if not agent_path:
                raise ValueError("Method 'metric' requires 'agent_path' in config")
            
            return load_neurons_from_metric(agent_path, metric, num_neurons)
            
        elif method == 'layer_balanced':
            # Select specific number of neurons from each layer
            layer_distribution = neuron_selection.get('layer_distribution', {})
            if not layer_distribution:
                raise ValueError("Method 'layer_balanced' requires 'layer_distribution' dict in config")
            
            selected_neurons = []
            for layer_name, count in layer_distribution.items():
                if layer_name not in self.layer_info:
                    print(f"Warning: Layer '{layer_name}' not found in target layers, skipping")
                    continue
                    
                available_in_layer = [(layer_name, i) for i in range(self.layer_info[layer_name]['num_neurons'])]
                if count > len(available_in_layer):
                    raise ValueError(f"Requested {count} neurons from layer '{layer_name}' but only {len(available_in_layer)} available")
                
                selected_neurons.extend(random.sample(available_in_layer, count))
            
            return selected_neurons
            
        else:  # Default to 'random'
            return self._select_random_with_constraints(model, num_neurons, neuron_selection, seed)
    
    def _select_random_with_constraints(self,
                                       model: torch.nn.Module,
                                       num_neurons: int,
                                       neuron_selection: Dict,
                                       seed: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Select neurons randomly with optional constraints.
        
        Args:
            model: PyTorch model
            num_neurons: Number of neurons to select
            neuron_selection: Neuron selection configuration
            seed: Random seed for reproducibility
            
        Returns:
            List of (layer_name, neuron_index) tuples
        """
        if seed is not None:
            random.seed(seed)
        
        # Collect all available neurons
        available_neurons = []
        for layer_name, info in self.layer_info.items():
            for neuron_idx in range(info['num_neurons']):
                available_neurons.append((layer_name, neuron_idx))
        
        # Apply exclusions if specified
        exclude_neurons = neuron_selection.get('exclude_neurons', [])
        if exclude_neurons:
            available_neurons = [n for n in available_neurons if n not in exclude_neurons]
        
        # Apply layer prioritization if specified
        prioritize_layers = neuron_selection.get('prioritize_layers', [])
        if prioritize_layers:
            priority_neurons = [n for n in available_neurons if n[0] in prioritize_layers]
            other_neurons = [n for n in available_neurons if n[0] not in prioritize_layers]
            
            # Try to select from priority layers first
            selected_from_priority = min(num_neurons, len(priority_neurons))
            selected_neurons = random.sample(priority_neurons, selected_from_priority)
            
            # Fill remaining slots from other layers
            remaining_needed = num_neurons - selected_from_priority
            if remaining_needed > 0:
                if remaining_needed > len(other_neurons):
                    raise ValueError(f"Cannot select {num_neurons} neurons: only {len(available_neurons)} available after constraints")
                selected_neurons.extend(random.sample(other_neurons, remaining_needed))
            
            return selected_neurons
        
        # Standard random selection
        if num_neurons > len(available_neurons):
            raise ValueError(f"Requested {num_neurons} neurons but only {len(available_neurons)} available after constraints")
            
        return random.sample(available_neurons, num_neurons)
    
    def neurons_to_weight_indices(self, 
                                 model: torch.nn.Module,
                                 neuron_specs: List[Tuple[str, int]]) -> Dict[str, List[int]]:
        """
        Convert neuron specifications to weight indices.
        
        Args:
            model: PyTorch model
            neuron_specs: List of (layer_name, neuron_index) tuples
            
        Returns:
            Dictionary mapping layer names to lists of weight indices
        """
        weight_indices = {}
        
        for layer_name, neuron_idx in neuron_specs:
            if layer_name not in weight_indices:
                weight_indices[layer_name] = []
                
            # Get the layer module
            layer_module = dict(model.named_modules())[layer_name]
            
            if not hasattr(layer_module, 'weight'):
                continue
                
            weight_shape = layer_module.weight.shape
            
            # For linear layers: weight[neuron_idx, :] are the incoming weights
            if len(weight_shape) == 2:
                start_idx = neuron_idx * weight_shape[1]
                end_idx = start_idx + weight_shape[1]
                weight_indices[layer_name].extend(range(start_idx, end_idx))
            else:
                # For other layer types, may need different indexing
                # For now, assume flattened indexing
                weights_per_neuron = weight_shape[1] if len(weight_shape) > 1 else 1
                start_idx = neuron_idx * weights_per_neuron
                end_idx = start_idx + weights_per_neuron
                weight_indices[layer_name].extend(range(start_idx, end_idx))
        
        return weight_indices
    
    def create_optimization_vector(self, 
                                  model: torch.nn.Module,
                                  neuron_specs: List[Tuple[str, int]]) -> Tuple[np.ndarray, Dict]:
        """
        Create the optimization variable vector from neuron specifications.
        
        Args:
            model: PyTorch model
            neuron_specs: List of (layer_name, neuron_index) tuples
            
        Returns:
            Tuple of (initial_weights_vector, mapping_info)
            mapping_info contains information to reconstruct the model weights
        """
        weight_indices = self.neurons_to_weight_indices(model, neuron_specs)
        
        optimization_vector = []
        mapping_info = {
            'layer_slices': {},
            'original_shapes': {},
            'neuron_specs': neuron_specs
        }
        
        current_idx = 0
        
        for layer_name, indices in weight_indices.items():
            layer_module = dict(model.named_modules())[layer_name]
            original_weights = layer_module.weight.data.flatten().cpu().numpy()
            
            # Extract weights for this layer
            layer_weights = original_weights[indices]
            optimization_vector.extend(layer_weights)
            
            # Store mapping information
            layer_end_idx = current_idx + len(layer_weights)
            mapping_info['layer_slices'][layer_name] = (current_idx, layer_end_idx)
            mapping_info['original_shapes'][layer_name] = layer_module.weight.shape
            
            current_idx = layer_end_idx
        
        return np.array(optimization_vector), mapping_info
    
    def apply_perturbations(self, 
                           model: torch.nn.Module,
                           perturbations: np.ndarray,
                           mapping_info: Dict) -> torch.nn.Module:
        """
        Apply weight perturbations to the model.
        
        Args:
            model: PyTorch model to modify
            perturbations: Vector of weight changes
            mapping_info: Mapping information from create_optimization_vector
            
        Returns:
            Modified model (modifies in-place and returns for convenience)
        """
        model_copy = model  # Work with the original model for now
        
        for layer_name, (start_idx, end_idx) in mapping_info['layer_slices'].items():
            layer_perturbations = perturbations[start_idx:end_idx]
            layer_module = dict(model_copy.named_modules())[layer_name]
            
            # Get weight indices for this layer
            neuron_specs_for_layer = [(ln, ni) for ln, ni in mapping_info['neuron_specs'] if ln == layer_name]
            weight_indices = self.neurons_to_weight_indices(model_copy, neuron_specs_for_layer)[layer_name]
            
            # Apply perturbations
            with torch.no_grad():
                flattened_weights = layer_module.weight.data.flatten()
                flattened_weights[weight_indices] += torch.from_numpy(layer_perturbations).float()
                layer_module.weight.data = flattened_weights.reshape(mapping_info['original_shapes'][layer_name])
        
        return model_copy 