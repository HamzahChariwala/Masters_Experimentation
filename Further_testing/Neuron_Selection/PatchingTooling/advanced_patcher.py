import torch
from torch import nn
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
import numpy as np
import copy

class AdvancedPatcher:
    """
    Enhanced patching system that supports different patching strategies and
    activation logging during forward passes.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize the patcher.
        
        Args:
            model: The PyTorch model to patch (e.g., model.policy.q_net)
        """
        self.model = model
        self.hooks = {}
        self.orig_forwards = {}
        self.layer_activations = {}
        self.patched_layers = set()
        
    def register_activation_hooks(self, target_layers: List[str]) -> None:
        """
        Register forward hooks to capture activations for target layers.
        
        Args:
            target_layers: List of layer names to capture activations for
        """
        # Clear any existing hooks
        self.remove_hooks()
        self.layer_activations = {}
        
        # Define hook function
        def hook_fn(name):
            def _hook(module, input, output):
                self.layer_activations[name] = output.detach().cpu()
            return _hook
        
        # Register hooks for all target layers
        for name, module in self.model.named_modules():
            if name in target_layers:
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks[name] = hook
                print(f"Registered activation hook for layer: {name}")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}
    
    def patch_layers(self, 
                    source_activations: Dict[str, torch.Tensor], 
                    patch_spec: Dict[str, Union[List[int], str]]) -> None:
        """
        Patch specific layers and neurons with source activations.
        
        Args:
            source_activations: Dictionary mapping layer_name -> activation tensor
            patch_spec: Dictionary mapping layer_name -> patching specification
                        Value can be:
                        - List[int]: List of specific neuron indices to patch
                        - "all": Patch the entire layer output
        """
        # Restore any previous patching
        self.restore_all()
        
        # For each layer to patch
        for layer_name, patch_type in patch_spec.items():
            # Find the corresponding module
            module = None
            for name, mod in self.model.named_modules():
                if name == layer_name:
                    module = mod
                    break
            
            if module is None:
                print(f"Warning: Layer {layer_name} not found in model.")
                continue
            
            # Skip if we don't have activations for this layer
            if layer_name not in source_activations:
                print(f"Warning: No source activations for layer {layer_name}")
                continue
            
            # Get source activations for this layer
            source_act = source_activations[layer_name]
            
            # Save original forward
            self.orig_forwards[layer_name] = module.forward
            
            # Create patched forward function
            if patch_type == "all":
                # Patch entire layer
                def patched_forward(module_name, src_activation):
                    def _forward(x):
                        # Call original to maintain gradients if needed
                        _ = self.orig_forwards[module_name](x)
                        # Return the source activation instead
                        return src_activation
                    return _forward
                
                module.forward = patched_forward(layer_name, source_act)
                
            else:
                # Patch specific neurons
                neuron_indices = patch_type if isinstance(patch_type, list) else [int(patch_type)]
                
                def patched_forward(module_name, src_activation, indices):
                    def _forward(x):
                        # Compute original output
                        out = self.orig_forwards[module_name](x)
                        # Clone to avoid modifying autograd tensors in-place
                        out = out.clone()
                        # Replace specified neuron activations
                        for idx in indices:
                            if idx < out.shape[1]:
                                out[:, idx] = src_activation[:, idx]
                        return out
                    return _forward
                
                module.forward = patched_forward(layer_name, source_act, neuron_indices)
            
            # Mark as patched
            self.patched_layers.add(layer_name)
            print(f"Patched layer {layer_name} with " + 
                  ("all neurons" if patch_type == "all" else f"neurons {patch_type}"))
    
    def restore_layer(self, layer_name: str) -> None:
        """
        Restore original forward function for a specific layer.
        
        Args:
            layer_name: Name of the layer to restore
        """
        if layer_name in self.orig_forwards:
            # Find the module
            for name, module in self.model.named_modules():
                if name == layer_name:
                    # Restore original forward
                    module.forward = self.orig_forwards[layer_name]
                    # Remove from patched layers
                    self.patched_layers.discard(layer_name)
                    print(f"Restored original forward for layer: {layer_name}")
                    break
    
    def restore_all(self) -> None:
        """Restore all patched layers to their original forward functions."""
        for layer_name in list(self.patched_layers):
            self.restore_layer(layer_name)
        self.orig_forwards = {}
        self.patched_layers = set()
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """
        Get all captured activations from the current forward pass.
        
        Returns:
            Dictionary mapping layer_name -> activation tensor
        """
        return copy.deepcopy(self.layer_activations)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.remove_hooks()
        self.restore_all() 