"""
Integration utilities for working with existing StateTooling.

Provides functions to load state data from the existing optimization
state filter results and integrate with other project components.
"""

import os
import json
import sys
from typing import Dict, List, Tuple, Optional, Any
import random

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)


def load_states_from_tooling(agent_path: str, 
                           alter_sample_size: Optional[int] = None,
                           preserve_sample_size: Optional[int] = None) -> Tuple[Dict, Dict]:
    """
    Load ALTER and PRESERVE states from StateTooling results.
    Only includes ALTER states that have 'manually_verified': true.
    
    Args:
        agent_path: Path to the agent directory
        alter_sample_size: Number of ALTER states to sample (None for all)
        preserve_sample_size: Number of PRESERVE states to sample (None for all)
        
    Returns:
        Tuple of (alter_states, preserve_states) dictionaries
    """
    # Load ALTER states
    alter_path = os.path.join(agent_path, "optimisation_states", "alter", "states.json")
    if not os.path.exists(alter_path):
        raise FileNotFoundError(f"ALTER states file not found: {alter_path}")
    
    with open(alter_path, 'r') as f:
        all_alter_states = json.load(f)
    
    # Filter ALTER states for manually verified only
    alter_states = {}
    for state_id, state_data in all_alter_states.items():
        if state_data.get('manually_verified', False):
            alter_states[state_id] = state_data
    
    print(f"Loaded {len(alter_states)} manually verified ALTER states (out of {len(all_alter_states)} total)")
    
    # Sample if requested
    if alter_sample_size is not None and len(alter_states) > alter_sample_size:
        sampled_keys = random.sample(list(alter_states.keys()), alter_sample_size)
        alter_states = {k: alter_states[k] for k in sampled_keys}
        print(f"Sampled {len(alter_states)} ALTER states")
    
    # Load PRESERVE states (all of them, no manual verification filter)
    preserve_path = os.path.join(agent_path, "optimisation_states", "preserve", "states.json")
    if not os.path.exists(preserve_path):
        raise FileNotFoundError(f"PRESERVE states file not found: {preserve_path}")
    
    with open(preserve_path, 'r') as f:
        preserve_states = json.load(f)
    
    print(f"Loaded {len(preserve_states)} PRESERVE states")
    
    # Sample if requested
    if preserve_sample_size is not None and len(preserve_states) > preserve_sample_size:
        sampled_keys = random.sample(list(preserve_states.keys()), preserve_sample_size)
        preserve_states = {k: preserve_states[k] for k in sampled_keys}
        print(f"Sampled {len(preserve_states)} PRESERVE states")
    
    return alter_states, preserve_states


def load_target_actions_from_tooling(alter_states: Dict) -> Dict[str, List[int]]:
    """
    Extract target actions from ALTER states.
    Only processes states that have 'manually_verified': true.
    
    Args:
        alter_states: Dictionary of ALTER states (already filtered for manual verification)
        
    Returns:
        Dictionary mapping state_id to list of desired action indices
    """
    target_actions = {}
    
    for state_id, state_data in alter_states.items():
        # Ensure this is a manually verified state (should already be filtered)
        if not state_data.get('manually_verified', False):
            print(f"Warning: Skipping non-manually verified state {state_id}")
            continue
            
        # Extract desired actions
        desired_action = state_data.get('desired_action')
        if desired_action is None:
            print(f"Warning: No 'desired_action' found for state {state_id}")
            continue
            
        if not isinstance(desired_action, list):
            print(f"Warning: 'desired_action' for state {state_id} is not a list: {desired_action}")
            continue
            
        target_actions[state_id] = desired_action
    
    print(f"Extracted target actions for {len(target_actions)} manually verified ALTER states")
    return target_actions


def validate_neuron_indices(model: Any, 
                          neuron_specs: List[Tuple[str, int]],
                          target_layers: List[str]) -> bool:
    """
    Validate that neuron specifications are valid for the given model.
    
    Args:
        model: PyTorch model
        neuron_specs: List of (layer_name, neuron_index) tuples
        target_layers: List of allowed layer names
        
    Returns:
        True if all specifications are valid, False otherwise
    """
    try:
        model_layers = dict(model.named_modules())
        
        for layer_name, neuron_idx in neuron_specs:
            # Check if layer exists
            if layer_name not in model_layers:
                print(f"Layer {layer_name} not found in model")
                return False
                
            # Check if layer is in target layers
            if not any(target in layer_name for target in target_layers):
                print(f"Layer {layer_name} not in target layers: {target_layers}")
                return False
                
            # Check if neuron index is valid
            layer_module = model_layers[layer_name]
            if hasattr(layer_module, 'weight') and layer_module.weight is not None:
                num_neurons = layer_module.weight.shape[0]
                if neuron_idx >= num_neurons:
                    print(f"Neuron index {neuron_idx} out of range for layer {layer_name} (max: {num_neurons-1})")
                    return False
        
        return True
        
    except Exception as e:
        print(f"Error validating neuron indices: {e}")
        return False


def load_model_from_agent_path(agent_path: str):
    """
    Load a DQN model from the agent directory using SB3.
    
    Args:
        agent_path: Path to the agent directory
        
    Returns:
        Loaded PyTorch model
    """
    import sys
    import os
    
    # Add project root to path for imports (same as done in agent_functionality.py)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    sys.path.insert(0, project_root)
    
    try:
        from stable_baselines3 import DQN
        from Environment_Tooling.BespokeEdits.FeatureExtractor import CustomCombinedExtractor
        
        # Check if agent_path already ends with agent.zip
        if agent_path.endswith("agent.zip"):
            zip_path = agent_path
        else:
            # Make sure the path doesn't have trailing slashes
            agent_path = agent_path.rstrip('/')
            zip_path = os.path.join(agent_path, "agent.zip")
        
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Agent file not found: {zip_path}")
        
        print(f"Loading DQN agent from: {zip_path}")
        
        # Load the model using SB3 with custom feature extractor
        model = DQN.load(
            zip_path,
            custom_objects={
                "features_extractor_class": CustomCombinedExtractor
            }
        )
        
        print(f"Successfully loaded DQN model")
        return model.policy  # Return the PyTorch policy network
        
    except Exception as e:
        print(f"Error loading model from {agent_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_preserve_constraints(preserve_states: Dict[str, Dict],
                              model: Any,
                              weight_selector) -> List[Dict]:
    """
    Create constraint specifications for preserving behavior on PRESERVE states.
    
    Args:
        preserve_states: Dictionary of preserve state data
        model: PyTorch model
        weight_selector: WeightSelector instance
        
    Returns:
        List of constraint dictionaries for the solver
    """
    constraints = []
    
    # Get original Q-values for preserve states
    model.eval()
    original_argmax = {}
    
    with torch.no_grad():
        for state_id, state_data in preserve_states.items():
            if 'observation' in state_data:
                obs = state_data['observation']
                if isinstance(obs, (list, np.ndarray)):
                    state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                elif isinstance(obs, torch.Tensor):
                    state_tensor = obs.float().unsqueeze(0)
                else:
                    continue
                    
                q_values = model(state_tensor)
                original_argmax[state_id] = torch.argmax(q_values).item()
    
    # Create constraint function
    def preserve_constraint(weight_perturbations, mapping_info):
        """Constraint function to ensure argmax preservation."""
        # Reset and apply perturbations
        weight_selector.apply_perturbations(model, weight_perturbations, mapping_info)
        
        violations = 0
        model.eval()
        
        with torch.no_grad():
            for state_id, state_data in preserve_states.items():
                if state_id not in original_argmax:
                    continue
                    
                if 'observation' in state_data:
                    obs = state_data['observation']
                    if isinstance(obs, (list, np.ndarray)):
                        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    elif isinstance(obs, torch.Tensor):
                        state_tensor = obs.float().unsqueeze(0)
                    else:
                        continue
                        
                    q_values = model(state_tensor)
                    current_argmax = torch.argmax(q_values).item()
                    
                    # Constraint violation if argmax changed
                    if current_argmax != original_argmax[state_id]:
                        violations += 1
        
        # Return 0 if no violations (constraint satisfied), positive if violated
        return violations
    
    constraints.append({
        'type': 'eq',
        'fun': preserve_constraint
    })
    
    return constraints 