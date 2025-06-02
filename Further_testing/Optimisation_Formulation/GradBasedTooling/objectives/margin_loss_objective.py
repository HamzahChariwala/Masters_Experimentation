"""
Margin-based loss objective for DQN optimization.

Implements margin-based loss that ensures target actions win by a specified margin,
with support for multiple acceptable actions per state.
"""

import sys
import os
from typing import Dict, List, Any, Optional, Union
import numpy as np
import torch
import copy

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.base_objective import BaseObjective


class MarginLossObjective(BaseObjective):
    """
    Margin-based objective function for DQN Q-value optimization.
    
    Minimizes: max(0, max_a∉acceptable(Q(s,a)) - max_a∈acceptable(Q(s,a)) + margin)
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 weight_selector,
                 alter_states: Dict[str, Dict],
                 target_actions: Dict[str, List[int]],
                 margin: float,
                 lambda_sparse: float = 0.0,
                 lambda_magnitude: float = 0.0,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the margin-based objective.
        
        Args:
            model: PyTorch DQN model
            weight_selector: WeightSelector instance for applying perturbations
            alter_states: Dictionary of states to modify {state_id: state_data}
            target_actions: Dictionary of target actions {state_id: [action_indices]}
            margin: Margin value for the loss function
            lambda_sparse: Coefficient for sparsity regularization
            lambda_magnitude: Coefficient for magnitude regularization
            config: Additional configuration parameters
        """
        super().__init__(config)
        
        self.model = model
        self.weight_selector = weight_selector
        self.alter_states = alter_states
        self.target_actions = target_actions
        self.margin = margin
        self.lambda_sparse = lambda_sparse
        self.lambda_magnitude = lambda_magnitude
        
        # Store original model state for perturbations (handle None model for demo)
        if model is not None:
            self.original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.original_state_dict = None
        
        # Preprocess states to tensors
        self._preprocess_states()
        
    def _preprocess_states(self):
        """Convert state data to tensors for efficient computation."""
        self.state_tensors = {}
        
        for state_id, state_data in self.alter_states.items():
            # The state data uses 'input' key for the observation (not 'observation')
            if 'input' in state_data:
                obs = state_data['input']
                if isinstance(obs, dict):
                    # Convert dict observation to format expected by model
                    self.state_tensors[state_id] = obs
                elif isinstance(obs, (list, np.ndarray)):
                    self.state_tensors[state_id] = torch.tensor(obs, dtype=torch.float32)
                elif isinstance(obs, torch.Tensor):
                    self.state_tensors[state_id] = obs.float()
                else:
                    raise ValueError(f"Unsupported observation type: {type(obs)}")
            else:
                print(f"Warning: State {state_id} has no 'input' key. Keys: {list(state_data.keys())}")
                # Create a dummy observation for demo purposes
                self.state_tensors[state_id] = torch.zeros(10, dtype=torch.float32)
    
    def _compute_q_values(self, weight_perturbations: np.ndarray, mapping_info: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute Q-values for all states with given weight perturbations.
        
        Args:
            weight_perturbations: Vector of weight changes
            mapping_info: Mapping information for applying perturbations
            
        Returns:
            Dictionary mapping state_id to Q-value tensors
        """
        if self.model is None:
            # Handle case where model is None (should not happen in real usage)
            print("Warning: Model is None, returning mock Q-values")
            q_values = {}
            for state_id in self.state_tensors.keys():
                q_values[state_id] = torch.randn(4)  # Mock 4 actions
            return q_values
        
        # Create a deep copy of the model to avoid modifying the original
        model_copy = copy.deepcopy(self.model)
        
        # Apply perturbations to the copy
        self.weight_selector.apply_perturbations(model_copy, weight_perturbations, mapping_info)
        
        # Compute Q-values for all states
        q_values = {}
        model_copy.eval()
        
        with torch.no_grad():
            for state_id, obs in self.state_tensors.items():
                if isinstance(obs, dict):
                    # Handle dict observations - create the proper observation format
                    # Based on agent config, the MLP keys are: 
                    # ["four_way_goal_direction", "four_way_angle_alignment", "barrier_mask", "lava_mask"]
                    # Each with sizes: 4x1, 4x1, 7x7, 7x7 respectively
                    
                    try:
                        # Get the flattened input data
                        if 'input' in obs:
                            input_data = obs['input']
                        else:
                            # Fallback - obs might already be the input data
                            input_data = obs
                        
                        # Convert to numpy array if needed
                        if isinstance(input_data, list):
                            input_data = np.array(input_data, dtype=np.float32)
                        elif isinstance(input_data, torch.Tensor):
                            input_data = input_data.cpu().numpy()
                        
                        # Reconstruct the original observation structure
                        # Based on config: four_way_goal_direction (4), four_way_angle_alignment (4), barrier_mask (49), lava_mask (49)
                        # Total: 4 + 4 + 49 + 49 = 106 elements
                        expected_size = 4 + 4 + 49 + 49  # 106
                        
                        if len(input_data) != expected_size:
                            print(f"Warning: Expected {expected_size} elements, got {len(input_data)} for state {state_id}")
                            # Try to handle different sizes gracefully
                            if len(input_data) < expected_size:
                                input_data = np.pad(input_data, (0, expected_size - len(input_data)), mode='constant')
                            else:
                                input_data = input_data[:expected_size]
                        
                        # Split the flattened data back into components
                        four_way_goal_direction = input_data[0:4]
                        four_way_angle_alignment = input_data[4:8]
                        barrier_mask = input_data[8:57].reshape(7, 7)  # 49 elements -> 7x7
                        lava_mask = input_data[57:106].reshape(7, 7)   # 49 elements -> 7x7
                        
                        # Create the structured observation dict
                        structured_obs = {
                            'four_way_goal_direction': four_way_goal_direction,
                            'four_way_angle_alignment': four_way_angle_alignment,
                            'barrier_mask': barrier_mask,
                            'lava_mask': lava_mask
                        }
                        
                        # Use model.predict for consistency with existing codebase
                        action, _ = model_copy.predict(structured_obs, deterministic=True)
                        
                        # Get Q-values directly using the q_net
                        # Convert structured observation to model format for direct Q-value computation
                        batch_obs = {'MLP_input': torch.tensor([input_data], dtype=torch.float32)}
                        
                        # Get Q-values using the q_net directly 
                        q_vals = model_copy.q_net(batch_obs)
                        q_values[state_id] = q_vals.squeeze(0)  # Remove batch dimension
                        
                        print(f"✓ Successfully computed Q-values for {state_id}: {q_vals.squeeze(0).numpy()}")
                        
                    except Exception as e:
                        print(f"Error processing observation for state {state_id}: {e}")
                        print(f"  Observation type: {type(obs)}")
                        print(f"  Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
                        # Fallback to dummy Q-values
                        q_values[state_id] = torch.zeros(4)  # Assuming 4 actions
                        
                else:
                    # Handle tensor observations (fallback)
                    try:
                        # Convert to numpy for predict
                        if isinstance(obs, torch.Tensor):
                            obs_np = obs.cpu().numpy()
                        else:
                            obs_np = np.array(obs)
                        
                        # Try to use as MLP input directly
                        batch_obs = {'MLP_input': torch.tensor([obs_np], dtype=torch.float32)}
                        q_vals = model_copy.q_net(batch_obs)
                        q_values[state_id] = q_vals.squeeze(0)  # Remove batch dimension
                        
                    except Exception as e:
                        print(f"Error processing tensor observation for state {state_id}: {e}")
                        # Fallback to dummy Q-values
                        q_values[state_id] = torch.zeros(4)  # Assuming 4 actions
        
        # Clean up the copy (Python garbage collection will handle this, but being explicit)
        del model_copy
        
        return q_values
    
    def _compute_margin_loss(self, q_values: Dict[str, torch.Tensor]) -> float:
        """
        Compute the margin-based loss for all ALTER states.
        
        Args:
            q_values: Dictionary of Q-values for each state
            
        Returns:
            Total margin loss across all states
        """
        total_loss = 0.0
        
        for state_id, q_vals in q_values.items():
            if state_id not in self.target_actions:
                continue
                
            acceptable_actions = self.target_actions[state_id]
            num_actions = len(q_vals)
            
            # Get Q-values for acceptable and non-acceptable actions
            acceptable_q = torch.max(q_vals[acceptable_actions])
            
            # Get Q-values for non-acceptable actions
            non_acceptable_mask = torch.ones(num_actions, dtype=torch.bool)
            non_acceptable_mask[acceptable_actions] = False
            
            if non_acceptable_mask.any():
                non_acceptable_q = torch.max(q_vals[non_acceptable_mask])
                
                # Margin loss: max(0, non_acceptable_max - acceptable_max + margin)
                loss = torch.clamp(non_acceptable_q - acceptable_q + self.margin, min=0.0)
                total_loss += loss.item()
        
        return total_loss
    
    def _compute_sparsity_penalty(self, weight_perturbations: np.ndarray) -> float:
        """
        Compute sparsity penalty (L1 norm approximation).
        
        Args:
            weight_perturbations: Vector of weight changes
            
        Returns:
            Sparsity penalty value
        """
        return np.sum(np.abs(weight_perturbations))
    
    def _compute_magnitude_penalty(self, weight_perturbations: np.ndarray) -> float:
        """
        Compute magnitude penalty (L2 norm).
        
        Args:
            weight_perturbations: Vector of weight changes
            
        Returns:
            Magnitude penalty value
        """
        return np.sum(weight_perturbations ** 2)
    
    def compute_objective(self, 
                         weight_perturbations: np.ndarray,
                         mapping_info: Dict,
                         **kwargs) -> float:
        """
        Compute the total objective function value.
        
        Args:
            weight_perturbations: Vector of weight changes (Δw)
            mapping_info: Mapping information for weight application
            **kwargs: Additional arguments
            
        Returns:
            Scalar objective function value
        """
        # Compute Q-values with perturbations
        q_values = self._compute_q_values(weight_perturbations, mapping_info)
        
        # Compute margin loss
        margin_loss = self._compute_margin_loss(q_values)
        
        # Compute regularization terms
        sparsity_penalty = self._compute_sparsity_penalty(weight_perturbations)
        magnitude_penalty = self._compute_magnitude_penalty(weight_perturbations)
        
        # Total objective
        total_objective = (margin_loss + 
                          self.lambda_sparse * sparsity_penalty + 
                          self.lambda_magnitude * magnitude_penalty)
        
        return total_objective
    
    def compute_gradient(self, 
                        weight_perturbations: np.ndarray,
                        mapping_info: Dict,
                        debug=False,
                        epsilon=1e-6,
                        **kwargs) -> np.ndarray:
        """
        Compute the gradient of the objective function using finite differences.
        
        Args:
            weight_perturbations: Vector of weight changes (Δw)
            mapping_info: Mapping information for weight application
            debug: Whether to print debugging information
            epsilon: Step size for finite differences
            **kwargs: Additional arguments
            
        Returns:
            Gradient vector with same shape as weight_perturbations
        """
        if debug:
            print(f"\n=== GRADIENT COMPUTATION DEBUG ===")
            print(f"Epsilon (step size): {epsilon}")
            print(f"Weight perturbations shape: {weight_perturbations.shape}")
            print(f"Weight perturbations norm: {np.linalg.norm(weight_perturbations):.8f}")
            
            # Check model requires_grad status
            if self.model is not None:
                grad_status = {}
                for name, param in self.model.named_parameters():
                    grad_status[name] = param.requires_grad
                
                total_params = len(grad_status)
                grad_enabled = sum(grad_status.values())
                print(f"Model requires_grad status: {grad_enabled}/{total_params} parameters have requires_grad=True")
                
                if grad_enabled < total_params:
                    print("WARNING: Some model parameters have requires_grad=False")
                    non_grad_params = [name for name, req_grad in grad_status.items() if not req_grad]
                    print(f"Non-gradient parameters: {non_grad_params[:5]}{'...' if len(non_grad_params) > 5 else ''}")
        
        gradient = np.zeros_like(weight_perturbations)
        
        # Compute gradient using finite differences
        base_objective = self.compute_objective(weight_perturbations, mapping_info)
        
        if debug:
            print(f"Base objective value: {base_objective:.8f}")
            
        # Track gradient computation statistics
        non_zero_gradients = 0
        max_gradient = 0.0
        
        for i in range(len(weight_perturbations)):
            # Forward difference
            perturb_plus = weight_perturbations.copy()
            perturb_plus[i] += epsilon
            obj_plus = self.compute_objective(perturb_plus, mapping_info)
            
            gradient[i] = (obj_plus - base_objective) / epsilon
            
            if abs(gradient[i]) > 1e-10:
                non_zero_gradients += 1
            max_gradient = max(max_gradient, abs(gradient[i]))
            
            if debug and i < 5:  # Show first 5 gradients
                print(f"  Gradient[{i}]: base={base_objective:.8f}, plus={obj_plus:.8f}, diff={obj_plus - base_objective:.8f}, grad={gradient[i]:.8f}")
        
        if debug:
            print(f"Gradient computation summary:")
            print(f"  Non-zero gradients: {non_zero_gradients}/{len(gradient)}")
            print(f"  Max gradient magnitude: {max_gradient:.8f}")
            print(f"  Gradient norm: {np.linalg.norm(gradient):.8f}")
            print(f"  Mean absolute gradient: {np.mean(np.abs(gradient)):.8f}")
            
            # Check for numerical issues
            if max_gradient == 0:
                print(f"  WARNING: All gradients are zero!")
            elif max_gradient < 1e-8:
                print(f"  WARNING: All gradients are very small (< 1e-8)")
            elif np.isnan(gradient).any():
                print(f"  ERROR: NaN gradients detected!")
            elif np.isinf(gradient).any():
                print(f"  ERROR: Infinite gradients detected!")
            
        return gradient 