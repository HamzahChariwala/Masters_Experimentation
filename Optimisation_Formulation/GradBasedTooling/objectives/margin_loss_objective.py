"""
Margin Loss Objective for Neural Network Weight Optimization.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Union

class MarginLossObjective:
    """Margin-based objective for neural network weight optimization with preserve constraints."""
    
    def __init__(self, model, weight_selector, alter_states: Dict, target_actions: Dict, 
                 preserve_states: Dict = None, margin: float = 0.1, config: Dict = None):
        """
        Initialize the margin loss objective.
        
        Args:
            model: The neural network model to optimize
            weight_selector: Weight selection utility
            alter_states: States to alter behavior for
            target_actions: Target actions for each state
            preserve_states: States to preserve behavior for (optional)
            margin: Margin threshold for loss computation
            config: Configuration dictionary with preserve constraint settings
        """
        self.model = model
        self.weight_selector = weight_selector
        self.alter_states = alter_states
        self.target_actions = target_actions
        self.preserve_states = preserve_states or {}
        self.margin = margin
        self.config = config or {}
        
        # Initialize preserve constraint system
        self.preserve_config = self.config.get('preserve_constraints', {})
        self.preserve_enabled = self.preserve_config.get('enabled', False)
        
        # Cache original preserve margins for relative penalty calculation
        self._original_preserve_margins = {}
        if self.preserve_enabled and self.preserve_states:
            self._compute_original_preserve_margins()
        
    def _compute_original_preserve_margins(self):
        """Compute and cache original margins for preserve states."""
        print(f"Computing original margins for {len(self.preserve_states)} preserve states...")
        
        for state_id, state_data in self.preserve_states.items():
            obs = state_data['input']
            with torch.no_grad():
                q_vals = self.model.q_net({'MLP_input': torch.tensor(obs, dtype=torch.float32).unsqueeze(0)})
                q_vals_np = q_vals.squeeze(0).cpu().numpy()
                
                # Find current top action (what we want to preserve)
                top_action = np.argmax(q_vals_np)
                max_target_q = q_vals_np[top_action]
                
                # Find second highest Q-value
                non_target_q_values = np.delete(q_vals_np, top_action)
                max_non_target_q = np.max(non_target_q_values) if len(non_target_q_values) > 0 else 0.0
                
                # Store original margin (positive = target winning)
                original_margin = max_target_q - max_non_target_q
                self._original_preserve_margins[state_id] = {
                    'margin': original_margin,
                    'top_action': top_action,
                    'max_target_q': max_target_q,
                    'max_non_target_q': max_non_target_q
                }
        
        print(f"Original preserve margins computed: {list(self._original_preserve_margins.keys())}")
        
    def _compute_preserve_penalty(self, preserve_q_values: Dict) -> float:
        """
        Compute preserve constraint penalty using the configured penalty type.
        
        Args:
            preserve_q_values: Dictionary of current Q-values for preserve states
            
        Returns:
            Total penalty value
        """
        if not self.preserve_enabled or not self.preserve_states:
            return 0.0
            
        penalty_type = self.preserve_config.get('penalty_type', 'relative_margin')
        penalty_weight = self.preserve_config.get('penalty_weight', 1.0)
        
        if penalty_type == 'relative_margin':
            return self._compute_relative_margin_penalty(preserve_q_values, penalty_weight)
        else:
            # Future penalty types can be added here
            print(f"Warning: Unknown penalty type '{penalty_type}', using relative_margin")
            return self._compute_relative_margin_penalty(preserve_q_values, penalty_weight)
    
    def _compute_relative_margin_penalty(self, preserve_q_values: Dict, penalty_weight: float) -> float:
        """
        Compute relative margin penalty for preserve states.
        
        Penalty = λ * Σ max(0, (original_margin * threshold) - current_margin)
        """
        margin_threshold = self.preserve_config.get('margin_threshold', 0.9)
        total_penalty = 0.0
        
        for state_id, q_vals in preserve_q_values.items():
            if state_id not in self._original_preserve_margins:
                continue
                
            original_info = self._original_preserve_margins[state_id]
            original_margin = original_info['margin']
            original_top_action = original_info['top_action']
            
            # Convert to numpy for easier manipulation
            if hasattr(q_vals, 'cpu'):
                q_vals_np = q_vals.cpu().numpy()
            else:
                q_vals_np = q_vals.numpy()
            
            # Calculate current margin for the originally top action
            if original_top_action < len(q_vals_np):
                current_target_q = q_vals_np[original_top_action]
                non_target_indices = [i for i in range(len(q_vals_np)) if i != original_top_action]
                
                if non_target_indices:
                    current_max_non_target_q = max(q_vals_np[i] for i in non_target_indices)
                    current_margin = current_target_q - current_max_non_target_q
                    
                    # Compute penalty: penalize when current margin falls below threshold * original
                    required_margin = original_margin * margin_threshold
                    penalty = max(0.0, required_margin - current_margin)
                    total_penalty += penalty
        
        return penalty_weight * total_penalty
        
    def compute_objective(self, perturbations: np.ndarray, mapping_info: Dict) -> float:
        """
        Compute the objective value for given weight perturbations.
        
        Args:
            perturbations: Weight perturbations to apply
            mapping_info: Mapping information for weight application
            
        Returns:
            Objective value (margin loss + regularization)
        """
        # Store original model state
        original_state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_state_dict[name] = param.data.clone()
        
        try:
            # Apply perturbations to model
            self.weight_selector.apply_perturbations(self.model, perturbations, mapping_info)
            
            # Compute Q-values and margin loss for ALTER states
            alter_q_values = self._compute_q_values(self.alter_states)
            margin_loss = self._compute_margin_loss(alter_q_values)
            
            # Compute preserve constraint penalty if enabled
            preserve_penalty = 0.0
            if self.preserve_enabled and self.preserve_states:
                preserve_q_values = self._compute_q_values(self.preserve_states)
                preserve_penalty = self._compute_preserve_penalty(preserve_q_values)
            
            # Compute regularization terms
            l1_penalty = self._compute_l1_regularization(perturbations)
            l2_penalty = self._compute_l2_regularization(perturbations)
            
            # Combined objective: margin loss + preserve penalty + L1 + L2
            total_objective = margin_loss + preserve_penalty + l1_penalty + l2_penalty
            
            # Restore original model state
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in original_state_dict:
                        param.data = original_state_dict[name]
            
            return float(total_objective)
            
        except Exception as e:
            # Restore state even on error
            try:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in original_state_dict:
                            param.data = original_state_dict[name]
            except:
                pass
            print(f"Error in objective computation: {e}")
            return float('inf')
            
    def _compute_l1_regularization(self, perturbations: np.ndarray) -> float:
        """
        Compute L1 regularization penalty.
        
        Args:
            perturbations: Weight perturbations
            
        Returns:
            L1 penalty: λ_sparse * ||perturbations||_1
        """
        lambda_sparse = self.config.get('lambda_sparse', 0.0)
        if lambda_sparse == 0.0:
            return 0.0
        
        l1_norm = np.sum(np.abs(perturbations))
        return lambda_sparse * l1_norm
    
    def _compute_l2_regularization(self, perturbations: np.ndarray) -> float:
        """
        Compute L2 regularization penalty.
        
        Args:
            perturbations: Weight perturbations
            
        Returns:
            L2 penalty: λ_magnitude * ||perturbations||_2^2
        """
        lambda_magnitude = self.config.get('lambda_magnitude', 0.0)
        if lambda_magnitude == 0.0:
            return 0.0
        
        l2_norm_squared = np.sum(perturbations ** 2)
        return lambda_magnitude * l2_norm_squared
            
    def _compute_q_values(self, states: Dict) -> Dict:
        """Compute Q-values for all states."""
        q_values = {}
        
        for state_id, state_data in states.items():
            obs = state_data['input']
            obs_dict = {'MLP_input': np.array(obs, dtype=np.float32)}
            
            # Get Q-values from model
            with torch.no_grad():
                q_vals = self.model.q_net({'MLP_input': torch.tensor(obs, dtype=torch.float32).unsqueeze(0)})
                q_values[state_id] = q_vals.squeeze(0)
                
        return q_values
        
    def _compute_margin_loss(self, q_values: Dict) -> float:
        """Compute margin loss from Q-values."""
        total_loss = 0.0
        
        for state_id, q_vals in q_values.items():
            target_action_list = self.target_actions.get(state_id, [])
            
            if not target_action_list:
                continue
                
            # Convert to numpy for easier manipulation
            if hasattr(q_vals, 'cpu'):
                q_vals_np = q_vals.cpu().numpy()
            else:
                q_vals_np = q_vals.numpy()
                
            # Find maximum Q-value among target actions
            target_q_values = [q_vals_np[action] for action in target_action_list if action < len(q_vals_np)]
            if not target_q_values:
                continue
                
            max_target_q = max(target_q_values)
            
            # Find maximum Q-value among non-target actions
            non_target_actions = [i for i in range(len(q_vals_np)) if i not in target_action_list]
            if not non_target_actions:
                continue
                
            max_non_target_q = max(q_vals_np[i] for i in non_target_actions)
            
            # Compute margin loss: we want max_target_q > max_non_target_q + margin
            loss = max(0, max_non_target_q + self.margin - max_target_q)
            total_loss += loss
            
        return total_loss 