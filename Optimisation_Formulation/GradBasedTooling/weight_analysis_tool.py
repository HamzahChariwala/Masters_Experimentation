#!/usr/bin/env python3
"""
Weight Analysis Tool for Identifying Large Weight Changes
Analyzes optimization results to identify specific weights and neurons with large perturbations.
"""
import sys
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add the correct paths for imports
sys.path.insert(0, '.')
sys.path.insert(0, 'Optimisation_Formulation/GradBasedTooling')

from Optimisation_Formulation.GradBasedTooling.utils.integration_utils import load_model_from_agent_path
from Optimisation_Formulation.GradBasedTooling.utils.weight_selection import WeightSelector


class WeightChangeAnalyzer:
    """Analyzes weight changes and maps them to specific neurons and layers"""
    
    def __init__(self, agent_path: str):
        self.agent_path = agent_path
        self.model = load_model_from_agent_path(agent_path)
        self.weight_selector = WeightSelector()
        
    def analyze_perturbations(self, 
                            perturbations: np.ndarray, 
                            neuron_specs: List[Tuple[str, int]], 
                            mapping_info: Dict,
                            threshold: float = 100.0) -> Dict:
        """
        Analyze perturbations to identify large weight changes and their corresponding neurons.
        
        Args:
            perturbations: Array of weight perturbations
            neuron_specs: List of (layer_name, neuron_index) tuples
            mapping_info: Mapping information from weight selector
            threshold: Threshold for considering a weight change "large"
            
        Returns:
            Dictionary with analysis results
        """
        
        # Find large perturbations
        large_changes = np.abs(perturbations) > threshold
        large_indices = np.where(large_changes)[0]
        large_values = perturbations[large_changes]
        
        print(f"\nWEIGHT CHANGE ANALYSIS")
        print(f"{'='*50}")
        print(f"Total weights analyzed: {len(perturbations)}")
        print(f"Large changes (>{threshold}): {len(large_indices)}")
        print(f"Largest absolute change: {np.max(np.abs(perturbations)):.6f}")
        print(f"Weight change statistics:")
        print(f"  Mean: {np.mean(perturbations):.6f}")
        print(f"  Std:  {np.std(perturbations):.6f}")
        print(f"  Min:  {np.min(perturbations):.6f}")
        print(f"  Max:  {np.max(perturbations):.6f}")
        
        # Map large changes to neurons
        neuron_mappings = []
        for idx in large_indices:
            layer_info = self._find_layer_for_weight_index(idx, mapping_info)
            if layer_info:
                neuron_info = self._find_neuron_for_weight_index(idx, layer_info, mapping_info)
                if neuron_info:
                    neuron_mappings.append({
                        'weight_index': int(idx),
                        'perturbation': float(perturbations[idx]),
                        'layer_name': layer_info['layer_name'],
                        'neuron_index': neuron_info['neuron_index'],
                        'weight_within_neuron': neuron_info['weight_within_neuron'],
                        'global_layer_weight_index': neuron_info['global_layer_weight_index']
                    })
        
        # Sort by absolute perturbation value
        neuron_mappings.sort(key=lambda x: abs(x['perturbation']), reverse=True)
        
        print(f"\nLARGE WEIGHT CHANGES (>{threshold}):")
        print(f"{'Rank':<4} {'Weight Idx':<10} {'Change':<12} {'Layer':<25} {'Neuron':<8} {'Local Wt':<8}")
        print(f"{'-'*75}")
        
        for i, mapping in enumerate(neuron_mappings[:20]):  # Show top 20
            print(f"{i+1:<4} {mapping['weight_index']:<10} {mapping['perturbation']:<12.6f} "
                  f"{mapping['layer_name']:<25} {mapping['neuron_index']:<8} {mapping['weight_within_neuron']:<8}")
        
        return {
            'total_weights': len(perturbations),
            'large_changes_count': len(large_indices),
            'threshold': threshold,
            'statistics': {
                'mean': float(np.mean(perturbations)),
                'std': float(np.std(perturbations)),
                'min': float(np.min(perturbations)),
                'max': float(np.max(perturbations)),
                'largest_abs': float(np.max(np.abs(perturbations)))
            },
            'large_changes': neuron_mappings
        }
    
    def _find_layer_for_weight_index(self, weight_index: int, mapping_info: Dict) -> Optional[Dict]:
        """Find which layer a weight index belongs to"""
        for layer_name, (start_idx, end_idx) in mapping_info['layer_slices'].items():
            if start_idx <= weight_index < end_idx:
                return {
                    'layer_name': layer_name,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'layer_weight_index': weight_index - start_idx
                }
        return None
    
    def _find_neuron_for_weight_index(self, weight_index: int, layer_info: Dict, mapping_info: Dict) -> Optional[Dict]:
        """Find which neuron within a layer a weight index belongs to"""
        layer_name = layer_info['layer_name']
        layer_weight_index = layer_info['layer_weight_index']
        
        # Get neuron specs for this layer
        layer_neurons = [(ln, ni) for ln, ni in mapping_info['neuron_specs'] if ln == layer_name]
        
        # Get layer module to understand weight structure
        layer_module = dict(self.model.named_modules())[layer_name]
        if not hasattr(layer_module, 'weight'):
            return None
            
        weight_shape = layer_module.weight.shape
        
        # For linear layers: weight[neuron_idx, :] are the incoming weights
        if len(weight_shape) == 2:
            weights_per_neuron = weight_shape[1]
        else:
            weights_per_neuron = weight_shape[1] if len(weight_shape) > 1 else 1
            
        # Find which neuron this weight belongs to
        for layer_neuron_name, neuron_idx in layer_neurons:
            if layer_neuron_name == layer_name:
                # Calculate weight indices for this neuron
                neuron_start = neuron_idx * weights_per_neuron
                neuron_end = neuron_start + weights_per_neuron
                
                # Convert to layer-relative indices
                weight_indices = self.weight_selector.neurons_to_weight_indices(
                    self.model, [(layer_name, neuron_idx)]
                )[layer_name]
                
                # Check if our weight_index falls within this neuron's weights
                global_layer_weight_index = layer_weight_index + layer_info['start_idx']
                relative_weight_index = global_layer_weight_index - layer_info['start_idx']
                
                if relative_weight_index in [idx - min(weight_indices) for idx in weight_indices]:
                    weight_within_neuron = relative_weight_index - (neuron_idx * weights_per_neuron - min(weight_indices))
                    return {
                        'neuron_index': neuron_idx,
                        'weight_within_neuron': weight_within_neuron,
                        'global_layer_weight_index': relative_weight_index
                    }
        
        return None
    
    def analyze_preserve_constraint_impact(self, 
                                         perturbations: np.ndarray,
                                         neuron_specs: List[Tuple[str, int]],
                                         mapping_info: Dict,
                                         preserve_states: Dict,
                                         threshold: float = 100.0) -> Dict:
        """
        Analyze why large weight changes don't violate preserve constraints.
        
        Args:
            perturbations: Array of weight perturbations
            neuron_specs: List of (layer_name, neuron_index) tuples
            mapping_info: Mapping information from weight selector
            preserve_states: Dictionary of preserve states
            threshold: Threshold for considering a weight change "large"
            
        Returns:
            Analysis of constraint preservation despite large weight changes
        """
        
        print(f"\nPRESERVE CONSTRAINT IMPACT ANALYSIS")
        print(f"{'='*50}")
        
        # Apply perturbations to model
        original_model = load_model_from_agent_path(self.agent_path)
        perturbed_model = load_model_from_agent_path(self.agent_path)
        self.weight_selector.apply_perturbations(perturbed_model, perturbations, mapping_info)
        
        # Find large changes
        large_changes = np.abs(perturbations) > threshold
        large_indices = np.where(large_changes)[0]
        
        print(f"Analyzing {len(preserve_states)} preserve states...")
        print(f"Large weight changes: {len(large_indices)} (threshold: {threshold})")
        
        constraint_analysis = {}
        
        for state_id, state_data in preserve_states.items():
            obs = state_data['input']
            obs_dict = {'MLP_input': torch.tensor(obs, dtype=torch.float32).unsqueeze(0)}
            
            # Get original action
            with torch.no_grad():
                original_action, _ = original_model.predict({'MLP_input': obs}, deterministic=True)
                original_q_vals = original_model.q_net(obs_dict).squeeze(0).cpu().numpy()
                
                # Get perturbed action
                perturbed_action, _ = perturbed_model.predict({'MLP_input': obs}, deterministic=True)
                perturbed_q_vals = perturbed_model.q_net(obs_dict).squeeze(0).cpu().numpy()
            
            action_changed = original_action != perturbed_action
            q_value_change = perturbed_q_vals - original_q_vals
            max_q_change = np.max(np.abs(q_value_change))
            
            constraint_analysis[state_id] = {
                'original_action': int(original_action),
                'perturbed_action': int(perturbed_action),
                'action_changed': bool(action_changed),
                'original_q_values': original_q_vals.tolist(),
                'perturbed_q_values': perturbed_q_vals.tolist(),
                'q_value_changes': q_value_change.tolist(),
                'max_q_change': float(max_q_change)
            }
            
            if action_changed:
                print(f"CONSTRAINT VIOLATED - State {state_id}: {original_action} → {perturbed_action}")
            else:
                print(f"CONSTRAINT PRESERVED - State {state_id}: {original_action} → {perturbed_action} (max Q-change: {max_q_change:.6f})")
        
        violated_count = sum(1 for analysis in constraint_analysis.values() if analysis['action_changed'])
        violation_rate = violated_count / len(preserve_states) if preserve_states else 0
        
        print(f"\nCONSTRAINT SUMMARY:")
        print(f"Total preserve states: {len(preserve_states)}")
        print(f"Violated constraints: {violated_count}")
        print(f"Violation rate: {violation_rate:.2%}")
        
        return {
            'total_states': len(preserve_states),
            'violated_count': violated_count,
            'violation_rate': violation_rate,
            'state_analysis': constraint_analysis
        }


def analyze_optimization_result(agent_path: str, 
                              perturbations: np.ndarray, 
                              neuron_specs: List[Tuple[str, int]], 
                              mapping_info: Dict,
                              preserve_states: Dict = None,
                              threshold: float = 100.0) -> Dict:
    """
    Comprehensive analysis of optimization weight changes.
    
    Args:
        agent_path: Path to agent directory
        perturbations: Array of weight perturbations from optimization
        neuron_specs: List of (layer_name, neuron_index) tuples
        mapping_info: Mapping information from weight selector
        preserve_states: Dictionary of preserve states (optional)
        threshold: Threshold for considering a weight change "large"
        
    Returns:
        Complete analysis results
    """
    
    analyzer = WeightChangeAnalyzer(agent_path)
    
    # Analyze weight changes
    weight_analysis = analyzer.analyze_perturbations(perturbations, neuron_specs, mapping_info, threshold)
    
    # Analyze preserve constraint impact if preserve states provided
    constraint_analysis = None
    if preserve_states:
        constraint_analysis = analyzer.analyze_preserve_constraint_impact(
            perturbations, neuron_specs, mapping_info, preserve_states, threshold
        )
    
    return {
        'weight_analysis': weight_analysis,
        'constraint_analysis': constraint_analysis
    }


if __name__ == "__main__":
    # Example usage - you would typically call this from your optimization script
    print("Weight Analysis Tool")
    print("To use this tool, call analyze_optimization_result() with your optimization results") 