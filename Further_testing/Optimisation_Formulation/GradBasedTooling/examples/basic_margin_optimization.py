#!/usr/bin/env python3
"""
Basic margin-based optimization example.

Demonstrates the prototype gradient-based optimization system with:
- Random neuron selection from first 3 layers
- Small sample of ALTER and PRESERVE states  
- Margin-based objective with configurable parameters
"""

import sys
import os
import argparse
from typing import Dict, List, Optional
import numpy as np
import random
import json
from datetime import datetime
import torch
import torch.nn.functional as F

# Add project directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
grad_tooling_dir = os.path.dirname(script_dir)
sys.path.insert(0, grad_tooling_dir)

from utils.weight_selection import WeightSelector
from utils.integration_utils import (
    load_states_from_tooling, 
    load_target_actions_from_tooling,
    validate_neuron_indices,
    load_model_from_agent_path
)
from objectives.margin_loss_objective import MarginLossObjective


def compute_q_values_with_debug(model, obs, state_id):
    """
    Compute Q-values with detailed debugging.
    
    Returns the Q-values as a numpy array.
    """
    print(f"\n--- Q-VALUE COMPUTATION DEBUG for {state_id} ---")
    print(f"Original observation type: {type(obs)}")
    print(f"Original observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
    
    if isinstance(obs, dict) and 'input' in obs:
        input_data = obs['input']
        print(f"Input data type: {type(input_data)}")
        print(f"Input data length: {len(input_data) if hasattr(input_data, '__len__') else 'No length'}")
        print(f"Input data first 5 elements: {input_data[:5] if hasattr(input_data, '__getitem__') else 'No indexing'}")
        
        # Create observation dict that matches the model's expected format
        # Based on CustomCombinedExtractor, the model expects 'MLP_input' for vector data
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        model_obs = {'MLP_input': input_tensor}
        
        print(f"Created model observation: {list(model_obs.keys())}")
        print(f"MLP_input shape: {model_obs['MLP_input'].shape}")
        
        print(f"\n=== ATTEMPTING Q-VALUE COMPUTATION ===")
        approaches = []
        
        # Try q_net directly with observation dict (this gives us the Q-values)
        if hasattr(model, 'q_net'):
            try:
                with torch.no_grad():
                    q_values = model.q_net(model_obs)
                    approaches.append(("q_net(obs_dict)", q_values))
                    print(f"✓ q_net(obs_dict) worked: {q_values.shape}")
            except Exception as e:
                print(f"✗ q_net(obs_dict) failed: {e}")
        
        # Try direct model call with observation dict (this gives us the action, not Q-values)
        try:
            with torch.no_grad():
                model.eval()
                action_result = model(model_obs)
                print(f"✓ Direct model(obs_dict) returned action: {action_result.shape} (this is action, not Q-values)")
        except Exception as e:
            print(f"✗ Direct model(obs_dict) failed: {e}")
        
        # Try using model.predict with numpy array
        try:
            np_obs = {'MLP_input': np.array(input_data, dtype=np.float32)}
            action, _states = model.predict(np_obs, deterministic=True)
            print(f"✓ model.predict returned action: {action} (this is action, not Q-values)")
            # Get Q-values separately using the model's Q-network
            with torch.no_grad():
                q_values = model.q_net(model_obs)
                approaches.append(("predict+q_net", q_values))
                print(f"✓ predict+q_net Q-values: {q_values.shape}")
        except Exception as e:
            print(f"✗ model.predict approach failed: {e}")
        
        # Return the first successful Q-value approach (should be q_net)
        if approaches:
            print(f"Using successful approach: {approaches[0][0]}")
            q_values = approaches[0][1]
            if hasattr(q_values, 'squeeze'):
                q_array = q_values.squeeze(0).cpu().numpy()
                print(f"Final Q-values vector: {q_array}")
                return q_array
            else:
                q_array = np.array(q_values)
                print(f"Final Q-values vector: {q_array}")
                return q_array
        else:
            print("✗ All approaches failed!")
            return None
    
    print("Invalid observation format")
    return None


def verify_weight_perturbations(model, perturbations, mapping_info, weight_selector):
    """
    Verify that weight perturbations are correctly applied.
    
    Returns dict with verification results.
    """
    # Get original weights
    original_weights = {}
    for layer_name in mapping_info['layer_slices']:
        layer_module = dict(model.named_modules())[layer_name]
        original_weights[layer_name] = layer_module.weight.data.clone()
    
    # Apply perturbations
    weight_selector.apply_perturbations(model, perturbations, mapping_info)
    
    # Check changes
    verification_results = {
        'weights_changed': {},
        'perturbation_magnitudes': {},
        'max_changes': {}
    }
    
    for layer_name, (start_idx, end_idx) in mapping_info['layer_slices'].items():
        layer_module = dict(model.named_modules())[layer_name]
        new_weights = layer_module.weight.data
        
        # Calculate differences
        weight_diff = (new_weights - original_weights[layer_name]).abs()
        max_change = float(weight_diff.max().item())  # Convert to Python float
        mean_change = float(weight_diff.mean().item())  # Convert to Python float
        perturbation_norm = float(np.linalg.norm(perturbations[start_idx:end_idx]))  # Convert to Python float
        
        verification_results['weights_changed'][layer_name] = max_change > 1e-8
        verification_results['perturbation_magnitudes'][layer_name] = {
            'max_change': max_change,
            'mean_change': mean_change,
            'perturbation_norm': perturbation_norm
        }
        verification_results['max_changes'][layer_name] = max_change
    
    return verification_results


def verify_q_value_changes(model, alter_states, perturbations, mapping_info, weight_selector):
    """
    Verify that weight perturbations actually change Q-values.
    
    Returns dict with Q-value verification results.
    """
    verification_results = {
        'q_value_changes': {},
        'max_q_change': 0.0,
        'states_affected': 0,
        'debug_info': {}
    }
    
    # Get original Q-values
    original_q_values = {}
    model.eval()
    
    print("=== Q-VALUE COMPUTATION DEBUG ===")
    
    # Test first 3 states for debugging
    test_states = list(alter_states.items())[:3]
    
    for state_id, state_data in test_states:
        print(f"\nProcessing state: {state_id}")
        
        obs = state_data
        print(f"State data type: {type(obs)}")
        print(f"State data keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
        
        # Get original Q-values with debugging
        original_q = compute_q_values_with_debug(model, obs, state_id)
        if original_q is not None:
            original_q_values[state_id] = original_q
            print(f"Successfully computed original Q-values: {original_q}")
        else:
            print(f"Failed to compute Q-values for state {state_id}")
            continue
    
    # Apply perturbations
    print(f"\n=== APPLYING PERTURBATIONS ===")
    weight_selector.apply_perturbations(model, perturbations, mapping_info)
    
    # Get perturbed Q-values
    perturbed_q_values = {}
    for state_id, state_data in test_states:
        if state_id in original_q_values:
            print(f"\nComputing perturbed Q-values for state: {state_id}")
            
            perturbed_q = compute_q_values_with_debug(model, state_data, f"{state_id}_perturbed")
            if perturbed_q is not None:
                perturbed_q_values[state_id] = perturbed_q
                
                # Compare Q-values
                q_diff = np.abs(perturbed_q - original_q_values[state_id])
                max_diff = np.max(q_diff)
                verification_results['q_value_changes'][state_id] = {
                    'original_q': original_q_values[state_id].tolist(),
                    'perturbed_q': perturbed_q.tolist(),
                    'max_difference': float(max_diff),
                    'mean_difference': float(np.mean(q_diff))
                }
                
                if max_diff > 1e-6:  # Consider any change > 1e-6 as significant
                    verification_results['states_affected'] += 1
                
                verification_results['max_q_change'] = max(verification_results['max_q_change'], max_diff)
                
                print(f"Q-value changes for {state_id}:")
                print(f"  Original: {original_q_values[state_id]}")
                print(f"  Perturbed: {perturbed_q}")
                print(f"  Max difference: {max_diff}")
    
    # Restore original weights
    weight_selector.apply_perturbations(model, -perturbations, mapping_info)
    
    verification_results['debug_info'] = {
        'total_states_tested': len(test_states),
        'states_with_original_q': len(original_q_values),
        'states_with_perturbed_q': len(perturbed_q_values)
    }
    
    return verification_results


def run_optimization_steps(objective, mapping_info, initial_vector, num_steps=10, step_size=0.01):
    """
    Run simple gradient descent optimization steps.
    
    Returns optimization results with step-by-step tracking.
    """
    optimization_results = {
        'steps': [],
        'final_objective': None,
        'total_improvement': None,
        'converged': False
    }
    
    current_vector = initial_vector.copy()
    
    for step in range(num_steps):
        # Compute objective and gradient
        obj_value = objective.compute_objective(current_vector, mapping_info)
        gradient = objective.compute_gradient(current_vector, mapping_info)
        gradient_norm = np.linalg.norm(gradient)
        
        # Store step results
        step_results = {
            'step': step,
            'objective': float(obj_value),
            'gradient_norm': float(gradient_norm),
            'vector_norm': float(np.linalg.norm(current_vector))
        }
        optimization_results['steps'].append(step_results)
        
        print(f"Step {step}: obj={obj_value:.6f}, grad_norm={gradient_norm:.6f}")
        
        # Check convergence
        if gradient_norm < 1e-6:
            optimization_results['converged'] = True
            print(f"Converged at step {step}")
            break
        
        # Update weights (gradient descent)
        current_vector -= step_size * gradient
    
    optimization_results['final_objective'] = float(obj_value)
    optimization_results['total_improvement'] = float(optimization_results['steps'][0]['objective'] - obj_value)
    
    return optimization_results, current_vector


def run_basic_optimization(agent_path: str,
                         num_neurons: int = 2,
                         alter_samples: int = 3,
                         preserve_samples: int = 5,
                         margin: float = 0.1,
                         lambda_sparse: float = 0.0001,
                         lambda_magnitude: float = 0.001,
                         seed: Optional[int] = None) -> Dict:
    """
    Run basic margin-based optimization prototype.
    
    Args:
        agent_path: Path to agent directory with StateTooling results
        num_neurons: Number of neurons to randomly select for optimization
        alter_samples: Number of ALTER states to use
        preserve_samples: Number of PRESERVE states to use  
        margin: Margin value for margin-based loss
        lambda_sparse: Sparsity regularization coefficient
        lambda_magnitude: Magnitude regularization coefficient
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with optimization results
    """
    print(f"Starting basic margin optimization for: {agent_path}")
    print(f"Parameters: neurons={num_neurons}, alter={alter_samples}, preserve={preserve_samples}")
    print(f"Margin={margin}, λ_sparse={lambda_sparse}, λ_magnitude={lambda_magnitude}")
    
    # Step 1: Load model
    print("\n1. Loading model...")
    model = load_model_from_agent_path(agent_path)
    if model is None:
        print("ERROR: Could not load model. You need to implement load_model_from_agent_path()")
        return {'success': False, 'error': 'Model loading not implemented'}
    
    # Step 2: Load state data from StateTooling
    print("\n2. Loading state data...")
    alter_states, preserve_states = load_states_from_tooling(
        agent_path, alter_samples, preserve_samples
    )
    
    print(f"Loaded {len(alter_states)} ALTER states, {len(preserve_states)} PRESERVE states")
    
    if not alter_states:
        print("ERROR: No ALTER states found. Run StateTooling first.")
        return {'success': False, 'error': 'No ALTER states'}
    
    # Step 3: Load target actions
    print("\n3. Loading target actions...")
    target_actions = load_target_actions_from_tooling(agent_path, list(alter_states.keys()))
    print(f"Target actions: {target_actions}")
    
    # Step 4: Set up weight selector
    print("\n4. Setting up weight selection...")
    weight_selector = WeightSelector(
        target_layers=['q_net.features_extractor.mlp.0', 'q_net.q_net.0', 'q_net.q_net.2'],
        weights_per_neuron=64
    )
    
    # Analyze model structure
    layer_info = weight_selector.analyze_model_structure(model)
    print(f"Available layers: {list(layer_info.keys())}")
    
    # Step 5: Randomly select neurons
    print(f"\n5. Selecting {num_neurons} random neurons...")
    try:
        selected_neurons = weight_selector.select_random_neurons(model, num_neurons, seed)
        print(f"Selected neurons: {selected_neurons}")
        
        # Validate selections
        if not validate_neuron_indices(model, selected_neurons, weight_selector.target_layers):
            print("ERROR: Invalid neuron selections")
            return {'success': False, 'error': 'Invalid neuron selections'}
            
    except Exception as e:
        print(f"ERROR: Could not select neurons: {e}")
        return {'success': False, 'error': f'Neuron selection failed: {e}'}
    
    # Step 6: Create optimization vector
    print("\n6. Creating optimization variables...")
    try:
        initial_weights, mapping_info = weight_selector.create_optimization_vector(model, selected_neurons)
        print(f"Optimization vector size: {len(initial_weights)}")
        print(f"Layer mapping: {list(mapping_info['layer_slices'].keys())}")
        
    except Exception as e:
        print(f"ERROR: Could not create optimization vector: {e}")
        return {'success': False, 'error': f'Vector creation failed: {e}'}
    
    # Step 7: Set up objective function
    print("\n7. Setting up margin-based objective...")
    try:
        objective = MarginLossObjective(
            model=model,
            weight_selector=weight_selector,
            alter_states=alter_states,
            target_actions=target_actions,
            margin=margin,
            lambda_sparse=lambda_sparse,
            lambda_magnitude=lambda_magnitude
        )
        
        # Test objective computation
        initial_obj = objective.compute_objective(
            weight_perturbations=initial_weights * 0,  # Zero perturbations initially
            mapping_info=mapping_info
        )
        print(f"Initial objective value: {initial_obj}")
        
    except Exception as e:
        print(f"ERROR: Could not set up objective: {e}")
        return {'success': False, 'error': f'Objective setup failed: {e}'}
    
    # Step 8: Test gradient computation
    print("\n8. Testing gradient computation...")
    try:
        zero_perturbations = initial_weights * 0
        gradient = objective.compute_gradient(zero_perturbations, mapping_info)
        print(f"Gradient computed successfully, norm: {(gradient**2).sum()**0.5:.6f}")
        
    except Exception as e:
        print(f"ERROR: Gradient computation failed: {e}")
        return {'success': False, 'error': f'Gradient computation failed: {e}'}
    
    print("\n" + "="*60)
    print("PROTOTYPE SETUP SUCCESSFUL")
    print("="*60)
    print("The basic optimization infrastructure is working.")
    print("Next steps:")
    print("1. Implement actual solver (SLSQP) integration")
    print("2. Add preserve behavior constraints") 
    print("3. Run full optimization and validate results")
    print("4. Implement load_model_from_agent_path() for your model format")
    
    return {
        'success': True,
        'num_variables': len(initial_weights),
        'selected_neurons': selected_neurons,
        'initial_objective': initial_obj,
        'gradient_norm': (gradient**2).sum()**0.5,
        'alter_states_count': len(alter_states),
        'preserve_states_count': len(preserve_states)
    }


def test_multiple_neuron_selections(model, weight_selector, objective, mapping_base, num_tests=5, neurons_per_test=10, seed=42):
    """
    Test multiple different neuron selections to find ones that produce non-zero gradients.
    
    Args:
        model: The neural network model
        weight_selector: WeightSelector instance
        objective: Objective function instance
        mapping_base: Base mapping info for reference
        num_tests: Number of different neuron selections to try
        neurons_per_test: Number of neurons per test
        seed: Base random seed
        
    Returns:
        List of test results with neuron selections and gradient information
    """
    print(f"\n=== TESTING {num_tests} DIFFERENT NEURON SELECTIONS ===")
    
    # Calculate total network size for coverage tracking
    layer_info = weight_selector.analyze_model_structure(model)
    total_neurons = sum(info['num_neurons'] for info in layer_info.values())
    target_coverage = int(0.75 * total_neurons)
    
    print(f"Total neurons in target layers: {total_neurons}")
    print(f"Target coverage (75%): {target_coverage}")
    
    test_results = []
    neurons_covered = set()
    
    for test_idx in range(num_tests):
        test_seed = seed + test_idx * 100  # Ensure different selections
        print(f"\n--- Test {test_idx + 1}/{num_tests} (seed={test_seed}) ---")
        
        try:
            # Generate new neuron selection
            selected_neurons = weight_selector.select_random_neurons(
                model, neurons_per_test, seed=test_seed
            )
            print(f"Selected neurons: {selected_neurons}")
            
            # Track coverage
            for layer_name, neuron_idx in selected_neurons:
                neurons_covered.add((layer_name, neuron_idx))
            
            coverage_pct = (len(neurons_covered) / total_neurons) * 100
            print(f"Cumulative coverage: {len(neurons_covered)}/{total_neurons} ({coverage_pct:.1f}%)")
            
            # Create optimization vector for this selection
            optimization_vector, mapping_info = weight_selector.create_optimization_vector(
                model, selected_neurons
            )
            
            # Test gradient computation with zero perturbations
            zero_perturbations = optimization_vector * 0
            gradient = objective.compute_gradient(zero_perturbations, mapping_info)
            gradient_norm = np.linalg.norm(gradient)
            
            # Test objective value
            obj_value = objective.compute_objective(zero_perturbations, mapping_info)
            
            test_result = {
                'test_index': test_idx,
                'seed': test_seed,
                'selected_neurons': selected_neurons,
                'num_parameters': len(optimization_vector),
                'gradient_norm': float(gradient_norm),
                'objective_value': float(obj_value),
                'non_zero_gradients': int(np.sum(np.abs(gradient) > 1e-8)),
                'coverage_neurons': len(neurons_covered),
                'coverage_percentage': float(coverage_pct)
            }
            
            test_results.append(test_result)
            
            print(f"Parameters: {len(optimization_vector)}")
            print(f"Gradient norm: {gradient_norm:.8f}")
            print(f"Non-zero gradients: {test_result['non_zero_gradients']}/{len(gradient)}")
            print(f"Objective value: {obj_value:.6f}")
            
            # If we have non-zero gradients, this is promising
            if gradient_norm > 1e-6:
                print(f"✓ FOUND NON-ZERO GRADIENTS! This selection looks promising.")
            
            # Check if we've reached target coverage
            if len(neurons_covered) >= target_coverage:
                print(f"✓ Reached target coverage of 75% ({coverage_pct:.1f}%)")
                break
                
        except Exception as e:
            print(f"✗ Test {test_idx + 1} failed: {e}")
            test_results.append({
                'test_index': test_idx,
                'seed': test_seed,
                'error': str(e),
                'coverage_neurons': len(neurons_covered),
                'coverage_percentage': float((len(neurons_covered) / total_neurons) * 100)
            })
    
    print(f"\n=== NEURON SELECTION TEST SUMMARY ===")
    print(f"Total tests completed: {len([r for r in test_results if 'error' not in r])}")
    print(f"Final coverage: {len(neurons_covered)}/{total_neurons} ({(len(neurons_covered)/total_neurons)*100:.1f}%)")
    
    # Find best performing selection
    valid_results = [r for r in test_results if 'error' not in r]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['gradient_norm'])
        print(f"Best gradient norm: {best_result['gradient_norm']:.8f} (test {best_result['test_index'] + 1})")
        
        if best_result['gradient_norm'] > 1e-6:
            print(f"✓ Found promising neuron selection in test {best_result['test_index'] + 1}")
            return test_results, best_result
        else:
            print(f"✗ All tested selections produced near-zero gradients")
    
    return test_results, None


def test_gradient_flow_with_linear_model():
    """
    Test gradient computation with a simple linear model to verify gradient flow.
    
    This creates a trivial optimization problem with known gradients to verify
    that the finite difference computation is working correctly.
    
    Returns:
        Dict with test results
    """
    print(f"\n=== TESTING GRADIENT FLOW WITH SIMPLE LINEAR MODEL ===")
    
    # Create a simple linear model
    class SimpleLinearModel(torch.nn.Module):
        def __init__(self, input_size=10, output_size=5):
            super().__init__()
            self.linear = torch.nn.Linear(input_size, output_size)
            # Initialize with known weights for predictable behavior
            torch.nn.init.constant_(self.linear.weight, 0.1)
            torch.nn.init.constant_(self.linear.bias, 0.0)
            
        def forward(self, x):
            return self.linear(x)
        
        def q_net(self, obs_dict):
            # For compatibility with existing code
            if 'MLP_input' in obs_dict:
                return self.forward(obs_dict['MLP_input'])
            else:
                return self.forward(list(obs_dict.values())[0])
    
    # Create simple model
    input_size, output_size = 10, 5
    model = SimpleLinearModel(input_size=input_size, output_size=output_size)
    print(f"Created simple linear model: {input_size} -> {output_size}")
    
    # Create simple objective function
    class SimpleObjective:
        def __init__(self, model, target_output):
            self.model = model
            self.target_output = target_output  # Shape: (output_size,)
            
        def compute_objective(self, weight_perturbations, mapping_info):
            # Apply perturbations to model
            original_weights = self.model.linear.weight.data.clone()
            
            # For simplicity, just perturb first few weights
            perturb_size = min(len(weight_perturbations), self.model.linear.weight.numel())
            flat_weights = self.model.linear.weight.data.flatten()
            flat_weights[:perturb_size] += torch.tensor(weight_perturbations[:perturb_size], dtype=torch.float32)
            self.model.linear.weight.data = flat_weights.reshape(self.model.linear.weight.shape)
            
            # Simple quadratic objective: ||output - target||^2
            test_input = torch.ones(1, 10)  # Fixed input
            output = self.model(test_input).squeeze(0)
            objective = torch.sum((output - self.target_output) ** 2).item()
            
            # Restore original weights
            self.model.linear.weight.data = original_weights
            
            return objective
            
        def compute_gradient(self, weight_perturbations, mapping_info, debug=False, epsilon=1e-6):
            """Compute gradient using finite differences with debugging."""
            if debug:
                print(f"\n--- Simple Model Gradient Debug ---")
                print(f"Epsilon: {epsilon}")
                print(f"Perturbations shape: {weight_perturbations.shape}")
                print(f"Model weight shape: {self.model.linear.weight.shape}")
            
            gradient = np.zeros_like(weight_perturbations)
            base_objective = self.compute_objective(weight_perturbations, mapping_info)
            
            if debug:
                print(f"Base objective: {base_objective:.8f}")
            
            for i in range(len(weight_perturbations)):
                perturb_plus = weight_perturbations.copy()
                perturb_plus[i] += epsilon
                obj_plus = self.compute_objective(perturb_plus, mapping_info)
                gradient[i] = (obj_plus - base_objective) / epsilon
                
                if debug and i < 5:
                    print(f"  Grad[{i}]: {gradient[i]:.8f} (obj_plus={obj_plus:.8f})")
            
            if debug:
                print(f"Gradient norm: {np.linalg.norm(gradient):.8f}")
                print(f"Non-zero gradients: {np.sum(np.abs(gradient) > 1e-10)}/{len(gradient)}")
            
            return gradient
    
    # Create test scenario
    target_output = torch.tensor([1.0, 0.5, -0.5, 0.0, 0.8])  # Target we want model to output
    objective = SimpleObjective(model, target_output)
    
    # Test gradient computation
    num_params = 20  # Test first 20 parameters
    zero_perturbations = np.zeros(num_params)
    
    print(f"Testing gradient computation with {num_params} parameters...")
    
    # Test with different epsilon values
    epsilon_values = [1e-4, 1e-5, 1e-6, 1e-7]
    results = {}
    
    for eps in epsilon_values:
        print(f"\n--- Testing with epsilon = {eps} ---")
        gradient = objective.compute_gradient(zero_perturbations, {}, debug=True, epsilon=eps)
        
        results[f"eps_{eps}"] = {
            'epsilon': eps,
            'gradient_norm': float(np.linalg.norm(gradient)),
            'non_zero_gradients': int(np.sum(np.abs(gradient) > 1e-10)),
            'max_gradient': float(np.max(np.abs(gradient))),
            'mean_gradient': float(np.mean(np.abs(gradient)))
        }
        
        print(f"Results: norm={results[f'eps_{eps}']['gradient_norm']:.8f}, "
              f"non_zero={results[f'eps_{eps}']['non_zero_gradients']}/{num_params}")
    
    # Find best epsilon
    best_eps = max(epsilon_values, key=lambda eps: results[f"eps_{eps}"]['gradient_norm'])
    
    print(f"\n=== SIMPLE MODEL TEST SUMMARY ===")
    print(f"Best epsilon: {best_eps} (gradient_norm: {results[f'eps_{best_eps}']['gradient_norm']:.8f})")
    
    # Test that we get reasonable gradients
    success = results[f'eps_{best_eps}']['gradient_norm'] > 1e-6
    print(f"Gradient flow test: {'✓ PASSED' if success else '✗ FAILED'}")
    
    return {
        'success': success,
        'best_epsilon': best_eps,
        'results': results,
        'recommended_epsilon': best_eps if success else 1e-4
    }


if __name__ == "__main__":
    # Configure agent and parameters for testing
    agent_path = "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6"
    
    # Set random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Test configuration
    alter_sample_size = 5
    preserve_sample_size = 5
    
    # Optimization parameters
    margin = 0.1
    lambda_sparse = 0.01
    lambda_magnitude = 0.001
    
    # Create results log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"Optimisation_Formulation/GradBasedTooling/optimization_log_{timestamp}.json"
    
    results_log = {
        "timestamp": timestamp,
        "agent_path": agent_path,
        "random_seed": random_seed,
        "debugging_steps": {},
        "final_results": {}
    }
    
    print("=== SYSTEMATIC GRADIENT DEBUGGING ===")
    print(f"Following 6-step debugging protocol:")
    print(f"1. Try different sets of neurons (75% coverage)")
    print(f"2. Add gradient debugging to finite difference computation")
    print(f"3. Check if model requires_grad=True properly set")
    print(f"4. Test with simple linear model to verify gradient flow")
    print(f"5. Test with larger finite difference step size")
    print(f"6. Verify objective function is not already optimal")
    print(f"Logging results to: {log_file}")
    print()
    
    try:
        # SETUP: Load states and model
        print("=== SETUP ===")
        alter_states, preserve_states = load_states_from_tooling(
            agent_path, alter_sample_size, preserve_sample_size
        )
        target_actions = load_target_actions_from_tooling(alter_states)
        model = load_model_from_agent_path(agent_path)
        
        print(f"Loaded {len(alter_states)} ALTER states, {len(preserve_states)} PRESERVE states")
        print(f"Model loaded: {type(model)}")
        
        weight_selector = WeightSelector(target_layers=['q_net.features_extractor.mlp.0', 'q_net.q_net.0', 'q_net.q_net.2'])
        
        # STEP 4: Test with simple linear model first (verify gradient computation works)
        print("\n" + "="*60)
        print("STEP 4: Test with simple linear model to verify gradient flow")
        print("="*60)
        
        linear_test_results = test_gradient_flow_with_linear_model()
        results_log["debugging_steps"]["step4_linear_model"] = linear_test_results
        
        if not linear_test_results['success']:
            print("❌ CRITICAL: Linear model test failed - gradient computation is broken!")
            print("Fix gradient computation before proceeding.")
            raise RuntimeError("Gradient computation fundamentally broken")
        else:
            print("✅ Linear model test passed - gradient computation works")
            recommended_epsilon = linear_test_results['recommended_epsilon']
            print(f"Recommended epsilon: {recommended_epsilon}")
        
        # STEP 1: Try different sets of neurons (75% coverage)
        print("\n" + "="*60)
        print("STEP 1: Try different sets of neurons (75% coverage)")
        print("="*60)
        
        # Create objective for neuron testing
        objective = MarginLossObjective(
            model=model,
            weight_selector=weight_selector,
            alter_states=alter_states,
            target_actions=target_actions,
            margin=margin,
            lambda_sparse=lambda_sparse,
            lambda_magnitude=lambda_magnitude
        )
        
        neuron_test_results, best_neuron_selection = test_multiple_neuron_selections(
            model=model,
            weight_selector=weight_selector,
            objective=objective,
            mapping_base=None,
            num_tests=8,  # Try 8 different selections
            neurons_per_test=15,  # More neurons per test
            seed=random_seed
        )
        
        results_log["debugging_steps"]["step1_neuron_selection"] = {
            "test_results": neuron_test_results,
            "best_selection": best_neuron_selection
        }
        
        # Choose neurons for further testing
        if best_neuron_selection and best_neuron_selection['gradient_norm'] > 1e-6:
            print(f"✅ Found neurons with non-zero gradients!")
            selected_neurons_for_testing = weight_selector.select_random_neurons(
                model, 15, seed=best_neuron_selection['seed']
            )
        else:
            print(f"⚠️ All neuron selections produced near-zero gradients, using default selection")
            selected_neurons_for_testing = weight_selector.select_random_neurons(model, 15, seed=random_seed)
        
        # Create optimization vector with selected neurons
        optimization_vector, mapping_info = weight_selector.create_optimization_vector(
            model, selected_neurons_for_testing
        )
        
        print(f"Using {len(selected_neurons_for_testing)} neurons, {len(optimization_vector)} parameters")
        
        # STEP 2 & 3: Gradient debugging and requires_grad check
        print("\n" + "="*60)
        print("STEP 2 & 3: Gradient debugging and requires_grad check")
        print("="*60)
        
        zero_perturbations = optimization_vector * 0
        gradient_debug = objective.compute_gradient(
            zero_perturbations, mapping_info, debug=True, epsilon=recommended_epsilon
        )
        
        results_log["debugging_steps"]["step2_3_gradient_debug"] = {
            "gradient_norm": float(np.linalg.norm(gradient_debug)),
            "non_zero_gradients": int(np.sum(np.abs(gradient_debug) > 1e-10)),
            "epsilon_used": recommended_epsilon
        }
        
        # STEP 5: Test with larger finite difference step size
        print("\n" + "="*60)
        print("STEP 5: Test with larger finite difference step size")
        print("="*60)
        
        large_epsilon_values = [1e-4, 1e-3, 1e-2]
        epsilon_results = {}
        
        for eps in large_epsilon_values:
            print(f"\n--- Testing epsilon = {eps} ---")
            gradient_large_eps = objective.compute_gradient(
                zero_perturbations, mapping_info, debug=True, epsilon=eps
            )
            
            epsilon_results[f"eps_{eps}"] = {
                'epsilon': eps,
                'gradient_norm': float(np.linalg.norm(gradient_large_eps)),
                'non_zero_gradients': int(np.sum(np.abs(gradient_large_eps) > 1e-10)),
                'max_gradient': float(np.max(np.abs(gradient_large_eps)))
            }
            
            print(f"Results: norm={epsilon_results[f'eps_{eps}']['gradient_norm']:.8f}")
        
        # Find best large epsilon
        best_large_eps = max(large_epsilon_values, 
                           key=lambda eps: epsilon_results[f"eps_{eps}"]['gradient_norm'])
        
        results_log["debugging_steps"]["step5_large_epsilon"] = {
            "epsilon_results": epsilon_results,
            "best_large_epsilon": best_large_eps
        }
        
        print(f"\nBest large epsilon: {best_large_eps}")
        print(f"Gradient norm: {epsilon_results[f'eps_{best_large_eps}']['gradient_norm']:.8f}")
        
        # STEP 6: Verify objective function is not already optimal
        print("\n" + "="*60)
        print("STEP 6: Verify objective function is not already optimal")
        print("="*60)
        
        # Test objective at different perturbation magnitudes
        test_magnitudes = [0.0, 0.001, 0.01, 0.1]
        objective_values = {}
        
        for magnitude in test_magnitudes:
            if magnitude == 0.0:
                test_perturbations = zero_perturbations
            else:
                # Random perturbations of given magnitude
                test_perturbations = np.random.normal(0, magnitude, size=optimization_vector.shape)
            
            obj_value = objective.compute_objective(test_perturbations, mapping_info)
            objective_values[f"mag_{magnitude}"] = float(obj_value)
            
            print(f"Magnitude {magnitude:5.3f}: objective = {obj_value:.8f}")
        
        # Check if objective varies with perturbations
        obj_variation = max(objective_values.values()) - min(objective_values.values())
        is_varying = obj_variation > 1e-6
        
        results_log["debugging_steps"]["step6_objective_analysis"] = {
            "objective_values": objective_values,
            "objective_variation": float(obj_variation),
            "objective_varies": is_varying
        }
        
        print(f"\nObjective variation: {obj_variation:.8f}")
        print(f"Objective varies with perturbations: {'✅ Yes' if is_varying else '❌ No (potentially optimal)'}")
        
        # FINAL ANALYSIS
        print("\n" + "="*60)
        print("FINAL DEBUGGING ANALYSIS")
        print("="*60)
        
        # Determine the likely cause of zero gradients
        issues_found = []
        
        if not linear_test_results['success']:
            issues_found.append("Gradient computation is fundamentally broken")
        
        if all(result.get('gradient_norm', 0) < 1e-8 for result in neuron_test_results if 'gradient_norm' in result):
            issues_found.append("All neuron selections produce near-zero gradients")
        
        if results_log["debugging_steps"]["step2_3_gradient_debug"]["gradient_norm"] < 1e-8:
            issues_found.append("Primary gradient computation produces near-zero gradients")
        
        if max(epsilon_results[f"eps_{eps}"]['gradient_norm'] for eps in large_epsilon_values) < 1e-6:
            issues_found.append("Even large epsilon values produce small gradients")
        
        if not is_varying:
            issues_found.append("Objective function doesn't vary with weight perturbations")
        
        print(f"Issues identified: {len(issues_found)}")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if not is_varying:
            print(f"  • Objective function may already be optimal (margin already satisfied)")
            print(f"  • Try different target actions or larger margin values")
        
        if all(result.get('gradient_norm', 0) < 1e-8 for result in neuron_test_results if 'gradient_norm' in result):
            print(f"  • Selected neurons may have minimal impact on Q-values")
            print(f"  • Try different layer selection or weight selection strategy")
        
        if linear_test_results['success'] and max(epsilon_results[f"eps_{eps}"]['gradient_norm'] for eps in large_epsilon_values) > 1e-6:
            print(f"  • Gradient computation works, but real model has structural issues")
            print(f"  • Consider Q-value computation or model architecture problems")
        
        results_log["final_results"] = {
            "success": len(issues_found) == 0,
            "issues_found": issues_found,
            "primary_recommendation": "Check objective function optimality" if not is_varying else "Check neuron selection",
            "gradient_computation_works": linear_test_results['success']
        }
        
        # Fix JSON serialization for logging
        def make_json_serializable(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.astype(float).tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        # Create JSON-safe results log
        json_safe_log = make_json_serializable(results_log)
        
        # Save detailed results log
        with open(log_file, 'w') as f:
            json.dump(json_safe_log, f, indent=2)
        print(f"\n✓ Results logged to: {log_file}")
        
        print("\n=== SYSTEMATIC DEBUGGING COMPLETED ===")
        
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()
        
        # Log the error
        results_log["final_results"] = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        try:
            def make_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.astype(float).tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                else:
                    return obj
            
            json_safe_log = make_json_serializable(results_log)
            with open(log_file, 'w') as f:
                json.dump(json_safe_log, f, indent=2)
            print(f"Error log saved to: {log_file}")
        except:
            pass 