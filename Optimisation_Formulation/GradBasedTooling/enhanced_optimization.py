#!/usr/bin/env python3
"""
Enhanced Gradient-Based Neural Network Optimization with Comprehensive Logging
Creates both detailed and summary logs in agent's optimisation_results directory.
"""
import sys
import numpy as np
import torch
import json
import shutil
from datetime import datetime
from pathlib import Path
from scipy.optimize import minimize
import copy
import argparse

# Add the correct paths for imports when running from Further_Testing
sys.path.insert(0, '.')
sys.path.insert(0, 'Optimisation_Formulation/GradBasedTooling')

from Optimisation_Formulation.GradBasedTooling.configs.default_config import OPTIMIZATION_CONFIG
from Optimisation_Formulation.GradBasedTooling.utils.integration_utils import load_model_from_agent_path, load_states_from_tooling, load_target_actions_from_tooling
from Optimisation_Formulation.GradBasedTooling.utils.weight_selection import WeightSelector  
from Optimisation_Formulation.GradBasedTooling.objectives.margin_loss_objective import MarginLossObjective
from Optimisation_Formulation.GradBasedTooling.weight_analysis_tool import analyze_optimization_result


class EnhancedOptimizationLogger:
    """Enhanced logger that creates both detailed and summary logs in agent directory"""
    
    def __init__(self, agent_path: str, config: dict):
        self.agent_path = agent_path
        self.config = config
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        self.results_dir = Path(agent_path) / "optimisation_results" / f"optimisation_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()
        
        # Copy config.yaml from original agent for evaluation compatibility
        original_config = Path(agent_path) / "config.yaml"
        if original_config.exists():
            target_config = self.results_dir / "config.yaml"
            shutil.copy2(original_config, target_config)
            print(f"📋 Copied config.yaml to results directory")
    
        # Initialize detailed log data
        self.detailed_log = {
            'metadata': {
                'timestamp': self.start_time.isoformat(),
                'agent_path': agent_path,
                'config': self._make_serializable(config)
            },
            'initial_state': {},
            'iterations': [],
            'final_state': {},
            'result': {}
        }
        self.iteration_count = 0
    
    def log_initial_state(self, selected_neurons, initial_objective, alter_states, preserve_states, target_actions, model, mapping_info):
        """Log initial state with Q-values and neuron information"""
        print(f"\n{'='*70}")
        print(f"ENHANCED GRADIENT-BASED OPTIMIZATION START")
        print(f"{'='*70}")
        print(f"Agent: {Path(self.agent_path).name}")
        print(f"Selected Neurons: {len(selected_neurons)}")
        print(f"ALTER states: {len(alter_states)}")
        print(f"PRESERVE states: {len(preserve_states)}")
        print(f"Initial objective: {initial_objective:.6f}")
        print(f"Results directory: {self.results_dir}")
        
        # Log neuron details
        neuron_details = []
        for neuron in selected_neurons:
            neuron_details.append({
                'layer': neuron[0] if isinstance(neuron, tuple) else str(neuron),
                'neuron_id': neuron[1] if isinstance(neuron, tuple) and len(neuron) > 1 else 0
            })
        
        # Log initial ALTER state Q-values
        initial_alter_states = {}
        for state_id, state_data in alter_states.items():
            obs = state_data['input']
            with torch.no_grad():
                q_vals = model.q_net({'MLP_input': torch.tensor(obs, dtype=torch.float32).unsqueeze(0)})
                q_vals_np = q_vals.squeeze(0).cpu().numpy()
            
            initial_alter_states[state_id] = {
                'input': state_data['input'].tolist() if hasattr(state_data['input'], 'tolist') else state_data['input'],
                'q_values': q_vals_np.tolist(),
                'target_actions': target_actions.get(state_id, [])
            }
        
        # Log initial PRESERVE state Q-values
        initial_preserve_states = {}
        for state_id, state_data in preserve_states.items():
            obs = state_data['input']
            with torch.no_grad():
                q_vals = model.q_net({'MLP_input': torch.tensor(obs, dtype=torch.float32).unsqueeze(0)})
                q_vals_np = q_vals.squeeze(0).cpu().numpy()
                action, _ = model.predict({'MLP_input': obs}, deterministic=True)
            
            initial_preserve_states[state_id] = {
                'input': state_data['input'].tolist() if hasattr(state_data['input'], 'tolist') else state_data['input'],
                'q_values': q_vals_np.tolist(),
                'original_action': int(action)
            }
        
        self.detailed_log['initial_state'] = {
            'objective': float(initial_objective),
            'selected_neurons': neuron_details,
            'neuron_count': len(selected_neurons),
            'weight_count': mapping_info.get('total_weights', 0) if mapping_info else 0,
            'alter_states': initial_alter_states,
            'preserve_states': initial_preserve_states
        }
    
    def log_iteration(self, objective, perturbations):
        """Log iteration progress"""
        self.iteration_count += 1
        
        grad_norm = np.linalg.norm(perturbations)
        max_change = np.max(np.abs(perturbations))
        l1_norm = np.sum(np.abs(perturbations))
        l2_norm_squared = np.sum(perturbations ** 2)
        
        if self.iteration_count <= 20 or self.iteration_count % 20 == 0:
            print(f"Iter {self.iteration_count:3d}: obj={objective:.6f}, norm={grad_norm:.4f}, max={max_change:.4f}, L1={l1_norm:.4f}, L2²={l2_norm_squared:.4f}")
        
        # Store iteration data
        self.detailed_log['iterations'].append({
            'iteration': self.iteration_count,
            'objective': float(objective),
            'perturbations_norm': float(grad_norm),
            'perturbations_max': float(max_change),
            'l1_norm': float(l1_norm),
            'l2_norm_squared': float(l2_norm_squared),
            'perturbations': perturbations.tolist()
        })
    
    def save_results(self, initial_obj, final_obj, final_perturbations, alter_success_info, preserve_info, result, config, best_iteration, optimizer_method):
        """Save both detailed and summary results"""
        
        # Calculate metrics
        l0_norm = float(np.count_nonzero(final_perturbations))
        l1_norm = float(np.sum(np.abs(final_perturbations)))
        l2_norm = float(np.linalg.norm(final_perturbations))
        linf_norm = float(np.max(np.abs(final_perturbations)))
        
        alter_success_rate = alter_success_info.get('success_count', 0) / max(alter_success_info.get('total_count', 1), 1)
        preserve_violation_rate = preserve_info.get('violation_count', 0) / max(preserve_info.get('total_count', 1), 1)
        
        # Complete detailed log
        self.detailed_log['result'] = {
            'success': bool(result.get('success', False)),
            'message': str(result.get('message', '')),
            'initial_objective': float(initial_obj),
            'final_objective': float(final_obj),
            'objective_improvement': float(initial_obj - final_obj),
            'best_iteration': int(best_iteration),
            'total_iterations': int(result.get('nit', 0)),
            'function_evaluations': int(result.get('nfev', 0)),
            'final_perturbations': final_perturbations.tolist(),
            'weight_norms': {
                'l0': l0_norm,
                'l1': l1_norm,
                'l2': l2_norm,
                'linf': linf_norm
            },
            'behavioral_results': {
                'alter_success_count': alter_success_info.get('success_count', 0),
                'alter_total_count': alter_success_info.get('total_count', 0),
                'alter_success_rate': alter_success_rate,
                'preserve_violation_count': preserve_info.get('violation_count', 0),
                'preserve_total_count': preserve_info.get('total_count', 0),
                'preserve_violation_rate': preserve_violation_rate
            },
            'execution_time_seconds': (datetime.now() - self.start_time).total_seconds(),
            'end_time': datetime.now().isoformat()
        }
        
        self.detailed_log['final_state'] = {
            'alter_states': alter_success_info.get('detailed_results', {}),
            'preserve_states': preserve_info.get('detailed_results', {})
        }
        
        # Save detailed log
        detailed_path = self.results_dir / f"detailed_log_{self.timestamp}.json"
        with open(detailed_path, 'w') as f:
            json.dump(self.detailed_log, f, indent=2)
        
        # Create summary log
        summary = {
            "SUMMARY": {
                "agent": Path(self.agent_path).name,
                "timestamp": self.start_time.isoformat(),
                "execution_time_seconds": (datetime.now() - self.start_time).total_seconds(),
                "success": bool(result.get('success', False)),
                "objective_change": {
                    "initial": float(initial_obj),
                    "final": float(final_obj),
                    "improvement": float(initial_obj - final_obj)
                },
                "weight_norms": {
                    "L0": l0_norm,
                    "L1": l1_norm,
                    "L2": l2_norm,
                    "L_infinity": linf_norm
                },
                "behavioral_success": {
                    "alter_states_success_rate": float(alter_success_rate),
                    "alter_states_changed": f"{alter_success_info.get('success_count', 0)}/{alter_success_info.get('total_count', 0)}",
                    "preserve_states_violation_rate": float(preserve_violation_rate),
                    "preserve_states_violated": f"{preserve_info.get('violation_count', 0)}/{preserve_info.get('total_count', 0)}"
                },
                "hyperparameters": {
                    "margin": config.get('margin'),
                    "target_layers": config.get('target_layers'),
                    "weights_per_neuron": config.get('weights_per_neuron'),
                    "max_iterations": config.get('max_iterations'),
                    "optimizer": optimizer_method,
                    "seed": config.get('seed')
                },
                "optimization_stats": {
                    "best_iteration": int(best_iteration),
                    "total_iterations": int(result.get('nit', 0)),
                    "function_evaluations": int(result.get('nfev', 0))
                }
            },
            "DETAILED_STATE_CHANGES": {
                "alter_states": alter_success_info.get('detailed_results', {}),
                "preserve_states": preserve_info.get('detailed_results', {})
            }
        }
        
        # Save summary
        summary_path = self.results_dir / f"summary_{self.timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Status: {'✅ SUCCESS' if summary['SUMMARY']['success'] else '❌ FAILED'}")
        print(f"Best found at iteration: {best_iteration}/{result.get('nit', 0)}")
        print(f"Objective: {initial_obj:.6f} → {final_obj:.6f}")
        print(f"Improvement: {initial_obj - final_obj:.6f}")
        print(f"\nWeight Changes:")
        print(f"  L0 (nonzero): {l0_norm:.0f}")
        print(f"  L1: {l1_norm:.6f}")
        print(f"  L2: {l2_norm:.6f}")
        print(f"  L∞: {linf_norm:.6f}")
        print(f"\nBehavioral Changes:")
        print(f"  ALTER success: {alter_success_info.get('success_count', 0)}/{alter_success_info.get('total_count', 0)} ({alter_success_rate:.1%})")
        print(f"  PRESERVE violations: {preserve_info.get('violation_count', 0)}/{preserve_info.get('total_count', 0)} ({preserve_violation_rate:.1%})")
        print(f"\n📁 Logs saved to:")
        print(f"   Detailed: {detailed_path}")
        print(f"   Summary:  {summary_path}")
        print(f"{'='*70}")
        
        return summary
    
    def _make_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


def analyze_alter_states(model, weight_selector, alter_states, target_actions, perturbations, mapping_info):
    """Analyze ALTER state success"""
    try:
        weight_selector.apply_perturbations(model, perturbations, mapping_info)
        
        success_count = 0
        total_count = len(alter_states)
        detailed_results = {}
        
        for state_id, state_data in alter_states.items():
            obs = state_data['input']
            target_action_list = target_actions.get(state_id, [])
            
            action, _ = model.predict({'MLP_input': obs}, deterministic=True)
            current_action = int(action)
            
            success = current_action in target_action_list
            if success:
                success_count += 1
            
            detailed_results[state_id] = {
                'current_action': current_action,
                'target_actions': target_action_list,
                'success': success
            }
        
        return {
            'success_count': success_count,
            'total_count': total_count,
            'detailed_results': detailed_results
        }
        
    finally:
        weight_selector.apply_perturbations(model, -perturbations, mapping_info)


def analyze_preserve_states(model, weight_selector, preserve_states, perturbations, mapping_info):
    """Analyze PRESERVE state violations"""
    # Get original actions
    original_actions = {}
    for state_id, state_data in preserve_states.items():
        obs = state_data['input']
        action, _ = model.predict({'MLP_input': obs}, deterministic=True)
        original_actions[state_id] = int(action)
    
    try:
        weight_selector.apply_perturbations(model, perturbations, mapping_info)
        
        violation_count = 0
        total_count = len(preserve_states)
        detailed_results = {}
        
        for state_id, state_data in preserve_states.items():
            obs = state_data['input']
            original_action = original_actions[state_id]
            
            action, _ = model.predict({'MLP_input': obs}, deterministic=True)
            current_action = int(action)
            
            violated = current_action != original_action
            if violated:
                violation_count += 1
            
            detailed_results[state_id] = {
                'original_action': original_action,
                'current_action': current_action,
                'violated': violated
            }
        
        return {
            'violation_count': violation_count,
            'total_count': total_count,
            'detailed_results': detailed_results
        }
        
    finally:
        weight_selector.apply_perturbations(model, -perturbations, mapping_info)


def create_model_backup_and_verify(model):
    """
    Create a deep copy of the model and verification system to ensure original is never modified.
    
    Args:
        model: Original PyTorch model
        
    Returns:
        model_copy: Deep copy for optimization
        original_state_dict: State dict of original for verification
        verification_fn: Function to verify original is unchanged
    """
    print("🔒 Creating model backup and verification system...")
    
    # Create deep copy of the model for optimization
    model_copy = copy.deepcopy(model)
    
    # Store original state dict for verification
    original_state_dict = {}
    for name, param in model.named_parameters():
        original_state_dict[name] = param.data.clone().detach()
    
    def verify_original_unchanged():
        """Verify that the original model hasn't been modified"""
        violations = []
        for name, param in model.named_parameters():
            original_tensor = original_state_dict[name]
            current_tensor = param.data
            
            if not torch.equal(original_tensor, current_tensor):
                max_diff = torch.max(torch.abs(original_tensor - current_tensor)).item()
                violations.append(f"  {name}: max difference = {max_diff}")
        
        if violations:
            print("❌ CRITICAL: Original model has been modified!")
            for violation in violations:
                print(violation)
            return False
        else:
            print("✅ VERIFIED: Original model unchanged")
            return True
    
    # Initial verification
    print("📋 Initial model backup created:")
    print(f"   Original parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"   Copy parameters: {sum(p.numel() for p in model_copy.parameters())}")
    verify_original_unchanged()
    
    return model_copy, original_state_dict, verify_original_unchanged


def run_enhanced_optimization(agent_path: str = None, config: dict = None, config_path: str = None):
    """
    Run optimization with enhanced logging that saves to agent's directory.
    Creates both detailed and summary logs in timestamped subdirectory.
    IMPORTANT: Always operates on a copy of the model to preserve original.
    
    Args:
        agent_path: Path to the agent directory
        config: Configuration dictionary (takes precedence over config_path)
        config_path: Path to JSON config file to load
    """
    if agent_path is None:
        agent_path = "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6/"
    
    # Handle config loading
    if config is None:
        if config_path is not None:
            # Load config from file
            from Optimisation_Formulation.GradBasedTooling.configs.default_config import create_config_from_file
            config = create_config_from_file(config_path)
        else:
            # Use default config
            config = OPTIMIZATION_CONFIG.copy()
    
    # Import here to avoid scope issues
    from Optimisation_Formulation.GradBasedTooling.utils.integration_utils import load_model_from_agent_path
    
    # Initialize enhanced logger
    logger = EnhancedOptimizationLogger(agent_path, config)
    
    # Load data
    alter_states, preserve_states = load_states_from_tooling(agent_path, 3, 5)
    original_model = load_model_from_agent_path(agent_path)
    target_actions = load_target_actions_from_tooling(alter_states)
    
    # 🔒 CREATE MODEL COPY AND VERIFICATION SYSTEM
    model, original_state_dict, verify_original_unchanged = create_model_backup_and_verify(original_model)
    
    # Setup optimization components (now using the copy)
    weight_selector = WeightSelector(
        target_layers=config.get('target_layers'),
        weights_per_neuron=config.get('weights_per_neuron')
    )
    selected_neurons = weight_selector.select_random_neurons(
        model, 
        config.get('num_neurons', 2), 
        config.get('seed')
    )
    initial_weights, mapping_info = weight_selector.create_optimization_vector(model, selected_neurons)
    
    # Calculate bounds upfront before optimization
    def calculate_bounds(initial_weights, bounds_config):
        """Calculate bounds array once before optimization starts"""
        if bounds_config is None:
            return None
            
        # Handle different bound types
        if isinstance(bounds_config, tuple) and len(bounds_config) == 2:
            # Strict bounds: same for all weights
            return [bounds_config] * len(initial_weights)
        elif isinstance(bounds_config, str) and bounds_config == 'adaptive':
            # Adaptive bounds: relative to weight distribution statistics  
            adaptive_factor = config.get('adaptive_bounds_factor', 2.0)  # Default 2x std
            min_bound = config.get('adaptive_bounds_min', 0.01)
            max_bound = config.get('adaptive_bounds_max', 1.0)
            
            # Calculate bounds based on weight distribution
            weight_std = np.std(initial_weights) if len(initial_weights) > 1 else np.abs(initial_weights[0])
            base_bound = adaptive_factor * weight_std
            bound_magnitude = max(min_bound, min(base_bound, max_bound))
            
            bounds = [(-bound_magnitude, bound_magnitude)] * len(initial_weights)
            print(f"📏 Adaptive bounds: ±{bound_magnitude:.4f} (std={weight_std:.6f}, factor={adaptive_factor})")
            return bounds
        elif isinstance(bounds_config, dict):
            # Handle dictionary specification for mixed bounds
            if bounds_config.get('type') == 'adaptive':
                adaptive_factor = bounds_config.get('factor', 0.1)
                min_bound = bounds_config.get('min_bound', 1e-6)
                max_bound = bounds_config.get('max_bound', 1.0)
                bounds = []
                for weight in initial_weights:
                    abs_weight = abs(weight) if abs(weight) > 1e-8 else 1e-8
                    bound_magnitude = adaptive_factor * abs_weight
                    # Clamp to min/max bounds
                    bound_magnitude = max(min_bound, min(bound_magnitude, max_bound))
                    bounds.append((-bound_magnitude, bound_magnitude))
                return bounds
            elif bounds_config.get('type') == 'strict':
                bound_val = bounds_config.get('value', 0.3)
                return [(-bound_val, bound_val)] * len(initial_weights)
        elif isinstance(bounds_config, list):
            # Explicit bounds list
            return bounds_config
        else:
            print(f"⚠️  Unknown bounds configuration: {bounds_config}")
            return None
    
    objective = MarginLossObjective(
        model=model,
        weight_selector=weight_selector,
        alter_states=alter_states,
        target_actions=target_actions,
        preserve_states=preserve_states,
        margin=config.get('margin'),
        config=config
    )
    
    # Compute initial objective and log initial state
    initial_objective = objective.compute_objective(np.zeros_like(initial_weights), mapping_info)
    logger.log_initial_state(selected_neurons, initial_objective, alter_states, preserve_states, target_actions, model, mapping_info)
    
    # 🔒 VERIFICATION: Check original model before optimization
    print("\n🔒 PRE-OPTIMIZATION VERIFICATION:")
    verify_original_unchanged()
    
    # Track optimization with detailed logging
    iteration_count = 0
    best_objective = initial_objective
    best_perturbations = np.zeros_like(initial_weights)
    best_iteration = 0  # Track which iteration was best
    max_allowed_iterations = config.get('max_iterations', 200)
    
    def logged_objective(x):
        nonlocal iteration_count, best_objective, best_perturbations, best_iteration
        iteration_count += 1
        
        # Hard limit check - prevent runaway optimization
        if iteration_count > max_allowed_iterations:
            print(f"⚠️  STOPPING: Reached hard iteration limit ({max_allowed_iterations})")
            return float('inf')  # Return high value to stop optimization
        
        obj_val = objective.compute_objective(x, mapping_info)
        
        if obj_val < best_objective:
            best_objective = obj_val
            best_perturbations = x.copy()
            best_iteration = iteration_count
        
        # Log iteration progress
        logger.log_iteration(obj_val, x)
        
        # 🔒 PERIODIC VERIFICATION: Check original model every 50 iterations
        if iteration_count % 50 == 0:
            print(f"\n🔒 MID-OPTIMIZATION VERIFICATION (iteration {iteration_count}):")
            verify_original_unchanged()
        
        return obj_val
    
    # Run optimization
    optimizer_method = config.get('optimizer_method', 'Nelder-Mead')
    
    # Set optimizer-specific iteration limits
    optimizer_options = {
        'disp': False
    }
    
    # Add method-specific options and enforce iteration limits
    if optimizer_method == 'Powell':
        # Powell doesn't use maxiter directly, use maxfev instead
        optimizer_options['maxfev'] = max_allowed_iterations * 10  # Function evaluations
        optimizer_options['ftol'] = config.get('function_tolerance', 1e-6)
        bounds = None  # Powell doesn't support bounds
    elif optimizer_method == 'BFGS':
        optimizer_options['maxiter'] = max_allowed_iterations
        optimizer_options['gtol'] = config.get('gradient_tolerance', 1e-6)
        # BFGS cannot handle bounds - use L-BFGS-B if bounds specified
        bounds_spec = config.get('bounds', None)
        if bounds_spec is not None:
            print(f"⚠️  BFGS cannot handle bounds. Switching to L-BFGS-B optimizer.")
            optimizer_method = 'L-BFGS-B'
            # Create bounds array for each weight
            bounds = calculate_bounds(initial_weights, bounds_spec)
        else:
            bounds = None
    elif optimizer_method == 'L-BFGS-B':
        optimizer_options['maxiter'] = max_allowed_iterations
        optimizer_options['gtol'] = config.get('gradient_tolerance', 1e-6)
        bounds_spec = config.get('bounds', None)
        if bounds_spec is not None:
            # Create bounds array for each weight
            bounds = calculate_bounds(initial_weights, bounds_spec)
        else:
            bounds = None
    elif optimizer_method == 'CG':
        optimizer_options['maxiter'] = max_allowed_iterations
        optimizer_options['gtol'] = config.get('gradient_tolerance', 1e-6)
        bounds = None
    elif optimizer_method == 'Nelder-Mead':
        optimizer_options['maxiter'] = max_allowed_iterations
        optimizer_options['xatol'] = config.get('xatol', 1e-6)
        optimizer_options['fatol'] = config.get('fatol', 1e-6)
        bounds = None
    else:
        # Default for other optimizers
        optimizer_options['maxiter'] = max_allowed_iterations
        bounds_spec = config.get('bounds', None)
        if bounds_spec is not None:
            # Create bounds array for each weight
            bounds = calculate_bounds(initial_weights, bounds_spec)
        else:
            bounds = None
    
    print(f"🔧 Starting {optimizer_method} optimization with max {max_allowed_iterations} iterations")
    
    result = minimize(
        logged_objective,
        np.zeros_like(initial_weights),
        method=optimizer_method,
        bounds=bounds,
        options=optimizer_options
    )
    
    print(f"✅ Optimization completed: {iteration_count} iterations, best at iteration {best_iteration}")
    
    # 🔒 FINAL VERIFICATION: Ensure original model unchanged after optimization
    print(f"\n🔒 POST-OPTIMIZATION VERIFICATION:")
    original_preserved = verify_original_unchanged()
    
    final_objective = result.fun
    final_perturbations = best_perturbations  # Use best perturbations found
    
    # Analyze results
    alter_success_info = analyze_alter_states(
        model, weight_selector, alter_states, target_actions, 
        final_perturbations, mapping_info
    )
    preserve_violation_info = analyze_preserve_states(
        model, weight_selector, preserve_states,
        final_perturbations, mapping_info
    )
    
    # 🔒 VERIFICATION: Check original model after analysis
    print(f"\n🔒 POST-ANALYSIS VERIFICATION:")
    verify_original_unchanged()
    
    # Perform detailed weight analysis to identify large changes
    print(f"\n{'='*70}")
    print("PERFORMING DETAILED WEIGHT ANALYSIS")
    print(f"{'='*70}")
    
    weight_analysis_threshold = config.get('weight_analysis_threshold', 100.0)
    weight_analysis_results = analyze_optimization_result(
        agent_path=agent_path,
        perturbations=final_perturbations,
        neuron_specs=selected_neurons,
        mapping_info=mapping_info,
        preserve_states=preserve_states,
        threshold=weight_analysis_threshold
    )
    
    # Save perturbed model if requested
    perturbed_model_path = None
    if config.get('save_perturbed_model', True):
        print(f"\n{'='*70}")
        print("SAVING PERTURBED MODEL WITH ACTUAL WEIGHT CHANGES")
        print(f"{'='*70}")
        
        # Import dependencies
        from stable_baselines3 import DQN
        import shutil
        from pathlib import Path
        
        try:
            if np.any(final_perturbations != 0):
                print(f"Creating new model with {np.sum(final_perturbations != 0)} perturbations applied...")
                
                # Load original model just to get its parameters
                original_model = load_model_from_agent_path(agent_path)
                
                # Get the modified state dict
                state_dict = original_model.state_dict().copy()
                
                # Apply perturbations to the state dict copy
                for layer_name, (start_idx, end_idx) in mapping_info['layer_slices'].items():
                    layer_perturbations = final_perturbations[start_idx:end_idx]
                    
                    # Find corresponding state dict key
                    state_dict_key = None
                    if layer_name == 'q_net.features_extractor.mlp.0':
                        state_dict_key = 'q_net.features_extractor.mlp.0.weight'
                    elif layer_name == 'q_net.q_net.0':
                        state_dict_key = 'q_net.q_net.0.weight'
                    elif layer_name == 'q_net.q_net.2':
                        state_dict_key = 'q_net.q_net.2.weight'
                    
                    if state_dict_key and state_dict_key in state_dict:
                        # Get weight indices for this layer using the same logic as apply_perturbations
                        neuron_specs_for_layer = [(ln, ni) for ln, ni in mapping_info['neuron_specs'] if ln == layer_name]
                        weight_indices_dict = weight_selector.neurons_to_weight_indices(original_model, neuron_specs_for_layer)
                        weight_indices = weight_indices_dict[layer_name]
                        
                        # Apply perturbations to the state dict tensor
                        original_shape = state_dict[state_dict_key].shape
                        flattened_weights = state_dict[state_dict_key].flatten().clone()
                        
                        # Apply perturbations using the correct weight indices
                        flattened_weights[weight_indices] += torch.from_numpy(layer_perturbations).float()
                        
                        state_dict[state_dict_key] = flattened_weights.reshape(original_shape)
                
                # Create a completely new DQN instance from the original config
                original_agent_path = Path(agent_path) / "agent.zip"
                temp_agent_path = logger.results_dir / "temp_agent.zip"
                
                # Copy the original agent as starting point
                shutil.copy2(original_agent_path, temp_agent_path)
                
                # Load the copied agent
                new_model = DQN.load(str(temp_agent_path))
                
                # Apply the modified state dict to the new model
                new_model.policy.load_state_dict(state_dict)
                
                # Save the new model with modifications
                perturbed_model_path = logger.results_dir / "agent.zip"
                new_model.save(str(perturbed_model_path))
                
                # Clean up temp file
                temp_agent_path.unlink()
                
                print(f"Applied perturbations to fresh model instance")
            else:
                print("No perturbations to apply - copying original model")
                # Just copy the original if no changes
                original_model_path = Path(agent_path) / "agent.zip"
                perturbed_model_path = logger.results_dir / "agent.zip"
                shutil.copy2(original_model_path, perturbed_model_path)
            
            print(f"💾 Perturbed model saved to: {perturbed_model_path}")
            print(f"   Perturbation stats: L1={np.sum(np.abs(final_perturbations)):.6f}, L2={np.sqrt(np.sum(final_perturbations**2)):.6f}")
            print(f"   Total modified weights: {np.sum(final_perturbations != 0)}")
            print(f"   Largest change: {np.max(np.abs(final_perturbations)):.6f} at index {np.argmax(np.abs(final_perturbations))}")
            
        except Exception as e:
            print(f"❌ Error saving perturbed model: {e}")
            import traceback
            traceback.print_exc()
            perturbed_model_path = None
    
    # Save results with enhanced logging
    summary = logger.save_results(
        initial_objective, best_objective, final_perturbations,
        alter_success_info, preserve_violation_info, result, config, best_iteration, optimizer_method
    )
    
    # 🔒 FINAL VERIFICATION: Last check before returning
    print(f"\n🔒 FINAL MODEL INTEGRITY CHECK:")
    final_verification = verify_original_unchanged()
    
    # Add verification status to summary
    summary['MODEL_VERIFICATION'] = {
        'original_preserved': original_preserved and final_verification,
        'total_parameters': sum(p.numel() for p in original_model.parameters()),
        'verification_passed': final_verification,
        'perturbed_model_saved': perturbed_model_path is not None,
        'perturbed_model_path': str(perturbed_model_path) if perturbed_model_path else None
    }
    
    if not final_verification:
        print("🚨 WARNING: Original model integrity compromised!")
    else:
        print("🛡️  SUCCESS: Original model fully preserved")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Gradient-Based Neural Network Optimization')
    parser.add_argument('--agent_path', type=str, required=True,
                        help='Path to the agent directory (required)')
    parser.add_argument('--config_path', type=str, 
                        help='Path to JSON configuration file (optional, uses default config if not provided)')
    
    args = parser.parse_args()
    
    # Run optimization with command line arguments
    summary = run_enhanced_optimization(
        agent_path=args.agent_path,
        config_path=args.config_path
    )
    print(f"\n🎯 Optimization complete! Check the agent's optimisation_results directory for detailed logs.") 