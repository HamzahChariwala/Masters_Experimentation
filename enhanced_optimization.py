#!/usr/bin/env python3
"""
Enhanced Gradient-Based Neural Network Optimization with Comprehensive Logging
Creates both detailed and summary logs in agent's optimisation_results directory.
"""
import sys
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, '.')

from configs.default_config import OPTIMIZATION_CONFIG
from utils.integration_utils import load_model_from_agent_path, load_states_from_tooling, load_target_actions_from_tooling
from utils.weight_selection import WeightSelector  
from objectives.margin_loss_objective import MarginLossObjective


class EnhancedOptimizationLogger:
    """Enhanced logger that creates both detailed and summary logs in agent directory"""
    
    def __init__(self, agent_path: str, config: dict):
        self.agent_path = agent_path
        self.config = config
        self.timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        
        # Create timestamped subdirectory in agent's optimisation_results
        self.results_dir = Path(agent_path) / "optimisation_results" / f"optimisation_{self.timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = datetime.now()
        
        # Initialize detailed log data structure
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
    
    def log_initial_state(self, states_info: dict, selected_neurons: list, initial_objective: float, 
                         alter_states: dict, preserve_states: dict, target_actions: dict, model, mapping_info):
        """Log initial optimization state with all Q-values and neuron information"""
        
        print(f"\n{'='*70}")
        print(f"ENHANCED GRADIENT-BASED OPTIMIZATION START")
        print(f"{'='*70}")
        print(f"Agent: {Path(self.agent_path).name}")
        print(f"Selected Neurons: {len(selected_neurons)}")
        print(f"ALTER states: {states_info.get('alter_count', 0)}")
        print(f"PRESERVE states: {states_info.get('preserve_count', 0)}")
        print(f"Initial objective: {initial_objective:.6f}")
        print(f"Results directory: {self.results_dir}")
        
        # Log detailed neuron and weight information
        neuron_details = []
        for neuron in selected_neurons:
            neuron_details.append({
                'layer': neuron[0] if isinstance(neuron, tuple) else str(neuron),
                'neuron_id': neuron[1] if isinstance(neuron, tuple) and len(neuron) > 1 else 0
            })
        
        # Log initial ALTER state Q-values and analysis
        initial_alter_states = {}
        for state_id, state_data in alter_states.items():
            obs = state_data['input']
            with torch.no_grad():
                q_vals = model.q_net({'MLP_input': torch.tensor(obs, dtype=torch.float32).unsqueeze(0)})
                q_vals_np = q_vals.squeeze(0).cpu().numpy()
            
            target_actions_list = target_actions.get(state_id, [])
            if target_actions_list:
                target_q_values = [q_vals_np[action] for action in target_actions_list if action < len(q_vals_np)]
                non_target_actions = [i for i in range(len(q_vals_np)) if i not in target_actions_list]
                non_target_q_values = [q_vals_np[i] for i in non_target_actions]
                
                max_target_q = max(target_q_values) if target_q_values else float('-inf')
                max_non_target_q = max(non_target_q_values) if non_target_q_values else float('-inf')
                margin_violation = max_non_target_q + self.config.get('margin', 0.1) - max_target_q
            else:
                max_target_q = max_non_target_q = margin_violation = 0.0
            
            initial_alter_states[state_id] = {
                'input': state_data['input'].tolist() if hasattr(state_data['input'], 'tolist') else state_data['input'],
                'q_values': q_vals_np.tolist(),
                'target_actions': target_actions_list,
                'max_target_q': float(max_target_q),
                'max_non_target_q': float(max_non_target_q),
                'margin_violation': float(margin_violation)
            }
        
        # Log initial PRESERVE state Q-values and actions
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
        
        # Store in detailed log
        self.detailed_log['initial_state'] = {
            'objective': float(initial_objective),
            'selected_neurons': neuron_details,
            'neuron_count': len(selected_neurons),
            'weight_count': mapping_info.get('total_weights', 0) if mapping_info else 0,
            'alter_states': initial_alter_states,
            'preserve_states': initial_preserve_states,
            'states_info': states_info
        }
    
    def log_iteration(self, iteration: int, objective: float, perturbations: np.ndarray, 
                     alter_states: dict, preserve_states: dict, target_actions: dict, model, weight_selector, mapping_info):
        """Log detailed information for each iteration"""
        
        if iteration <= 20 or iteration % 20 == 0:
            grad_norm = np.linalg.norm(perturbations)
            max_change = np.max(np.abs(perturbations))
            print(f"Iter {iteration:3d}: obj={objective:.6f}, norm={grad_norm:.4f}, max={max_change:.4f}")
        
        # For detailed logging, capture Q-values with perturbations applied
        log_entry = {
            'iteration': iteration,
            'objective': float(objective),
            'perturbations_norm': float(np.linalg.norm(perturbations)),
            'perturbations_max': float(np.max(np.abs(perturbations))),
            'perturbations': perturbations.tolist()
        }
        
        # Apply perturbations temporarily for detailed analysis
        if iteration % 50 == 0 or iteration <= 10:  # Log detailed info every 50 iterations or first 10
            weight_selector.apply_perturbations(model, perturbations, mapping_info)
            
            # Log current ALTER state Q-values
            current_alter_q = {}
            for state_id, state_data in alter_states.items():
                obs = state_data['input']
                with torch.no_grad():
                    q_vals = model.q_net({'MLP_input': torch.tensor(obs, dtype=torch.float32).unsqueeze(0)})
                    q_vals_np = q_vals.squeeze(0).cpu().numpy()
                current_alter_q[state_id] = q_vals_np.tolist()
            
            # Log current PRESERVE state Q-values and actions
            current_preserve_q = {}
            preserve_action_changes = {}
            for state_id, state_data in preserve_states.items():
                obs = state_data['input']
                with torch.no_grad():
                    q_vals = model.q_net({'MLP_input': torch.tensor(obs, dtype=torch.float32).unsqueeze(0)})
                    q_vals_np = q_vals.squeeze(0).cpu().numpy()
                    action, _ = model.predict({'MLP_input': obs}, deterministic=True)
                
                current_preserve_q[state_id] = q_vals_np.tolist()
                original_action = self.detailed_log['initial_state']['preserve_states'][state_id]['original_action']
                preserve_action_changes[state_id] = {
                    'original_action': original_action,
                    'current_action': int(action),
                    'action_changed': int(action) != original_action
                }
            
            log_entry['detailed_analysis'] = {
                'alter_q_values': current_alter_q,
                'preserve_q_values': current_preserve_q,
                'preserve_action_changes': preserve_action_changes
            }
            
            # Restore original state
            weight_selector.apply_perturbations(model, -perturbations, mapping_info)
        
        self.detailed_log['iterations'].append(log_entry)
    
    def log_final_results(self, result: dict, initial_objective: float, final_perturbations: np.ndarray,
                         alter_states: dict, preserve_states: dict, target_actions: dict, 
                         model, weight_selector, mapping_info):
        """Log final results and create both detailed and summary logs"""
        
        # Apply final perturbations for analysis
        weight_selector.apply_perturbations(model, final_perturbations, mapping_info)
        
        # Analyze final ALTER states
        final_alter_states = {}
        alter_success_count = 0
        for state_id, state_data in alter_states.items():
            obs = state_data['input']
            with torch.no_grad():
                q_vals = model.q_net({'MLP_input': torch.tensor(obs, dtype=torch.float32).unsqueeze(0)})
                q_vals_np = q_vals.squeeze(0).cpu().numpy()
                action, _ = model.predict({'MLP_input': obs}, deterministic=True)
            
            original_q = np.array(self.detailed_log['initial_state']['alter_states'][state_id]['q_values'])
            q_change = q_vals_np - original_q
            
            target_actions_list = target_actions.get(state_id, [])
            current_action = int(action)
            success = current_action in target_actions_list
            if success:
                alter_success_count += 1
            
            if target_actions_list:
                target_q_values = [q_vals_np[action] for action in target_actions_list if action < len(q_vals_np)]
                non_target_actions = [i for i in range(len(q_vals_np)) if i not in target_actions_list]
                non_target_q_values = [q_vals_np[i] for i in non_target_actions]
                
                max_target_q = max(target_q_values) if target_q_values else float('-inf')
                max_non_target_q = max(non_target_q_values) if non_target_q_values else float('-inf')
                margin_violation = max_non_target_q + self.config.get('margin', 0.1) - max_target_q
                
                original_margin_violation = self.detailed_log['initial_state']['alter_states'][state_id]['margin_violation']
                margin_improvement = original_margin_violation - margin_violation
            else:
                max_target_q = max_non_target_q = margin_violation = margin_improvement = 0.0
            
            final_alter_states[state_id] = {
                'original_q_values': original_q.tolist(),
                'final_q_values': q_vals_np.tolist(),
                'q_value_changes': q_change.tolist(),
                'target_actions': target_actions_list,
                'current_action': current_action,
                'success': success,
                'final_max_target_q': float(max_target_q),
                'final_max_non_target_q': float(max_non_target_q),
                'final_margin_violation': float(margin_violation),
                'margin_improvement': float(margin_improvement)
            }
        
        # Analyze final PRESERVE states
        final_preserve_states = {}
        preserve_violation_count = 0
        for state_id, state_data in preserve_states.items():
            obs = state_data['input']
            with torch.no_grad():
                q_vals = model.q_net({'MLP_input': torch.tensor(obs, dtype=torch.float32).unsqueeze(0)})
                q_vals_np = q_vals.squeeze(0).cpu().numpy()
                action, _ = model.predict({'MLP_input': obs}, deterministic=True)
            
            original_q = np.array(self.detailed_log['initial_state']['preserve_states'][state_id]['q_values'])
            original_action = self.detailed_log['initial_state']['preserve_states'][state_id]['original_action']
            q_change = q_vals_np - original_q
            current_action = int(action)
            violated = current_action != original_action
            
            if violated:
                preserve_violation_count += 1
            
            final_preserve_states[state_id] = {
                'original_q_values': original_q.tolist(),
                'final_q_values': q_vals_np.tolist(),
                'q_value_changes': q_change.tolist(),
                'original_action': original_action,
                'current_action': current_action,
                'violated': violated
            }
        
        # Restore original state
        weight_selector.apply_perturbations(model, -final_perturbations, mapping_info)
        
        # Calculate metrics
        final_objective = result.get('fun', 0.0)
        l0_norm = float(np.count_nonzero(final_perturbations))
        l1_norm = float(np.sum(np.abs(final_perturbations)))
        l2_norm = float(np.linalg.norm(final_perturbations))
        linf_norm = float(np.max(np.abs(final_perturbations)))
        
        alter_success_rate = alter_success_count / max(len(alter_states), 1)
        preserve_violation_rate = preserve_violation_count / max(len(preserve_states), 1)
        
        # Complete detailed log
        self.detailed_log['final_state'] = {
            'alter_states': final_alter_states,
            'preserve_states': final_preserve_states
        }
        
        self.detailed_log['result'] = {
            'success': bool(result.get('success', False)),
            'message': str(result.get('message', '')),
            'initial_objective': float(initial_objective),
            'final_objective': float(final_objective),
            'objective_improvement': float(initial_objective - final_objective),
            'iterations': int(result.get('nit', 0)),
            'function_evaluations': int(result.get('nfev', 0)),
            'final_perturbations': final_perturbations.tolist(),
            'weight_norms': {
                'l0': l0_norm,
                'l1': l1_norm,
                'l2': l2_norm,
                'linf': linf_norm
            },
            'behavioral_results': {
                'alter_success_count': alter_success_count,
                'alter_total_count': len(alter_states),
                'alter_success_rate': alter_success_rate,
                'preserve_violation_count': preserve_violation_count,
                'preserve_total_count': len(preserve_states),
                'preserve_violation_rate': preserve_violation_rate
            },
            'execution_time_seconds': (datetime.now() - self.start_time).total_seconds(),
            'end_time': datetime.now().isoformat()
        }
        
        # Save detailed log
        detailed_path = self.results_dir / f"detailed_log_{self.timestamp}.json"
        with open(detailed_path, 'w') as f:
            json.dump(self.detailed_log, f, indent=2)
        
        # Create and save summary log
        summary_log = self._create_summary_log()
        summary_path = self.results_dir / f"summary_{self.timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_log, f, indent=2)
        
        # Print results
        self._print_console_summary()
        
        print(f"\n📁 Logs saved to:")
        print(f"   Detailed: {detailed_path}")
        print(f"   Summary:  {summary_path}")
        
        return summary_log
    
    def _create_summary_log(self) -> dict:
        """Create concise summary log with key metrics"""
        result = self.detailed_log['result']
        
        return {
            "SUMMARY": {
                "agent": Path(self.agent_path).name,
                "timestamp": self.start_time.isoformat(),
                "execution_time_seconds": result['execution_time_seconds'],
                "success": result['success'],
                "objective_change": {
                    "initial": result['initial_objective'],
                    "final": result['final_objective'],
                    "improvement": result['objective_improvement']
                },
                "weight_norms": {
                    "L0": result['weight_norms']['l0'],
                    "L1": result['weight_norms']['l1'],
                    "L2": result['weight_norms']['l2'],
                    "L_infinity": result['weight_norms']['linf']
                },
                "behavioral_success": {
                    "alter_states_success_rate": result['behavioral_results']['alter_success_rate'],
                    "alter_states_changed": f"{result['behavioral_results']['alter_success_count']}/{result['behavioral_results']['alter_total_count']}",
                    "preserve_states_violation_rate": result['behavioral_results']['preserve_violation_rate'],
                    "preserve_states_violated": f"{result['behavioral_results']['preserve_violation_count']}/{result['behavioral_results']['preserve_total_count']}"
                },
                "hyperparameters": {
                    "margin": self.config.get('margin'),
                    "target_layers": self.config.get('target_layers'),
                    "weights_per_neuron": self.config.get('weights_per_neuron'),
                    "optimizer": "Nelder-Mead",
                    "max_iterations": self.config.get('max_iterations'),
                    "seed": self.config.get('seed')
                },
                "optimization_stats": {
                    "iterations": result['iterations'],
                    "function_evaluations": result['function_evaluations']
                }
            }
        }
    
    def _print_console_summary(self):
        """Print concise console summary"""
        result = self.detailed_log['result']
        
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        
        success = result['success']
        print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"Iterations: {result['iterations']}")
        print(f"Objective: {result['initial_objective']:.6f} → {result['final_objective']:.6f}")
        print(f"Improvement: {result['objective_improvement']:.6f}")
        
        # Weight changes
        wc = result['weight_norms']
        print(f"\nWeight Changes:")
        print(f"  L0 (nonzero): {wc['l0']:.0f}")
        print(f"  L1: {wc['l1']:.6f}")
        print(f"  L2: {wc['l2']:.6f}")
        print(f"  L∞: {wc['linf']:.6f}")
        
        # Behavioral results
        bc = result['behavioral_results']
        print(f"\nBehavioral Changes:")
        print(f"  ALTER success: {bc['alter_success_count']}/{bc['alter_total_count']} ({bc['alter_success_rate']:.1%})")
        print(f"  PRESERVE violations: {bc['preserve_violation_count']}/{bc['preserve_total_count']} ({bc['preserve_violation_rate']:.1%})")
        
        print(f"{'='*70}")
    
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


def run_enhanced_optimization(agent_path: str = None, config: dict = None):
    """
    Run optimization with enhanced logging that saves to agent's directory.
    Creates both detailed and summary logs in timestamped subdirectory.
    """
    if agent_path is None:
        agent_path = "../../Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6/"
    
    if config is None:
        config = OPTIMIZATION_CONFIG.copy()
    
    # Initialize enhanced logger
    logger = EnhancedOptimizationLogger(agent_path, config)
    
    # Load data
    alter_states, preserve_states = load_states_from_tooling(agent_path, 3, 5)
    model = load_model_from_agent_path(agent_path)
    target_actions = load_target_actions_from_tooling(alter_states)
    
    states_info = {
        'alter_count': len(alter_states),
        'preserve_count': len(preserve_states)
    }
    
    # Setup optimization components
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
    
    objective = MarginLossObjective(
        model=model,
        weight_selector=weight_selector,
        alter_states=alter_states,
        target_actions=target_actions,
        margin=config.get('margin')
    )
    
    # Compute initial objective
    initial_objective = objective.compute_objective(np.zeros_like(initial_weights), mapping_info)
    
    # Log initial state with comprehensive information
    logger.log_initial_state(states_info, selected_neurons, initial_objective, 
                           alter_states, preserve_states, target_actions, model, mapping_info)
    
    # Track optimization with detailed logging
    iteration_count = 0
    best_objective = initial_objective
    best_perturbations = np.zeros_like(initial_weights)
    
    def logged_objective(x):
        nonlocal iteration_count, best_objective, best_perturbations
        iteration_count += 1
        obj_val = objective.compute_objective(x, mapping_info)
        
        if obj_val < best_objective:
            best_objective = obj_val
            best_perturbations = x.copy()
        
        # Log iteration with detailed information
        logger.log_iteration(iteration_count, obj_val, x, alter_states, preserve_states, 
                           target_actions, model, weight_selector, mapping_info)
        
        return obj_val
    
    # Run optimization
    result = minimize(
        logged_objective,
        np.zeros_like(initial_weights),
        method='Nelder-Mead',
        options={
            'maxiter': config.get('max_iterations', 200),
            'disp': False
        }
    )
    
    # Log final results with comprehensive analysis
    summary_log = logger.log_final_results(
        result, initial_objective, best_perturbations,
        alter_states, preserve_states, target_actions,
        model, weight_selector, mapping_info
    )
    
    return summary_log


if __name__ == "__main__":
    summary = run_enhanced_optimization()
    print(f"\n🎯 Optimization complete! Check the agent's optimisation_results directory for detailed logs.") 