#!/usr/bin/env python3
"""
Hyperparameter Sweep Leaderboard Analysis
Creates a comprehensive ranking of all tested configurations by ALTER states success rate.
Shows key performance metrics and hyperparameters for each configuration.
"""
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import time
import itertools

# Add the correct paths for imports
sys.path.insert(0, '.')
sys.path.insert(0, 'Optimisation_Formulation/GradBasedTooling')

from Optimisation_Formulation.GradBasedTooling.enhanced_optimization import run_enhanced_optimization
from Optimisation_Formulation.GradBasedTooling.configs.default_config import OPTIMIZATION_CONFIG


def create_hyperparameter_leaderboard(agent_path: str = None):
    """
    Run hyperparameter sweep and create comprehensive leaderboard ranked by ALTER success rate.
    Save results to agent's optimization_results directory.
    """
    
    if agent_path is None:
        agent_path = "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6/"
    
    # Create results directory
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    results_dir = Path(agent_path) / "optimisation_results" / f"hyperparameter_sweep_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("🏆 HYPERPARAMETER SWEEP LEADERBOARD ANALYSIS")
    print("="*100)
    print("Ranking all configurations by ALTER states success rate")
    print(f"Results will be saved to: {results_dir}")
    print("="*100)
    
    # Define parameter ranges to test
    lambda_sparse_values = [0.0, 0.001, 0.01, 0.1, 1.0]
    lambda_magnitude_values = [0.0, 0.001, 0.01, 0.1, 1.0]
    neuron_counts = [8, 12, 16]
    margin_values = [0.05, 0.1, 0.2]
    
    # Generate all combinations
    all_combinations = list(itertools.product(
        lambda_sparse_values,
        lambda_magnitude_values, 
        neuron_counts,
        margin_values
    ))
    
    print(f"Testing {len(all_combinations)} parameter combinations...")
    
    results = []
    test_count = 0
    
    for lambda_sparse, lambda_magnitude, num_neurons, margin in all_combinations:
        test_count += 1
        
        if test_count % 25 == 0:
            print(f"Progress: {test_count}/{len(all_combinations)} ({test_count/len(all_combinations)*100:.1f}%)")
        
        # Create test configuration
        config = OPTIMIZATION_CONFIG.copy()
        config.update({
            'optimizer_method': 'Powell',
            'lambda_sparse': lambda_sparse,
            'lambda_magnitude': lambda_magnitude,
            'num_neurons': num_neurons,
            'margin': margin,
            'max_iterations': 100,
            'bounds': None,
            'weight_analysis_threshold': 0.001,
            'seed': 42
        })
        
        start_time = time.time()
        
        try:
            summary = run_enhanced_optimization(agent_path=agent_path, config=config)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Extract comprehensive metrics
            if summary and 'SUMMARY' in summary:
                summary_data = summary['SUMMARY']
                
                result = {
                    # Hyperparameters
                    'lambda_sparse': lambda_sparse,
                    'lambda_magnitude': lambda_magnitude,
                    'num_neurons': num_neurons,
                    'margin': margin,
                    
                    # Performance metrics
                    'alter_success_rate': summary_data.get('behavioral_success', {}).get('alter_states_success_rate', 0),
                    'preserve_violation_rate': summary_data.get('behavioral_success', {}).get('preserve_states_violation_rate', 0),
                    'alter_states_changed': summary_data.get('behavioral_success', {}).get('alter_states_changed', '0/0'),
                    'preserve_states_violated': summary_data.get('behavioral_success', {}).get('preserve_states_violated', '0/0'),
                    
                    # Objective metrics
                    'objective_initial': summary_data.get('objective_change', {}).get('initial', 0),
                    'objective_final': summary_data.get('objective_change', {}).get('final', 0),
                    'objective_improvement': summary_data.get('objective_change', {}).get('improvement', 0),
                    
                    # Weight change metrics (L-norms)
                    'l0_norm': summary_data.get('weight_norms', {}).get('L0', 0),
                    'l1_norm': summary_data.get('weight_norms', {}).get('L1', 0),
                    'l2_norm': summary_data.get('weight_norms', {}).get('L2', 0),
                    'linf_norm': summary_data.get('weight_norms', {}).get('L_infinity', 0),
                    
                    # Optimization stats
                    'iterations': summary_data.get('optimization_stats', {}).get('total_iterations', 0),
                    'function_evals': summary_data.get('optimization_stats', {}).get('function_evaluations', 0),
                    'execution_time': execution_time,
                    'success': summary_data.get('success', False)
                }
                
                # Calculate regularization penalties
                l1_penalty = lambda_sparse * result['l1_norm']
                l2_penalty = lambda_magnitude * (result['l2_norm'] ** 2)
                result['l1_penalty'] = l1_penalty
                result['l2_penalty'] = l2_penalty
                result['total_regularization'] = l1_penalty + l2_penalty
                
                # Calculate efficiency metrics
                result['efficiency'] = result['objective_improvement'] / max(result['execution_time'], 0.001)
                result['sparsity_ratio'] = result['l0_norm'] / max(result['num_neurons'] * 64, 1)
                
                results.append(result)
                
        except Exception as e:
            print(f"❌ Test {test_count} failed: {e}")
            continue
    
    if not results:
        print("❌ No results to analyze")
        return []
    
    # Sort by ALTER success rate (primary), then by objective improvement (secondary)
    leaderboard = sorted(results, key=lambda x: (x['alter_success_rate'], x['objective_improvement']), reverse=True)
    
    # Save detailed results as JSON
    detailed_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'agent_path': agent_path,
            'total_configurations': len(results),
            'test_parameters': {
                'lambda_sparse_values': lambda_sparse_values,
                'lambda_magnitude_values': lambda_magnitude_values,
                'neuron_counts': neuron_counts,
                'margin_values': margin_values
            }
        },
        'leaderboard': leaderboard,
        'statistics': {
            'perfect_preserve': len([r for r in results if r['preserve_violation_rate'] == 0.0]),
            'meaningful_change': len([r for r in results if r['linf_norm'] > 0.001]),
            'small_perturbations': len([r for r in results if r['linf_norm'] <= 0.1]),
            'any_alter_success': len([r for r in results if r['alter_success_rate'] > 0.0])
        }
    }
    
    # Save JSON
    json_path = results_dir / f"hyperparameter_leaderboard_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Save CSV for easy analysis
    df = pd.DataFrame(leaderboard)
    csv_path = results_dir / f"hyperparameter_leaderboard_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Create comprehensive leaderboard display
    print(f"\n{'='*100}")
    print("HYPERPARAMETER LEADERBOARD - RANKED BY ALTER SUCCESS RATE")
    print(f"{'='*100}")
    
    # Print header
    print(f"{'Rank':<4} {'ALTER':<6} {'PRESERVE':<8} {'Obj Δ':<10} {'L0':<4} {'L1':<8} {'L2':<8} {'L∞':<8} "
          f"{'λ_spr':<6} {'λ_mag':<6} {'N':<3} {'Mrg':<5} {'Iter':<4} {'Time':<5}")
    print("-" * 100)
    
    # Print top results
    for i, result in enumerate(leaderboard[:20]):  # Top 20 for console
        rank = i + 1
        alter_pct = f"{result['alter_success_rate']:.1%}"
        preserve_pct = f"{result['preserve_violation_rate']:.1%}"
        obj_delta = f"{result['objective_improvement']:.6f}"
        l0 = f"{result['l0_norm']:.0f}"
        l1 = f"{result['l1_norm']:.4f}"
        l2 = f"{result['l2_norm']:.4f}"
        linf = f"{result['linf_norm']:.4f}"
        lambda_s = f"{result['lambda_sparse']}"
        lambda_m = f"{result['lambda_magnitude']}"
        neurons = f"{result['num_neurons']}"
        margin = f"{result['margin']}"
        iters = f"{result['iterations']}"
        time_str = f"{result['execution_time']:.1f}s"
        
        print(f"{rank:<4} {alter_pct:<6} {preserve_pct:<8} {obj_delta:<10} {l0:<4} {l1:<8} {l2:<8} {linf:<8} "
              f"{lambda_s:<6} {lambda_m:<6} {neurons:<3} {margin:<5} {iters:<4} {time_str:<5}")
    
    # Final recommendations
    if leaderboard:
        best_config = leaderboard[0]
        print(f"\n🚀 RECOMMENDED PARAMETERS:")
        print(f"   λ_sparse={best_config['lambda_sparse']}, λ_magnitude={best_config['lambda_magnitude']}")
        print(f"   neurons={best_config['num_neurons']}, margin={best_config['margin']}")
        print(f"   Expected: {best_config['alter_success_rate']:.1%} ALTER success, {best_config['preserve_violation_rate']:.1%} preserve violations")
    
    print(f"\n📁 Results saved to:")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")
    
    return leaderboard


if __name__ == "__main__":
    leaderboard = create_hyperparameter_leaderboard()
    
    if leaderboard:
        print(f"\n✅ Leaderboard analysis complete!")
        print(f"   Total configurations tested: {len(leaderboard)}")
        print(f"   Best ALTER success rate: {leaderboard[0]['alter_success_rate']:.1%}")
    else:
        print(f"\n❌ No configurations to analyze") 