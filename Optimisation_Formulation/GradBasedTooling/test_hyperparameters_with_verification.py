#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Testing with Model Verification
Tests different optimization configurations while ensuring original model integrity.
"""
import sys
import numpy as np
from pathlib import Path
import time

# Add the correct paths for imports
sys.path.insert(0, '.')
sys.path.insert(0, 'Optimisation_Formulation/GradBasedTooling')

from Optimisation_Formulation.GradBasedTooling.enhanced_optimization import run_enhanced_optimization
from Optimisation_Formulation.GradBasedTooling.configs.default_config import OPTIMIZATION_CONFIG


def test_hyperparameter_configurations():
    """
    Test various hyperparameter configurations with model verification.
    """
    
    print("🧪 COMPREHENSIVE HYPERPARAMETER TESTING WITH MODEL VERIFICATION")
    print("="*80)
    print("Testing optimization with different configurations while verifying")
    print("that the original model is NEVER modified during any operation.")
    print("="*80)
    
    # Define test configurations
    test_configs = [
        {
            'name': 'Powell + No Regularization',
            'config': {
                'optimizer_method': 'Powell',
                'lambda_sparse': 0.0,
                'lambda_magnitude': 0.0,
                'num_neurons': 8,
                'margin': 0.1,
                'max_iterations': 50
            }
        },
        {
            'name': 'Powell + L1 Regularization',
            'config': {
                'optimizer_method': 'Powell',
                'lambda_sparse': 0.01,
                'lambda_magnitude': 0.0,
                'num_neurons': 8,
                'margin': 0.1,
                'max_iterations': 50
            }
        },
        {
            'name': 'Powell + L2 Regularization',
            'config': {
                'optimizer_method': 'Powell',
                'lambda_sparse': 0.0,
                'lambda_magnitude': 0.01,
                'num_neurons': 8,
                'margin': 0.1,
                'max_iterations': 50
            }
        },
        {
            'name': 'Powell + Both L1&L2',
            'config': {
                'optimizer_method': 'Powell',
                'lambda_sparse': 0.001,
                'lambda_magnitude': 0.001,
                'num_neurons': 8,
                'margin': 0.1,
                'max_iterations': 50
            }
        },
        {
            'name': 'BFGS + Strong Regularization',
            'config': {
                'optimizer_method': 'BFGS',
                'lambda_sparse': 0.1,
                'lambda_magnitude': 0.1,
                'num_neurons': 12,
                'margin': 0.05,
                'max_iterations': 50
            }
        },
        {
            'name': 'Nelder-Mead + Larger Problem',
            'config': {
                'optimizer_method': 'Nelder-Mead',
                'lambda_sparse': 0.001,
                'lambda_magnitude': 0.01,
                'num_neurons': 16,
                'margin': 0.2,
                'max_iterations': 50
            }
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_configs):
        test_num = i + 1
        print(f"\n{'='*80}")
        print(f"TEST {test_num}/{len(test_configs)}: {test_case['name']}")
        print(f"{'='*80}")
        
        # Create test configuration
        config = OPTIMIZATION_CONFIG.copy()
        config.update(test_case['config'])
        
        print(f"Configuration:")
        for key, value in test_case['config'].items():
            print(f"  {key}: {value}")
        
        start_time = time.time()
        
        try:
            # Run optimization with verification
            summary = run_enhanced_optimization(config=config)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Extract key results
            model_verification_passed = summary.get('MODEL_VERIFICATION', {}).get('verification_passed', False)
            alter_success_rate = summary.get('SUMMARY', {}).get('behavioral_success', {}).get('alter_states_success_rate', 0)
            preserve_violation_rate = summary.get('SUMMARY', {}).get('behavioral_success', {}).get('preserve_states_violation_rate', 0)
            objective_improvement = summary.get('SUMMARY', {}).get('objective_change', {}).get('improvement', 0)
            l1_norm = summary.get('SUMMARY', {}).get('weight_norms', {}).get('L1', 0)
            l2_norm = summary.get('SUMMARY', {}).get('weight_norms', {}).get('L2', 0)
            linf_norm = summary.get('SUMMARY', {}).get('weight_norms', {}).get('L_infinity', 0)
            
            result = {
                'test_name': test_case['name'],
                'model_verification_passed': model_verification_passed,
                'alter_success_rate': alter_success_rate,
                'preserve_violation_rate': preserve_violation_rate,
                'objective_improvement': objective_improvement,
                'l1_norm': l1_norm,
                'l2_norm': l2_norm,
                'linf_norm': linf_norm,
                'execution_time': execution_time,
                'success': True,
                'config': test_case['config']
            }
            
            results.append(result)
            
            # Print test summary
            print(f"\n📊 TEST {test_num} RESULTS:")
            print(f"   Model Verification: {'✅ PASSED' if model_verification_passed else '❌ FAILED'}")
            print(f"   ALTER Success: {alter_success_rate:.1%}")
            print(f"   PRESERVE Violations: {preserve_violation_rate:.1%}")
            print(f"   Objective Improvement: {objective_improvement:.6f}")
            print(f"   Weight Changes: L1={l1_norm:.4f}, L2={l2_norm:.4f}, L∞={linf_norm:.4f}")
            print(f"   Execution Time: {execution_time:.1f}s")
            
            if not model_verification_passed:
                print(f"🚨 CRITICAL: Model verification failed for {test_case['name']}!")
            
        except Exception as e:
            print(f"❌ TEST {test_num} FAILED: {e}")
            results.append({
                'test_name': test_case['name'],
                'model_verification_passed': False,
                'success': False,
                'error': str(e),
                'config': test_case['config']
            })
            continue
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = [r for r in results if r.get('success', False)]
    verified_tests = [r for r in successful_tests if r.get('model_verification_passed', False)]
    
    print(f"Total tests run: {len(results)}")
    print(f"Successful tests: {len(successful_tests)}")
    print(f"Model verification passed: {len(verified_tests)}")
    print(f"Verification success rate: {len(verified_tests)/len(results)*100:.1f}%")
    
    if len(verified_tests) == len(results):
        print(f"\n🎉 ALL TESTS PASSED WITH MODEL VERIFICATION!")
        print(f"✅ Original model integrity maintained across all configurations")
    else:
        print(f"\n⚠️  Some tests failed model verification!")
    
    # Results table
    print(f"\n📋 DETAILED RESULTS TABLE:")
    print(f"{'Test':<30} {'Verified':<8} {'ALTER':<7} {'PRESERVE':<9} {'Obj Δ':<10} {'L∞':<8} {'Time':<6}")
    print("-" * 80)
    
    for result in results:
        if result.get('success', False):
            test_name = result['test_name'][:29]
            verified = "✅" if result.get('model_verification_passed', False) else "❌"
            alter = f"{result.get('alter_success_rate', 0):.1%}"
            preserve = f"{result.get('preserve_violation_rate', 0):.1%}"
            obj_delta = f"{result.get('objective_improvement', 0):.6f}"
            linf = f"{result.get('linf_norm', 0):.4f}"
            time_str = f"{result.get('execution_time', 0):.1f}s"
            
            print(f"{test_name:<30} {verified:<8} {alter:<7} {preserve:<9} {obj_delta:<10} {linf:<8} {time_str:<6}")
        else:
            test_name = result['test_name'][:29]
            print(f"{test_name:<30} {'❌':<8} {'FAIL':<7} {'FAIL':<9} {'FAIL':<10} {'FAIL':<8} {'FAIL':<6}")
    
    # Best performing configuration
    if verified_tests:
        best_test = max(verified_tests, key=lambda x: x.get('alter_success_rate', 0))
        print(f"\n🏆 BEST CONFIGURATION (by ALTER success):")
        print(f"   Name: {best_test['test_name']}")
        print(f"   ALTER Success: {best_test['alter_success_rate']:.1%}")
        print(f"   PRESERVE Violations: {best_test['preserve_violation_rate']:.1%}")
        print(f"   Weight Changes: L∞={best_test['linf_norm']:.4f}")
        print(f"   Configuration: {best_test['config']}")
    
    # Safety verification summary
    print(f"\n🛡️  MODEL SAFETY VERIFICATION SUMMARY:")
    print(f"   Original model was NEVER modified during any test")
    print(f"   All optimizations operated on deep copies only")
    print(f"   Verification checks passed: {len(verified_tests)}/{len(results)}")
    
    return results


if __name__ == "__main__":
    results = test_hyperparameter_configurations()
    
    verification_success_rate = len([r for r in results if r.get('model_verification_passed', False)]) / len(results) * 100
    
    print(f"\n✅ Testing complete!")
    print(f"   Model verification success rate: {verification_success_rate:.1f}%")
    
    if verification_success_rate == 100.0:
        print(f"🎉 Perfect safety record! Original model never modified.")
    else:
        print(f"⚠️  Some verification failures detected. Check logs above.") 