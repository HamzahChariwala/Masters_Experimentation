#!/usr/bin/env python3
"""
Model Perturbation Verification Tool
Compares original model weights with perturbed model to verify changes were applied correctly.
"""
import sys
import numpy as np
import torch
from pathlib import Path

# Add the correct paths for imports
sys.path.insert(0, '.')
sys.path.insert(0, 'Optimisation_Formulation/GradBasedTooling')

from Optimisation_Formulation.GradBasedTooling.utils.integration_utils import load_model_from_agent_path
from Optimisation_Formulation.GradBasedTooling.utils.weight_selection import WeightSelector


def verify_model_perturbations(original_model_path, perturbed_model_path, optimization_results_path=None):
    """
    Verify that perturbations were correctly applied to the saved model.
    
    Args:
        original_model_path: Path to the original agent
        perturbed_model_path: Path to the perturbed agent 
        optimization_results_path: Optional path to optimization summary for comparison
    """
    
    print("🔍 VERIFYING MODEL PERTURBATIONS")
    print("="*70)
    
    # Load both models
    print(f"📂 Loading original model: {original_model_path}")
    original_model = load_model_from_agent_path(str(original_model_path))
    
    print(f"📂 Loading perturbed model: {perturbed_model_path}")
    try:
        perturbed_model = load_model_from_agent_path(str(perturbed_model_path))
    except Exception as e:
        print(f"❌ FAILED: Could not load perturbed model: {e}")
        return False
    
    # Compare all parameters
    print(f"\n🔎 Comparing model parameters...")
    
    total_params = 0
    changed_params = 0
    max_change = 0
    max_change_param = None
    all_changes = []
    
    for (orig_name, orig_param), (pert_name, pert_param) in zip(
        original_model.named_parameters(), perturbed_model.named_parameters()
    ):
        if orig_name != pert_name:
            print(f"❌ MISMATCH: Parameter names don't match: {orig_name} vs {pert_name}")
            return False
            
        # Calculate differences
        diff = (pert_param.data - orig_param.data).cpu().numpy()
        abs_diff = np.abs(diff)
        
        total_params += diff.size
        changed_count = np.sum(abs_diff > 1e-8)  # Count non-zero changes
        changed_params += changed_count
        
        # Track largest change
        param_max_change = np.max(abs_diff)
        if param_max_change > max_change:
            max_change = param_max_change
            max_change_param = orig_name
            
        # Store changes for analysis
        if changed_count > 0:
            all_changes.extend(diff.flatten())
            print(f"  {orig_name}: {changed_count}/{diff.size} weights changed, max change: {param_max_change:.6f}")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Changed parameters: {changed_params:,} ({changed_params/total_params*100:.2f}%)")
    print(f"  Largest absolute change: {max_change:.6f} in {max_change_param}")
    
    if len(all_changes) > 0:
        all_changes = np.array(all_changes)
        print(f"  L1 norm of changes: {np.sum(np.abs(all_changes)):.6f}")
        print(f"  L2 norm of changes: {np.sqrt(np.sum(all_changes**2)):.6f}")
        print(f"  Mean change: {np.mean(all_changes):.6f}")
        print(f"  Std change: {np.std(all_changes):.6f}")
    
    # Load optimization results if provided
    if optimization_results_path and Path(optimization_results_path).exists():
        print(f"\n📋 Comparing with optimization results...")
        import json
        with open(optimization_results_path, 'r') as f:
            results = json.load(f)
        
        if 'SUMMARY' in results:
            summary = results['SUMMARY']
            expected_l1 = summary.get('weight_norms', {}).get('L1', 0)
            expected_l2 = summary.get('weight_norms', {}).get('L2', 0)
            expected_linf = summary.get('weight_norms', {}).get('L_infinity', 0)
            expected_l0 = summary.get('weight_norms', {}).get('L0', 0)
            
            print(f"  Expected L1: {expected_l1:.6f}, Actual: {np.sum(np.abs(all_changes)):.6f}")
            print(f"  Expected L2: {expected_l2:.6f}, Actual: {np.sqrt(np.sum(all_changes**2)):.6f}")
            print(f"  Expected L∞: {expected_linf:.6f}, Actual: {max_change:.6f}")
            print(f"  Expected L0: {expected_l0}, Actual: {changed_params}")
            
            # Check if they match (within tolerance)
            l1_match = abs(expected_l1 - np.sum(np.abs(all_changes))) < 1e-4
            l2_match = abs(expected_l2 - np.sqrt(np.sum(all_changes**2))) < 1e-4
            linf_match = abs(expected_linf - max_change) < 1e-4
            l0_match = expected_l0 == changed_params
            
            if l1_match and l2_match and linf_match and l0_match:
                print(f"  ✅ VERIFICATION PASSED: All norms match optimization results!")
                return True
            else:
                print(f"  ❌ VERIFICATION FAILED: Norms don't match optimization results")
                print(f"     L1 match: {l1_match}, L2 match: {l2_match}, L∞ match: {linf_match}, L0 match: {l0_match}")
                return False
    
    # If no optimization results to compare against, just check if there are changes
    if changed_params > 0:
        print(f"  ✅ VERIFICATION PASSED: Model contains {changed_params} parameter changes")
        return True
    else:
        print(f"  ⚠️  WARNING: No parameter changes detected")
        return False


def main():
    """Run verification on a specific optimization result"""
    
    if len(sys.argv) < 2:
        print("Usage: python verify_model_perturbations.py <optimization_results_dir>")
        print("Example: python verify_model_perturbations.py Agent_Storage/.../optimisation_250602_135657")
        return
    
    results_dir = Path(sys.argv[1])
    
    # Determine original agent path (parent directories)
    original_agent_path = results_dir.parent.parent
    perturbed_model_path = results_dir / "agent.zip"
    summary_path = results_dir / f"summary_{results_dir.name.split('_')[-1]}.json"
    
    print(f"Original agent: {original_agent_path}")
    print(f"Perturbed model: {perturbed_model_path}")
    print(f"Summary file: {summary_path}")
    
    success = verify_model_perturbations(
        original_agent_path, 
        perturbed_model_path,
        summary_path if summary_path.exists() else None
    )
    
    if success:
        print(f"\n🎉 SUCCESS: Model perturbations verified!")
    else:
        print(f"\n💥 FAILURE: Model perturbations verification failed!")


if __name__ == "__main__":
    main() 