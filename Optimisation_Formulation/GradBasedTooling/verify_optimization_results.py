#!/usr/bin/env python3
"""
Optimization Results Verification Tool
Verifies that optimization found meaningful weight changes and shows what the largest changes were.
"""
import sys
import numpy as np
import torch
import json
from pathlib import Path

# Add the correct paths for imports
sys.path.insert(0, '.')
sys.path.insert(0, 'Optimisation_Formulation/GradBasedTooling')

from Optimisation_Formulation.GradBasedTooling.utils.integration_utils import load_model_from_agent_path
from Optimisation_Formulation.GradBasedTooling.utils.weight_selection import WeightSelector


def verify_optimization_results(optimization_results_dir):
    """
    Verify optimization results using saved perturbation data.
    
    Args:
        optimization_results_dir: Path to the optimization results directory
    """
    
    print("🔍 VERIFYING OPTIMIZATION RESULTS")
    print("="*70)
    
    results_dir = Path(optimization_results_dir)
    
    # Load perturbation data
    perturbation_file = results_dir / "perturbations.json"
    if not perturbation_file.exists():
        print(f"❌ FAILED: No perturbation data found at {perturbation_file}")
        return False
    
    print(f"📂 Loading perturbation data: {perturbation_file}")
    with open(perturbation_file, 'r') as f:
        perturbation_data = json.load(f)
    
    final_perturbations = np.array(perturbation_data['final_perturbations'])
    mapping_info = perturbation_data['mapping_info']
    selected_neurons = perturbation_data['selected_neurons']
    optimization_stats = perturbation_data['optimization_stats']
    
    print(f"\n📊 OPTIMIZATION SUMMARY:")
    print(f"  Total perturbations: {len(final_perturbations)}")
    print(f"  Non-zero changes: {optimization_stats['L0']}")
    print(f"  L1 norm: {optimization_stats['L1']:.6f}")
    print(f"  L2 norm: {optimization_stats['L2']:.6f}")
    print(f"  L∞ norm: {optimization_stats['L_inf']:.6f}")
    print(f"  Largest change: {optimization_stats['largest_change']:.6f}")
    print(f"  Largest change index: {optimization_stats['largest_change_index']}")
    
    # Load original model
    original_agent_path = results_dir.parent.parent
    print(f"\n📂 Loading original model: {original_agent_path}")
    original_model = load_model_from_agent_path(str(original_agent_path))
    
    # Analyze the largest weight changes
    print(f"\n🔎 ANALYZING LARGEST WEIGHT CHANGES:")
    
    if optimization_stats['L0'] > 0:
        # Find indices of largest changes
        abs_perturbations = np.abs(final_perturbations)
        sorted_indices = np.argsort(abs_perturbations)[::-1]  # Descending order
        
        # Show top 5 changes
        print(f"  Top weight changes:")
        for i, idx in enumerate(sorted_indices[:5]):
            if abs_perturbations[idx] > 1e-8:  # Only show non-zero changes
                print(f"    {i+1}. Index {idx}: {final_perturbations[idx]:+.6f}")
            else:
                break
        
        # Map the largest change back to layer and neuron
        largest_idx = optimization_stats['largest_change_index']
        print(f"\n🎯 MAPPING LARGEST CHANGE (index {largest_idx}):")
        
        # Find which layer this index belongs to
        current_idx = 0
        found_layer = None
        layer_local_idx = None
        
        for layer_name, (start_idx, end_idx) in mapping_info['layer_slices'].items():
            if start_idx <= largest_idx < end_idx:
                found_layer = layer_name
                layer_local_idx = largest_idx - start_idx
                break
        
        if found_layer:
            print(f"  Layer: {found_layer}")
            print(f"  Local index in layer: {layer_local_idx}")
            
            # Find which neuron this corresponds to
            neuron_specs_for_layer = [(ln, ni) for ln, ni in mapping_info['neuron_specs'] if ln == found_layer]
            print(f"  Neurons in this layer: {neuron_specs_for_layer}")
            
            # Get the actual weight value from the model
            weight_selector = WeightSelector()
            weight_indices_dict = weight_selector.neurons_to_weight_indices(original_model, neuron_specs_for_layer)
            weight_indices = weight_indices_dict.get(found_layer, [])
            
            if layer_local_idx < len(weight_indices):
                actual_weight_idx = weight_indices[layer_local_idx]
                
                # Get the layer module
                layer_module = None
                for name, module in original_model.named_modules():
                    if name == found_layer:
                        layer_module = module
                        break
                
                if layer_module and hasattr(layer_module, 'weight'):
                    flattened_weights = layer_module.weight.data.flatten()
                    original_value = flattened_weights[actual_weight_idx].item()
                    proposed_new_value = original_value + optimization_stats['largest_change']
                    
                    print(f"  Original weight value: {original_value:.6f}")
                    print(f"  Proposed new value: {proposed_new_value:.6f}")
                    print(f"  Change: {optimization_stats['largest_change']:+.6f}")
        
        print(f"\n✅ VERIFICATION PASSED: Optimization found {optimization_stats['L0']} meaningful weight changes")
        print(f"   The largest change of {optimization_stats['largest_change']:.6f} was found at optimization index {largest_idx}")
        
        return True
    else:
        print(f"  ⚠️  WARNING: No weight changes found in optimization")
        return False


def simulate_perturbed_model(optimization_results_dir):
    """
    Create a temporary perturbed model in memory to verify the changes.
    """
    
    print(f"\n🧪 SIMULATING PERTURBED MODEL:")
    
    results_dir = Path(optimization_results_dir)
    
    # Load perturbation data
    perturbation_file = results_dir / "perturbations.json"
    with open(perturbation_file, 'r') as f:
        perturbation_data = json.load(f)
    
    final_perturbations = np.array(perturbation_data['final_perturbations'])
    mapping_info = perturbation_data['mapping_info']
    optimization_stats = perturbation_data['optimization_stats']
    
    # Load original model
    original_agent_path = results_dir.parent.parent
    model = load_model_from_agent_path(str(original_agent_path))
    
    print(f"  Applying perturbations to temporary model...")
    
    # Apply perturbations manually for verification
    changes_applied = 0
    for layer_name, (start_idx, end_idx) in mapping_info['layer_slices'].items():
        layer_perturbations = final_perturbations[start_idx:end_idx]
        
        # Get the layer module
        layer_module = None
        for name, module in model.named_modules():
            if name == layer_name:
                layer_module = module
                break
        
        if layer_module is None or not hasattr(layer_module, 'weight'):
            continue
        
        # Get weight indices for this layer
        neuron_specs_for_layer = [(ln, ni) for ln, ni in mapping_info['neuron_specs'] if ln == layer_name]
        weight_selector = WeightSelector()
        weight_indices_dict = weight_selector.neurons_to_weight_indices(model, neuron_specs_for_layer)
        weight_indices = weight_indices_dict.get(layer_name, [])
        
        # Apply perturbations
        with torch.no_grad():
            flattened_weights = layer_module.weight.data.flatten()
            for i, perturbation in enumerate(layer_perturbations):
                if abs(perturbation) > 1e-8 and i < len(weight_indices):
                    flattened_weights[weight_indices[i]] += perturbation
                    changes_applied += 1
            layer_module.weight.data = flattened_weights.reshape(layer_module.weight.shape)
    
    print(f"  ✅ Applied {changes_applied} perturbations to temporary model")
    print(f"  📊 Model now contains the optimized weight changes")
    print(f"  Note: This is only a temporary simulation - the saved model file contains original weights")
    
    return model


def main():
    """Run verification on a specific optimization result"""
    
    if len(sys.argv) < 2:
        print("Usage: python verify_optimization_results.py <optimization_results_dir>")
        print("Example: python verify_optimization_results.py Agent_Storage/.../optimisation_250602_135657")
        return
    
    results_dir = sys.argv[1]
    
    success = verify_optimization_results(results_dir)
    
    if success:
        # Also simulate the perturbed model
        simulate_perturbed_model(results_dir)
        print(f"\n🎉 SUCCESS: Optimization results verified!")
        print(f"  The optimization found meaningful weight changes.")
        print(f"  Use the saved perturbation data to apply changes when needed.")
    else:
        print(f"\n💥 FAILURE: Optimization results verification failed!")


if __name__ == "__main__":
    main() 