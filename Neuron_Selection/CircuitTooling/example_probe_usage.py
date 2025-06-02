#!/usr/bin/env python3
"""
Example script: example_probe_usage.py

Demonstrates how to load and use trained linear probes for analyzing activations.
"""
import sys
import os
from pathlib import Path

# Import the linear probes module
from linear_probes import load_trained_probe, predict_with_probe, analyze_layer_importance, load_activations


def example_probe_analysis(agent_path):
    """
    Example of how to use trained probes to analyze activation data.
    
    Args:
        agent_path: Path to agent directory with trained probes
    """
    agent_path = Path(agent_path)
    probe_dir = agent_path / "activation_probing"
    
    if not probe_dir.exists():
        print(f"No probe directory found at {probe_dir}")
        print("Run linear_probes.py first to train probes.")
        return
    
    print(f"Analyzing probes for agent: {agent_path.name}")
    print("="*60)
    
    # 1. Analyze layer importance
    print("\n1. Layer Performance Analysis:")
    try:
        layer_analysis = analyze_layer_importance(probe_dir)
    except Exception as e:
        print(f"Error analyzing layers: {e}")
        return
    
    # 2. Load a specific probe and test it
    print("\n2. Testing Individual Probe Predictions:")
    try:
        # Load the first available probe
        available_layers = ['q_net.0', 'q_net.2']
        
        for layer_name in available_layers:
            try:
                model, scaler, metrics = load_trained_probe(probe_dir, layer_name)
                print(f"\nLoaded probe for {layer_name}")
                print(f"Training accuracy: {metrics['train_accuracy']:.4f}")
                print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
                
                # Load some actual activation data to test with
                clean_activations_file = agent_path / "activation_logging" / "clean_activations_readable.json"
                if clean_activations_file.exists():
                    clean_activations = load_activations(clean_activations_file)
                    
                    # Get a sample activation
                    sample_id = next(iter(clean_activations.keys()))
                    if layer_name in clean_activations[sample_id]:
                        sample_activation = clean_activations[sample_id][layer_name]
                        if isinstance(sample_activation, list) and len(sample_activation) > 0:
                            if isinstance(sample_activation[0], list):
                                sample_activation = sample_activation[0]
                            
                            # Test prediction
                            prediction, confidence = predict_with_probe(model, scaler, sample_activation)
                            result = "Clean" if prediction == 0 else "Corrupt"
                            print(f"Sample prediction: {result} (confidence: {confidence:.4f})")
                
            except Exception as e:
                print(f"Error with layer {layer_name}: {e}")
                continue
    
    except Exception as e:
        print(f"Error testing probes: {e}")
    
    print("\n" + "="*60)
    print("Analysis complete!")


def main():
    """Main function for example usage."""
    if len(sys.argv) != 2:
        print("Usage: python example_probe_usage.py <agent_path>")
        print("Example: python example_probe_usage.py ../../Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6")
        sys.exit(1)
    
    agent_path = sys.argv[1]
    example_probe_analysis(agent_path)


if __name__ == "__main__":
    main() 