#!/usr/bin/env python3
"""
Example script demonstrating various ways to use custom configurations 
with the enhanced gradient-based optimization system.
"""

import sys
from pathlib import Path

# Add the root directory to path
sys.path.insert(0, '../..')

# Import necessary modules
from enhanced_optimization import run_enhanced_optimization, run_optimization_with_experiment_config
from Optimisation_Formulation.GradBasedTooling.configs.default_config import (
    OPTIMIZATION_CONFIG, 
    EXPERIMENT_CONFIGS,
    create_config_from_file,
    save_config_template,
    merge_configs
)

def example_1_inline_config():
    """Example 1: Using inline custom configuration"""
    print("="*70)
    print("EXAMPLE 1: Inline Custom Configuration")
    print("="*70)
    
    custom_config = {
        'num_neurons': 6,
        'seed': 123,
        'max_iterations': 150,
        'margin': 0.02,
        'neuron_selection': {
            'method': 'layer_balanced',
            'layer_distribution': {
                'q_net.features_extractor.mlp.0': 2,
                'q_net.q_net.0': 2,
                'q_net.q_net.2': 2
            }
        }
    }
    
    print(f"Custom config: {custom_config}")
    # Uncomment to run:
    # summary = run_enhanced_optimization(config=custom_config)
    print("✅ Config prepared for inline usage\n")

def example_2_json_config_file():
    """Example 2: Using JSON configuration file"""
    print("="*70)
    print("EXAMPLE 2: JSON Configuration File")
    print("="*70)
    
    # Create a custom config and save it to file
    custom_config = {
        'num_neurons': 4,
        'seed': 456,
        'neuron_selection': {
            'method': 'random',
            'prioritize_layers': ['q_net.q_net.0', 'q_net.q_net.2'],
            'exclude_neurons': [('q_net.features_extractor.mlp.0', 0)]
        },
        'preserve_constraints': {
            'enabled': True,
            'penalty_weight': 2.0,
            'margin_threshold': 0.95
        }
    }
    
    # Save to file
    config_path = "custom_experiment_config.json"
    save_config_template(config_path, custom_config)
    
    print(f"Saved custom config to: {config_path}")
    
    # Load and use the config
    # Uncomment to run:
    # summary = run_enhanced_optimization(config_path=config_path)
    print("✅ Config prepared for file-based usage\n")

def example_3_experiment_configs():
    """Example 3: Using predefined experiment configurations"""
    print("="*70)
    print("EXAMPLE 3: Predefined Experiment Configurations")
    print("="*70)
    
    print("Available experiment configs:")
    for name, config in EXPERIMENT_CONFIGS.items():
        print(f"  - {name}: {config}")
    
    # Use predefined experiment config
    print(f"\nUsing 'few_neurons_focused' experiment config")
    # Uncomment to run:
    # summary = run_optimization_with_experiment_config(experiment_name='few_neurons_focused')
    
    # Use experiment config with overrides
    overrides = {'seed': 789, 'max_iterations': 250}
    print(f"Using 'multi_layer_balanced' with overrides: {overrides}")
    # Uncomment to run:
    # summary = run_optimization_with_experiment_config(
    #     experiment_name='multi_layer_balanced',
    #     custom_overrides=overrides
    # )
    print("✅ Experiment configs demonstrated\n")

def example_4_specific_neuron_selection():
    """Example 4: Selecting specific neurons"""
    print("="*70)
    print("EXAMPLE 4: Specific Neuron Selection")
    print("="*70)
    
    # Select specific neurons manually
    specific_config = {
        'neuron_selection': {
            'method': 'specific',
            'specific_neurons': [
                ('q_net.features_extractor.mlp.0', 5),
                ('q_net.features_extractor.mlp.0', 15),
                ('q_net.q_net.0', 3),
                ('q_net.q_net.0', 7),
                ('q_net.q_net.2', 1),
                ('q_net.q_net.2', 4)
            ]
        }
    }
    
    print(f"Specific neuron selection config: {specific_config}")
    # Uncomment to run:
    # summary = run_enhanced_optimization(config=specific_config)
    print("✅ Specific neuron selection configured\n")

def example_5_merge_and_modify():
    """Example 5: Merging and modifying existing configs"""
    print("="*70)
    print("EXAMPLE 5: Merging and Modifying Configs")
    print("="*70)
    
    # Start with a base config
    base_config = EXPERIMENT_CONFIGS['high_precision'].copy()
    print(f"Base config (high_precision): {base_config}")
    
    # Create modifications
    modifications = {
        'num_neurons': 15,
        'neuron_selection': {
            'method': 'layer_balanced',
            'layer_distribution': {
                'q_net.features_extractor.mlp.0': 5,
                'q_net.q_net.0': 5,
                'q_net.q_net.2': 5
            }
        },
        'seed': 999
    }
    
    # Merge them
    final_config = merge_configs(OPTIMIZATION_CONFIG, base_config)
    final_config = merge_configs(final_config, modifications)
    
    print(f"\nFinal merged config keys: {list(final_config.keys())}")
    print(f"Neuron selection: {final_config['neuron_selection']}")
    print(f"Preserve constraints: {final_config['preserve_constraints']}")
    
    # Uncomment to run:
    # summary = run_enhanced_optimization(config=final_config)
    print("✅ Config merging demonstrated\n")

def show_current_neuron_selection():
    """Show how the current system selects neurons"""
    print("="*70)
    print("CURRENT NEURON SELECTION METHODS")
    print("="*70)
    
    print("Your optimization system supports these neuron selection methods:")
    print("1. 'random': Randomly select neurons from all target layers")
    print("2. 'layer_balanced': Select specific numbers from each layer")
    print("3. 'specific': Use manually specified neuron coordinates")
    print("\nAdditional random selection options:")
    print("- 'exclude_neurons': Exclude specific neurons from selection")
    print("- 'prioritize_layers': Prefer certain layers in random selection")
    print()
    
    print("Configuration is controlled through the 'neuron_selection' dict:")
    example_configs = {
        'random_basic': {
            'method': 'random'
        },
        'random_with_exclusions': {
            'method': 'random',
            'exclude_neurons': [('q_net.features_extractor.mlp.0', 0)],
            'prioritize_layers': ['q_net.q_net.0']
        },
        'layer_balanced': {
            'method': 'layer_balanced',
            'layer_distribution': {
                'q_net.features_extractor.mlp.0': 3,
                'q_net.q_net.0': 3,
                'q_net.q_net.2': 2
            }
        },
        'specific': {
            'method': 'specific',
            'specific_neurons': [
                ('q_net.q_net.0', 5),
                ('q_net.q_net.2', 10)
            ]
        }
    }
    
    for name, config in example_configs.items():
        print(f"{name}: {config}")
    
    print("\n✅ All neuron selection methods explained")

if __name__ == "__main__":
    print("🔧 ENHANCED OPTIMIZATION CONFIGURATION EXAMPLES")
    print("This script demonstrates various ways to use custom configurations.")
    print("Uncomment the run_enhanced_optimization() calls to execute.\n")
    
    show_current_neuron_selection()
    example_1_inline_config()
    example_2_json_config_file()
    example_3_experiment_configs()
    example_4_specific_neuron_selection()
    example_5_merge_and_modify()
    
    print("="*70)
    print("🎯 SUMMARY: Configuration Options Available")
    print("="*70)
    print("1. Pass config dict directly to run_enhanced_optimization(config=...)")
    print("2. Load config from JSON file with run_enhanced_optimization(config_path=...)")
    print("3. Use predefined experiments with run_optimization_with_experiment_config()")
    print("4. Merge configs with merge_configs() for complex setups")
    print("5. Create config templates with save_config_template()")
    print("\nAll methods support the enhanced neuron selection options!")
    print("="*70) 