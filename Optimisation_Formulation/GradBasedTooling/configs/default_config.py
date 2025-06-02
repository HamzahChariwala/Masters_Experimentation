"""
Default configuration for gradient-based optimization.

This centralizes all parameters that were previously hardcoded in various functions.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Optimization parameters
OPTIMIZATION_CONFIG = {
    # Weight selection parameters  
    'num_neurons': 8,
    'target_layers': [
        'q_net.features_extractor.mlp.0',
        'q_net.q_net.0', 
        'q_net.q_net.2'
    ],
    'weights_per_neuron': 64,
    
    # Enhanced neuron selection parameters
    'neuron_selection': {
        'method': 'random',  # 'random', 'specific', 'layer_balanced'
        'specific_neurons': None,  # For method='specific': [(layer_name, neuron_idx), ...]
        'layer_distribution': None,  # For method='layer_balanced': {'layer_name': count, ...}
        'exclude_neurons': None,  # List of (layer_name, neuron_idx) to exclude
        'prioritize_layers': None,  # List of layer names to prioritize in selection
    },
    
    # State sampling parameters
    'alter_samples': 5,
    'preserve_samples': 10,
    
    # Objective function parameters
    'margin': 0.05,  # Easier target for optimization
    'lambda_sparse': 0.0,  # No L1 regularization to allow changes
    'lambda_magnitude': 0.0,  # No L2 regularization to allow changes
    
    # Optimization solver parameters
    'max_iterations': 100,
    'gradient_epsilon': 1e-4,
    'step_size': 0.01,
    'num_optimization_steps': 10,
    
    # Model saving parameters
    'save_perturbed_model': True,  # Whether to save the optimized model with perturbations applied
    
    # Bounds parameters (NEW)
    'bounds': None,  # No bounds for maximum optimization freedom
    'adaptive_bounds_factor': 2.0,  # Factor times the standard deviation of weights
    'adaptive_bounds_min': 0.01,    # Minimum bound magnitude
    'adaptive_bounds_max': 1.0,     # Maximum bound magnitude
    
    # Alternative bound configurations:
    # 'bounds': (-0.3, 0.3),  # Strict symmetric bounds
    # 'bounds': {'type': 'adaptive', 'factor': 0.15, 'min_bound': 1e-5, 'max_bound': 0.5},
    # 'bounds': {'type': 'strict', 'value': 0.2},
    
    # Preserve constraint parameters (NEW)
    'preserve_constraints': {
        'enabled': True,
        'type': 'soft',  # 'soft' or 'hard' (hard not implemented yet)
        'penalty_type': 'relative_margin',  # Currently supported: 'relative_margin'
        'penalty_weight': 1.0,  # Lambda coefficient for penalty term
        'margin_threshold': 0.9,  # Threshold relative to original margin (90%)
        # Future penalty types could be added:
        # 'penalty_type': 'logit_difference',  # For simple top logit difference
        # 'penalty_type': 'ranking_loss',      # For full ranking preservation
    },
    
    # Optimizer parameters (NEW)
    'optimizer_method': 'Powell',   # Superior performance from our testing; gradient-free method that works well with bounds
    'gradient_tolerance': 1e-6,          # Convergence tolerance for gradient-based methods
    
    # Constraint parameters
    'enforce_preserve_constraints': True,
    'constraint_tolerance': 0.0,  # Strict preserve constraints
    
    # Logging parameters
    'log_level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'log_gradient_details': False,
    'log_q_value_details': False,
    
    # Weight analysis parameters
    'weight_analysis_threshold': 100.0,  # Threshold for identifying large weight changes to analyze
    
    # Random seed
    'seed': 42
}

# Derived configurations for different use cases
QUICK_TEST_CONFIG = OPTIMIZATION_CONFIG.copy()
QUICK_TEST_CONFIG.update({
    'num_neurons': 2,
    'alter_samples': 2,
    'preserve_samples': 3,
    'max_iterations': 20,
    'num_optimization_steps': 5
})

COMPREHENSIVE_TEST_CONFIG = OPTIMIZATION_CONFIG.copy()
COMPREHENSIVE_TEST_CONFIG.update({
    'num_neurons': 12,
    'alter_samples': 10,
    'preserve_samples': 20,
    'max_iterations': 200,
    'num_optimization_steps': 20
})

# Configuration loading utilities
def load_config_from_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file {config_path}: {e}")
    
    return custom_config

def merge_configs(base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge custom configuration with base configuration.
    Custom config values override base config values.
    
    Args:
        base_config: Base configuration dictionary
        custom_config: Custom configuration to overlay
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in custom_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override with custom value
            merged[key] = value
    
    return merged

def create_config_from_file(config_path: Union[str, Path], 
                           base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a complete configuration by loading from file and merging with base config.
    
    Args:
        config_path: Path to the configuration JSON file
        base_config: Base configuration to merge with (defaults to OPTIMIZATION_CONFIG)
        
    Returns:
        Complete merged configuration dictionary
    """
    if base_config is None:
        base_config = OPTIMIZATION_CONFIG
    
    custom_config = load_config_from_file(config_path)
    return merge_configs(base_config, custom_config)

def save_config_template(output_path: Union[str, Path], 
                        config: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a configuration template to a JSON file for easy customization.
    
    Args:
        output_path: Where to save the configuration template
        config: Configuration to save (defaults to OPTIMIZATION_CONFIG)
    """
    if config is None:
        config = OPTIMIZATION_CONFIG
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration template saved to: {output_path}")

# Example usage configurations for common scenarios
EXPERIMENT_CONFIGS = {
    'few_neurons_focused': {
        'num_neurons': 3,
        'target_layers': ['q_net.q_net.0'],  # Focus on one layer
        'neuron_selection': {
            'method': 'layer_balanced',
            'layer_distribution': {'q_net.q_net.0': 3}
        }
    },
    
    'multi_layer_balanced': {
        'num_neurons': 9,
        'neuron_selection': {
            'method': 'layer_balanced',
            'layer_distribution': {
                'q_net.features_extractor.mlp.0': 3,
                'q_net.q_net.0': 3,
                'q_net.q_net.2': 3
            }
        }
    },
    
    'high_precision': {
        'max_iterations': 500,
        'gradient_tolerance': 1e-8,
        'margin': 0.01,  # Stricter margin
        'preserve_constraints': {
            'enabled': True,
            'penalty_weight': 5.0,  # Stronger preserve penalty
            'margin_threshold': 0.95
        }
    }
} 