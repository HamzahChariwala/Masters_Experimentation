"""
Default configuration for gradient-based optimization.

This centralizes all parameters that were previously hardcoded in various functions.
"""

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