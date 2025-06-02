# Configuration Parameters Reference

This document lists ALL configurable hyperparameters available in the optimization system.

## Core Optimization Parameters

- `num_neurons` (int): Number of neurons to optimize (default: 8)
- `weights_per_neuron` (int): Number of weights per neuron (default: 64)
- `margin` (float): Margin threshold for loss computation (default: 0.05)
- `max_iterations` (int): Maximum optimization iterations (default: 100)
- `seed` (int): Random seed for reproducibility (default: 42)

## Regularization Parameters (Lambda Values)

- `lambda_sparse` (float): L1 regularization coefficient (default: 0.0)
- `lambda_magnitude` (float): L2 regularization coefficient (default: 0.0)

## State Sampling Parameters

- `alter_samples` (int): Number of ALTER states to sample (default: 5)
- `preserve_samples` (int): Number of PRESERVE states to sample (default: 10)

## Neuron Selection Parameters

- `neuron_selection.method` (str): Selection method ("random", "metric", "specific", "layer_balanced")
- `neuron_selection.metric` (str): Circuit verification metric name (for method="metric")
- `neuron_selection.specific_neurons` (list): Explicit neuron list (for method="specific")
- `neuron_selection.layer_distribution` (dict): Neurons per layer (for method="layer_balanced")

## Optimizer Parameters

- `optimizer_method` (str): Optimization method ("Powell", "BFGS", "L-BFGS-B", "CG", "Nelder-Mead")
- `gradient_tolerance` (float): Convergence tolerance for gradient-based methods (default: 1e-6)

## Bounds Parameters

- `bounds` (null/tuple/dict): Weight change bounds (default: null for no bounds)
- `adaptive_bounds_factor` (float): Factor for adaptive bounds (default: 2.0)
- `adaptive_bounds_min` (float): Minimum bound magnitude (default: 0.01)
- `adaptive_bounds_max` (float): Maximum bound magnitude (default: 1.0)

## Preserve Constraints Parameters

- `preserve_constraints.enabled` (bool): Enable preserve constraints (default: true)
- `preserve_constraints.type` (str): Constraint type ("soft" or "hard")
- `preserve_constraints.penalty_type` (str): Penalty computation method ("relative_margin")
- `preserve_constraints.penalty_weight` (float): Lambda for preserve penalty (default: 1.0)
- `preserve_constraints.margin_threshold` (float): Margin preservation threshold (default: 0.9)

## Other Parameters

- `target_layers` (list): Model layers to consider for optimization
- `save_perturbed_model` (bool): Whether to save optimized model (default: true)
- `weight_analysis_threshold` (float): Threshold for weight change analysis (default: 100.0)
- `enforce_preserve_constraints` (bool): Strict preserve constraint enforcement (default: true)
- `constraint_tolerance` (float): Tolerance for constraint satisfaction (default: 0.0)

## Example Configurations

### Basic Configuration
```json
{
  "num_neurons": 8,
  "margin": 0.05,
  "lambda_sparse": 0.0,
  "lambda_magnitude": 0.0,
  "max_iterations": 100
}
```

### With L1 Regularization
```json
{
  "num_neurons": 6,
  "margin": 0.05,
  "lambda_sparse": 0.01,
  "lambda_magnitude": 0.0,
  "max_iterations": 100
}
```

### Metric-Based Selection
```json
{
  "neuron_selection": {
    "method": "metric",
    "metric": "kl_divergence"
  },
  "num_neurons": 6,
  "lambda_sparse": 0.005,
  "lambda_magnitude": 0.001
}
``` 