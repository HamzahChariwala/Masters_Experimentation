# Gradient-Based Optimization Tooling

## Overview

This module provides gradient-based optimization capabilities for neural network weight perturbations within the existing DQN optimization framework. The current implementation focuses on margin-based optimization with neuron-level weight selection.

## Current Implementation Status

### ✅ Completed (Phase 1 Prototype)

- **Core Infrastructure**: Base classes for solvers and objectives
- **Weight Selection**: Neuron-level weight selection from first 3 layers (features, q_net.0, q_net.2)  
- **Margin-Based Objective**: Primary objective function with multi-target action support
- **Integration Utilities**: Load states from existing StateTooling results
- **Sparsity/Magnitude Regularization**: L1 and L2 penalty terms with configurable λ coefficients

### 🚧 In Progress 

- **Solver Integration**: Full SLSQP solver with constraints
- **Preserve Constraints**: Strict argmax preservation for PRESERVE states
- **Model Loading**: Integration with existing model loading infrastructure

### 📋 TODO

- Cross-entropy objective function
- Automatic solver selection
- Performance optimization for larger problems

## Usage

### Prerequisites

1. Run StateTooling to generate ALTER and PRESERVE state sets:
   ```bash
   python Optimisation_Formulation/StateTooling/optimization_state_filter.py \
     --path /path/to/agent \
     --alter "your_alter_criteria" \
     --preserve "your_preserve_criteria"
   ```

2. Add desired actions to ALTER states:
   ```bash
   python Optimisation_Formulation/StateTooling/add_desired_actions.py \
     --agent-path /path/to/agent
   ```

### Basic Example

```bash
python Optimisation_Formulation/GradBasedTooling/examples/basic_margin_optimization.py \
  --agent-path /path/to/agent \
  --neurons 2 \
  --alter-samples 3 \
  --preserve-samples 5 \
  --margin 0.1 \
  --lambda-sparse 0.0001 \
  --lambda-magnitude 0.001
```

### Programmatic Usage

```python
from Optimisation_Formulation.GradBasedTooling.utils.weight_selection import WeightSelector
from Optimisation_Formulation.GradBasedTooling.objectives.margin_loss_objective import MarginLossObjective
from Optimisation_Formulation.GradBasedTooling.utils.integration_utils import load_states_from_tooling

# Load states from StateTooling
alter_states, preserve_states = load_states_from_tooling(agent_path, alter_samples=5, preserve_samples=10)

# Set up weight selector
weight_selector = WeightSelector(target_layers=['features', 'q_net.0', 'q_net.2'])
selected_neurons = weight_selector.select_random_neurons(model, num_neurons=3)

# Create optimization variables  
initial_weights, mapping_info = weight_selector.create_optimization_vector(model, selected_neurons)

# Set up objective function
objective = MarginLossObjective(
    model=model,
    weight_selector=weight_selector,
    alter_states=alter_states,
    target_actions=target_actions,
    margin=0.1,
    lambda_sparse=0.0001,
    lambda_magnitude=0.001
)

# Compute objective and gradient
obj_value = objective.compute_objective(perturbations, mapping_info)
gradient = objective.compute_gradient(perturbations, mapping_info)
```

## Configuration Parameters

### Optimization Parameters

- **margin**: Margin for margin-based loss (default: 0.1)
- **lambda_sparse**: Sparsity regularization coefficient (default: 0.0001)  
- **lambda_magnitude**: Magnitude regularization coefficient (default: 0.001)

### Selection Parameters

- **neurons**: Number of neurons to randomly select for optimization
- **alter_samples**: Number of ALTER states to use in optimization
- **preserve_samples**: Number of PRESERVE states for constraints
- **seed**: Random seed for reproducible neuron selection

## Mathematical Formulation

### Objective Function

```
minimize f(Δw) = L_margin(Q_alter(w), targets) + λ_sparse * ||Δw||₁ + λ_magnitude * ||Δw||₂²
```

### Margin-Based Loss

For single target action:
```
L = max(0, max_a≠target(Q(s,a)) - Q(s,target) + margin)
```

For multiple acceptable actions:
```  
L = max(0, max_a∉acceptable(Q(s,a)) - max_a∈acceptable(Q(s,a)) + margin)
```

### Constraints

- **Bounds**: `lb ≤ Δw ≤ ub` (weight change limits)
- **Preserve**: `argmax(Q_preserve(s)) = argmax(Q_original(s))` (strict equality)

## Directory Structure

```
GradBasedTooling/
├── core/                    # Base classes and interfaces
│   ├── base_solver.py      # Abstract solver interface
│   ├── base_objective.py   # Abstract objective interface
│   └── optimization_problem.py
├── objectives/              # Objective function implementations
│   └── margin_loss_objective.py
├── utils/                   # Utility functions
│   ├── weight_selection.py # Neuron/weight selection utilities
│   └── integration_utils.py # StateTooling integration
├── examples/                # Usage examples
│   └── basic_margin_optimization.py
└── REQUIREMENTS_AND_DESIGN.txt  # Detailed specification
```

## Integration with Existing Tools

### StateTooling Integration

- Loads ALTER/PRESERVE states from `optimisation_states/alter/states.json` and `optimisation_states/preserve/states.json`
- Extracts target actions from `desired_action` fields in ALTER states
- Supports both single target actions and multiple acceptable actions

### Weight Selection

- **Current**: Neuron-level selection (all 64 weights per neuron)
- **Scope**: First 3 layers only (features, q_net.0, q_net.2)  
- **Future**: Individual weight selection with `{neuron_idx: [weight_indices]}` format

## Important Notes

### Required Implementation

You must implement `load_model_from_agent_path()` in `utils/integration_utils.py` to load your DQN models. This function should:

```python
def load_model_from_agent_path(agent_path: str):
    # Load your model using existing infrastructure
    # Return PyTorch model ready for optimization
    pass
```

### Multi-Target Actions

The system supports multiple acceptable actions per ALTER state:

```python
target_actions = {
    "state_1": [0, 1],    # Either action 0 or 1 acceptable
    "state_2": [3]        # Only action 3 acceptable  
}
```

### Constraint Enforcement

PRESERVE constraints are strictly enforced - the argmax index must remain exactly the same with zero tolerance.

## Next Steps

1. **Implement model loading**: Complete `load_model_from_agent_path()` for your model format
2. **Test prototype**: Run the basic example with your agents
3. **Add solver**: Integrate SLSQP with preserve constraints  
4. **Scale up**: Test with larger neuron sets and state samples
5. **Validate results**: Verify optimization actually improves behavior as expected 