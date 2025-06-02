# Weight Perturbation Tooling

This directory contains tools for applying targeted weight perturbations to DQN models and comparing the resulting logits.

## Files

- **`weight_perturbation.py`** - Main WeightPerturbationTool class
- **`perturbation_example.py`** - Comprehensive example showing usage
- **`README.md`** - This documentation file

## Quick Start

### Basic Usage

```python
from model_loader import DQNModelLoader
from weight_perturbation import WeightPerturbationTool

# Load model
loader = DQNModelLoader("path/to/agent")
tool = WeightPerturbationTool(loader)

# Define perturbations
perturbations = {
    "experiment_1": {
        "features_extractor.mlp.0.weight": {
            0: {5: 0.1}  # Change weight at [0,5] by +0.1
        }
    }
}

# Run experiments
input_data = {'MLP_input': torch.randn(1, 106)}
results = tool.run_perturbation_experiments(perturbations, input_data)
```

### Command Line Usage

```bash
# Show model layer information
python Parameter_Selection/PerturbationTooling/weight_perturbation.py --agent_path Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6 --show_layers

# Run example perturbations
python Parameter_Selection/PerturbationTooling/weight_perturbation.py --agent_path Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6

# Run comprehensive example
python Parameter_Selection/PerturbationTooling/perturbation_example.py
```

## Perturbation Dictionary Structure

The perturbation dictionary follows this hierarchical structure:

```python
perturbations = {
    experiment_id: {           # String: Name of the experiment
        layer_name: {          # String: e.g., "features_extractor.mlp.0.weight"
            neuron_index: {    # Int: Which neuron/row in the weight matrix
                weight_index: change_value  # Int -> Float: Which weight to change and by how much
            }
        }
    }
}
```

### Examples

#### Single Weight Change
```python
{
    "test_1": {
        "features_extractor.mlp.0.weight": {
            0: {5: 0.1}  # Change weight at position [0,5] by +0.1
        }
    }
}
```

#### Multiple Weights in Same Neuron
```python
{
    "test_2": {
        "features_extractor.mlp.0.weight": {
            0: {5: 0.1, 10: -0.05, 20: 0.2}  # Change 3 weights in neuron 0
        }
    }
}
```

#### Multiple Neurons
```python
{
    "test_3": {
        "features_extractor.mlp.0.weight": {
            0: {5: 0.1},      # Change one weight in neuron 0
            1: {3: -0.1},     # Change one weight in neuron 1
            5: {10: 0.15}     # Change one weight in neuron 5
        }
    }
}
```

#### Multiple Layers
```python
{
    "test_4": {
        "features_extractor.mlp.0.weight": {
            0: {5: 0.1}
        },
        "q_net.0.weight": {
            10: {15: -0.2}
        },
        "features_extractor.mlp.0.bias": {
            0: {0: 0.5}  # For bias, weight_index is typically 0
        }
    }
}
```

## Available Layers

For the DQN models, you can modify these layers:

### Weight Matrices
- **`features_extractor.mlp.0.weight`** - Feature extractor weights (64×106)
- **`q_net.0.weight`** - First Q-network layer (64×64)
- **`q_net.2.weight`** - Second Q-network layer (64×64)
- **`q_net.4.weight`** - Output layer (5×64)

### Bias Vectors
- **`features_extractor.mlp.0.bias`** - Feature extractor bias (64)
- **`q_net.0.bias`** - First Q-network bias (64)
- **`q_net.2.bias`** - Second Q-network bias (64)
- **`q_net.4.bias`** - Output bias (5)

## WeightPerturbationTool Methods

### Core Methods
- **`__init__(model_loader)`** - Initialize with a DQNModelLoader
- **`run_perturbation_experiments(perturbations, input_data)`** - Run multiple experiments
- **`run_single_perturbation_experiment(config, input_data)`** - Run single experiment
- **`analyze_results(results)`** - Print analysis of results

### Utility Methods
- **`apply_perturbations(spec, layer_name)`** - Apply changes to specific layer
- **`restore_original_weights()`** - Reset model to original state
- **`run_forward_pass(input_data)`** - Run model inference
- **`get_layer_info()`** - Get information about all layers
- **`print_layer_info()`** - Print layer information

## Results Structure

The `run_perturbation_experiments` method returns:

```python
{
    experiment_id: {
        'original_logits': torch.Tensor,     # Original model output
        'perturbed_logits': torch.Tensor,    # Modified model output
        'logit_difference': torch.Tensor,    # Difference between outputs
        'max_change': float,                 # Maximum absolute change
        'mean_abs_change': float,            # Mean absolute change
        'perturbation_config': dict          # Copy of the perturbation config
    }
}
```

## Input Data Format

The tool accepts input data in two formats:

### Dictionary Format (Recommended)
```python
input_data = {'MLP_input': torch.randn(1, 106)}
```

### Tensor Format
```python
input_data = torch.randn(1, 106)  # Will be converted to dict format
```

## Advanced Usage

### Batch Processing
```python
# Process multiple inputs
inputs = [
    {'MLP_input': torch.randn(1, 106)},
    {'MLP_input': torch.randn(1, 106)},
    {'MLP_input': torch.randn(1, 106)}
]

all_results = []
for i, input_data in enumerate(inputs):
    print(f"Processing input {i+1}/{len(inputs)}")
    results = tool.run_perturbation_experiments(perturbations, input_data)
    all_results.append(results)
```

### Custom Analysis
```python
results = tool.run_perturbation_experiments(perturbations, input_data)

for exp_id, result in results.items():
    if 'error' not in result:
        original = result['original_logits'].squeeze()
        perturbed = result['perturbed_logits'].squeeze()
        
        # Check if action changed
        original_action = torch.argmax(original).item()
        perturbed_action = torch.argmax(perturbed).item()
        
        if original_action != perturbed_action:
            print(f"{exp_id}: Action changed from {original_action} to {perturbed_action}")
        
        # Check magnitude of changes
        if result['max_change'] > 0.1:
            print(f"{exp_id}: Large change detected: {result['max_change']:.6f}")
```

### Layer Information Inspection
```python
# Get layer info programmatically
layer_info = tool.get_layer_info()

for layer_name, info in layer_info.items():
    print(f"{layer_name}:")
    print(f"  Shape: {info['shape']}")
    print(f"  Total parameters: {info['numel']:,}")
    
    # Check if it's a weight matrix or bias vector
    if len(info['shape']) == 2:
        print(f"  Weight matrix: {info['shape'][0]} neurons, {info['shape'][1]} inputs each")
    elif len(info['shape']) == 1:
        print(f"  Bias vector: {info['shape'][0]} neurons")
```

## Error Handling

The tool handles various error conditions:

- **Invalid layer names** - Raises ValueError with available layers
- **Index out of bounds** - Raises IndexError with layer shape information
- **Unsupported shapes** - Raises ValueError for unexpected parameter shapes

Failed experiments are included in results with an 'error' key containing the error message.

## Best Practices

1. **Start small** - Begin with single weight changes to understand the tool
2. **Check layer info** - Use `print_layer_info()` to understand available layers
3. **Use meaningful names** - Name experiments descriptively
4. **Validate indices** - Ensure neuron and weight indices are within bounds
5. **Monitor changes** - Check `max_change` and `mean_abs_change` to understand impact
6. **Restore weights** - The tool automatically restores weights, but verify if needed

## Examples

See `perturbation_example.py` for a comprehensive demonstration of all features.

## Troubleshooting

### Common Issues

1. **Index out of bounds**
   ```
   IndexError: Index [64, 106] out of bounds for layer features_extractor.mlp.0.weight with shape [64, 106]
   ```
   - Check layer shapes with `tool.print_layer_info()`
   - Remember that indices are 0-based

2. **Layer not found**
   ```
   ValueError: Layer 'wrong_layer_name' not found in model
   ```
   - Use exact layer names from `tool.get_layer_info()`
   - Case-sensitive layer names

3. **Input format issues**
   - Ensure input has correct shape (batch_size, 106)
   - Use dictionary format with 'MLP_input' key for clarity
``` 