# Parameter Selection Tools

This directory contains tools for loading and manipulating DQN agent parameters as standard PyTorch models.

## Files

- **`model_loader.py`** - Main module for loading DQN agents as PyTorch models
- **`example_usage.py`** - Comprehensive example demonstrating all functionality
- **`README.md`** - This documentation file

## Quick Start

### Basic Usage

```python
from model_loader import DQNModelLoader

# Load a DQN agent
loader = DQNModelLoader("Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6")

# Get the model as a standard PyTorch module
model = loader.get_model()

# Inspect all parameters
state_dict = model.state_dict()
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")
```

### Command Line Usage

```bash
# Basic model inspection
python Parameter_Selection/model_loader.py --agent_path Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6

# With forward pass testing
python Parameter_Selection/model_loader.py --agent_path Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6 --test_forward

# Save model components
python Parameter_Selection/model_loader.py --agent_path Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6 --output_dir extracted_models/

# Run comprehensive examples
python Parameter_Selection/example_usage.py
```

## Features

### DQNModelLoader Class

The `DQNModelLoader` class provides the following functionality:

#### Loading and Extraction
- **`__init__(agent_path, device="cpu")`** - Load agent and extract model
- **`get_model()`** - Get the main Q-network as a PyTorch module
- **`get_feature_extractor()`** - Get the feature extractor separately
- **`get_q_head()`** - Get the Q-value head separately

#### Inspection and Analysis
- **`print_model_summary()`** - Print comprehensive model information
- **`get_state_dict()`** - Get the model's state dictionary
- **`get_parameter_info()`** - Get detailed parameter statistics
- **`count_parameters(trainable_only=False)`** - Count model parameters

#### Testing and Validation
- **`test_forward_pass(batch_size=1)`** - Test model with dummy data
- **`save_model_components(output_dir)`** - Save model parts for later use

## Model Structure

The loaded models have the following structure:

```
QNetwork(
  (features_extractor): CustomCombinedExtractor(
    (mlp): Sequential(
      (0): Linear(in_features=106, out_features=64, bias=True)
      (1): ReLU()
    )
  )
  (q_net): Sequential(
    (0): Linear(in_features=64, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=5, bias=True)
  )
)
```

## Parameter Access

### State Dictionary Keys

The model's `state_dict()` contains the following parameters:

- **`features_extractor.mlp.0.weight`** - Feature extractor weights (64×106)
- **`features_extractor.mlp.0.bias`** - Feature extractor bias (64)
- **`q_net.0.weight`** - First Q-network layer weights (64×64)
- **`q_net.0.bias`** - First Q-network layer bias (64)
- **`q_net.2.weight`** - Second Q-network layer weights (64×64)
- **`q_net.2.bias`** - Second Q-network layer bias (64)
- **`q_net.4.weight`** - Output layer weights (5×64)
- **`q_net.4.bias`** - Output layer bias (5)

### Parameter Modification Examples

```python
# Get state dict
state_dict = model.state_dict()

# Modify specific weights
with torch.no_grad():
    # Set a specific weight to 1.0
    state_dict['features_extractor.mlp.0.weight'][0, 0] = 1.0
    
    # Add noise to all weights in a layer
    noise = torch.randn_like(state_dict['q_net.0.weight']) * 0.01
    state_dict['q_net.0.weight'] += noise
    
    # Zero out specific neurons
    state_dict['q_net.0.weight'][10, :] = 0.0  # Zero out neuron 10

# Load modified parameters back
model.load_state_dict(state_dict)
```

## Advanced Usage

### Hook-based Analysis

```python
# Register hooks to capture intermediate activations
activations = {}

def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output.detach().clone()
    return hook

# Register hooks on all Linear layers
hooks = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        hook = module.register_forward_hook(hook_fn(name))
        hooks.append(hook)

# Forward pass
with torch.no_grad():
    output = model(dummy_input)

# Clean up hooks
for hook in hooks:
    hook.remove()

# Analyze activations
for name, activation in activations.items():
    print(f"{name}: mean={activation.mean():.4f}, std={activation.std():.4f}")
```

### Parameter Statistics

```python
# Get detailed parameter information
param_info = loader.get_parameter_info()

for name, info in param_info.items():
    print(f"{name}:")
    print(f"  Shape: {info['shape']}")
    print(f"  Parameters: {info['numel']:,}")
    print(f"  Mean: {info['mean']:.4f}")
    print(f"  Std: {info['std']:.4f}")
    print(f"  Range: [{info['min']:.4f}, {info['max']:.4f}]")
```

## Requirements

- PyTorch
- stable-baselines3
- numpy
- Custom feature extractor (`Environment_Tooling.BespokeEdits.FeatureExtractor.CustomCombinedExtractor`)

## Notes

- The model is automatically set to evaluation mode (`model.eval()`)
- All parameters are loaded on CPU by default (specify `device="cuda"` for GPU)
- The feature extractor warning in the output is normal and doesn't affect functionality
- The model expects dictionary inputs with key `'MLP_input'` containing 106-dimensional vectors

## Troubleshooting

### Common Issues

1. **Import Error for CustomCombinedExtractor**
   - Ensure the `Environment_Tooling` directory is in your Python path
   - Check that the feature extractor file exists

2. **Agent File Not Found**
   - Verify the agent path points to a directory containing `agent.zip`
   - Or provide direct path to the `.zip` file

3. **CUDA Errors**
   - Use `device="cpu"` if you don't have CUDA available
   - Ensure PyTorch CUDA version matches your system

### Getting Help

Run the example script to see all functionality in action:
```bash
python Parameter_Selection/example_usage.py
```

This will demonstrate:
- Basic model loading
- Parameter analysis
- Parameter modification
- Layer-by-layer inspection
- Forward pass analysis with hooks 