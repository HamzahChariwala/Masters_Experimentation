# Model Analysis and Extraction Tools

A set of tools for analyzing and extracting neural networks from DRL agents.

## Overview

These tools provide the following capabilities:

1. Extract model components from trained DQN agents
2. Save models in formats suitable for analysis and modification
3. Generate statistical analyses of weights and biases
4. Create visualizations of network architecture and weight distributions
5. Support for investigating feature extractors and action networks

## Target Network Exclusion

DQN agents have two networks:
- The **online Q-network** (used for inference and updates)
- The **target Q-network** (used only during training for stability)

This tool deliberately excludes the target network from extraction since it's not needed for inference or analysis. Only the online network is extracted and saved.

## Usage

### Basic Model Extraction

```bash
python extract_agent_model.py <path_to_agent.zip> [output_directory]
```

Example:
```bash
python extract_agent_model.py ../DebuggingAgents/my_dqn_agent.zip
```

### Programmatic API

```python
from ModelAnalysis.ModelExtraction import DQNModelExtractor

# Create the extractor
extractor = DQNModelExtractor(
    agent_path="path/to/agent.zip",
    output_dir="output_directory",
    verbose=True
)

# Save model components
saved_paths = extractor.save_model_components()

# Get specific components
policy_net = extractor.get_policy_network()
q_network = extractor.get_q_network()
feature_extractor = extractor.get_feature_extractor()

# Generate statistics and visualizations
weight_stats = extractor.analyze_weight_statistics()
network_summary = extractor.generate_network_summary()
```

## Output Files

The extractor generates the following files for each agent:

- `architecture.json` - Model architecture details
- `weight_statistics.json` - Statistical analysis of weights and biases (online network only)
- `network_summary.json` - Summary of network layers and parameters (online network only)
- `policy.pt` - Saved PyTorch policy network
- `q_network.pt` - Saved PyTorch Q-network (online network only)
- `policy_state_dict.pt` - Policy state dictionary for weight editing (online network only)
- `q_network_state_dict.pt` - Q-network state dictionary (online network only)
- `visualizations/` - Directory containing visualization images:
  - `weight_distributions.png` - Histograms of weight distributions
  - `network_diagram.png` - Visual representation of network architecture

## Feature Extractor Support

The tool can extract and analyze custom feature extractors if they are available in the agent. For proper loading, ensure that your custom feature extractor classes (like `CustomCombinedExtractor`) are in the Python path before extraction.

## Requirements

- PyTorch
- NumPy
- Stable-Baselines3
- Matplotlib
- Seaborn

## Troubleshooting

If you encounter errors related to importing the feature extractor, ensure that:

1. The Python path includes the project root directory
2. The feature extractor class is in the same location it was during training
3. Dependencies for the feature extractor are installed

## License

MIT 