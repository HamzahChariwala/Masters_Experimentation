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

## Agent_Storage Directory Structure

The project now uses a centralized Agent_Storage system that organizes all agent-related data in a standardized structure:

```
Agent_Storage/
├── README.md
├── <agent_name_1>/
│   ├── config.yaml (model card with training parameters)
│   ├── agent/ (saved agent files)
│   │   └── agent.zip
│   ├── logs/ (training logs)
│   ├── evaluations/ (evaluation metrics)
│   ├── visualizations/ (plots and visualizations)
│   └── extracted_model/ (neural network components)
└── <agent_name_2>/
    └── ...
```

This extractor tool automatically places extracted model components in the `extracted_model` directory of the corresponding agent folder.

## Directory Structure

The project has the following structure:
- `Further_testing/` - Project root
  - `Agent_Storage/` - Centralized agent storage
    - `<agent_name>/` - Individual agent folders
  - `DRL_Training/` - Original training code
    - `EnvironmentEdits/` - Environment customizations
  - `Inference_Estimation/` - Code for inference and analysis
    - `ModelAnalysis/` - Model extraction tools

## Usage

### Basic Model Extraction

To extract a model, simply provide the agent name (no need for full path):

```bash
python Inference_Estimation/ModelAnalysis/extract_agent_model.py LavaS11N5_5_exp_10m
```

The tool will automatically:
1. Find the agent in the Agent_Storage directory
2. Extract the model components
3. Save them to the agent's extracted_model directory

You can also specify a custom output directory:
```bash
python Inference_Estimation/ModelAnalysis/extract_agent_model.py LavaS11N5_5_exp_10m custom_output_directory
```

### Programmatic API

```python
from Inference_Estimation.ModelAnalysis.ModelExtraction import DQNModelExtractor

# Create the extractor (it will automatically use the Agent_Storage directory)
extractor = DQNModelExtractor(
    agent_path="Agent_Storage/LavaS11N5_5_exp_10m/agent/agent.zip",
    verbose=True
)

# Save model components to the default location (Agent_Storage/<agent_name>/extracted_model)
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

After refactoring, the tool will automatically look in multiple locations for the FeatureExtractor module, including:
- `EnvironmentEdits.BespokeEdits.FeatureExtractor`  
- `DRL_Training.EnvironmentEdits.BespokeEdits.FeatureExtractor`

## Requirements

- PyTorch
- NumPy
- Stable-Baselines3
- Matplotlib
- Seaborn

## Troubleshooting

If you encounter errors related to importing the feature extractor, ensure that:

1. The Python path includes the project root directory and DRL_Training directory
2. The feature extractor class is in the same location it was during training
3. Dependencies for the feature extractor are installed

## License

MIT 