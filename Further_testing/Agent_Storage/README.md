# Agent Storage System

This directory serves as the central storage location for all trained DRL agents and their associated data. Each agent has its own folder containing all relevant information.

## Structure

```
Agent_Storage/
├── README.md (this file)
├── <agent_name_1>/
│   ├── config.yaml (model card with parameters used for training)
│   ├── agent/ (saved agent files)
│   │   └── agent.zip
│   ├── logs/ (training logs)
│   ├── evaluations/ (evaluation metrics)
│   ├── visualizations/ (plots and visualizations)
│   └── extracted_model/ (neural network components)
├── <agent_name_2>/
│   └── ...
└── ...
```

## Files That Interact With Agent_Storage

The following files read from or write to the Agent_Storage directory:

### Reading Agent Data
- `DRL_Training/import_vars.py` - Loads model configurations from config.yaml
- `DRL_Training/main.py` - Loads agent for continued training or evaluation
- `Inference_Estimation/ModelAnalysis/extract_agent_model.py` - Reads agent files for model extraction

### Writing Agent Data
- `DRL_Training/main.py` - Saves trained agents
- `DRL_Training/performance_logging.py` - Saves training logs and metrics
- `Inference_Estimation/ModelAnalysis/ModelExtraction.py` - Saves extracted model components
- `DRL_Training/tensorboard_visualisation.py` - Saves visualizations

## Usage

When creating a new agent:
1. Create a new directory with a descriptive name
2. Copy the training configuration into `config.yaml` 
3. All outputs related to this agent will automatically be saved in this folder

## Benefits

- Centralized storage makes it easier to find all information about an agent
- Standardized structure allows for easier automation and comparison
- Storing the config alongside the agent provides complete provenance information
- Simplifies backup and sharing

## Adding New Agents

To add a new agent to the system:

```bash
mkdir -p Agent_Storage/<agent_name>/agent
mkdir -p Agent_Storage/<agent_name>/logs
mkdir -p Agent_Storage/<agent_name>/evaluations
mkdir -p Agent_Storage/<agent_name>/visualizations
mkdir -p Agent_Storage/<agent_name>/extracted_model

# Copy the configuration file
cp <config_file> Agent_Storage/<agent_name>/config.yaml
``` 