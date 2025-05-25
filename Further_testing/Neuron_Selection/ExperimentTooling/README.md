# ExperimentTooling

This directory contains tools for generating and managing activation patching experiments.

## Overview

The ExperimentTooling package provides scripts to generate and configure different types of activation patching experiments in a structured, reproducible way. The tools generate JSON experiment definition files that specify which neurons or layers to patch in each experiment.

## Contents

- `generate_experiment.py`: Script to generate different types of patching experiment definitions
- `run_experiments.py`: Script to execute patching experiments using the generated definitions
- `Definitions/`: Directory containing generated experiment definition files

## Usage

### Generating Experiment Definitions

The `generate_experiment.py` script creates JSON files that define which neurons to patch in experiments. It supports several experiment types:

#### 1. Single Neuron Experiments

Test each neuron individually:

```bash
python generate_experiment.py --type single_neuron --agent_path /path/to/agent
```

This generates an experiment definition that patches each neuron in each layer individually, allowing for a comprehensive analysis of individual neuron effects.

#### 2. Layer-wise Experiments

Test each layer as a whole:

```bash
python generate_experiment.py --type layer --agent_path /path/to/agent
```

This creates a separate experiment for each layer, patching all neurons in that layer simultaneously.

#### 3. Neuron Group Experiments

Test neurons in groups of a specified size:

```bash
python generate_experiment.py --type neuron_groups --group_size 5 --agent_path /path/to/agent
```

This generates experiments where neurons are patched in groups, allowing for a more efficient search process.

#### 4. Custom Experiments

Define your own experiment specifications:

```bash
python generate_experiment.py --type custom --custom_spec '
[
  {"layer_name": [0, 1, 2]},
  {"another_layer": [5, 10, 15]}
]'
```

Or load from a file:

```bash
python generate_experiment.py --type custom --custom_spec custom_definition.json
```

### Additional Options for Generating Experiments

- `--agent_path`: Path to the agent directory (for automatically extracting layer structure)
- `--layer_structure`: JSON string or file path defining the layer structure (overrides agent_path)
- `--output`: Custom output file path for the generated experiment definition
- `--include_output_logits`: Include experiments that patch output logits (default: True)
- `--exclude_output_logits`: Exclude output logits patching from the experiments

### Dynamic Model Structure Extraction

The script can automatically detect and extract the model architecture from a saved agent:

```bash
python generate_experiment.py --type single_neuron --agent_path /path/to/agent
```

This will:
1. Find the model file (typically a .pt or .pth file) in the agent directory
2. Load the model and inspect its state dictionary
3. Extract layer names and neuron counts from the model's weights
4. Generate experiments based on the actual model structure

If the model analysis fails, the script will fall back to a default layer structure.

### Running Experiments

Once you've generated an experiment definition, you can use the `run_experiments.py` script to execute the defined experiments:

```bash
python run_experiments.py \
    --agent_path /path/to/agent \
    --definition_file ExperimentTooling/Definitions/single_neuron_experiment.json \
    --output_dir results/single_neuron_patching
```

#### Options for Running Experiments

- `--agent_path`: Path to the agent directory (required)
- `--definition_file`: Path to the experiment definition file (required)
- `--output_dir`: Directory to save results (required)
- `--start_index`: Index of the first experiment to run (default: 0)
- `--end_index`: Index of the last experiment to run, exclusive (default: run all)
- `--additional_args`: Additional arguments to pass to the patching script, comma-separated

#### Running a Subset of Experiments

To run only a subset of the experiments (useful for large experiment sets or for distributing work):

```bash
# Run experiments 10-20
python run_experiments.py \
    --agent_path /path/to/agent \
    --definition_file ExperimentTooling/Definitions/single_neuron_experiment.json \
    --output_dir results/single_neuron_patching \
    --start_index 10 \
    --end_index 20
```

## Output Format

### Experiment Definition Format

The generated JSON files contain a list of experiment specifications, where each specification defines which neurons to patch in each layer:

```json
[
  {
    "features_extractor.mlp.0": [0]
  },
  {
    "features_extractor.mlp.0": [1]
  },
  {
    "output_logits": [0]
  },
  ...
]
```

### Result Format

Each experiment will produce its own result file, containing the output of the patching experiment. The files are saved in the specified output directory with a unique suffix for each experiment. 