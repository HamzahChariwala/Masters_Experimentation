# Example Usage Guide

This guide provides step-by-step instructions for using the ExperimentTooling package to generate and run activation patching experiments.

## Prerequisites

- A trained agent model saved in a directory (typically containing a .pt or .pth file)
- The activation patching framework set up

## Workflow Example

### 1. Generating Experiment Definitions

#### Single Neuron Patching

```bash
# Generate experiments to patch each neuron individually
python ExperimentTooling/generate_experiment.py \
    --type single_neuron \
    --agent_path path/to/your/agent \
    --output ExperimentTooling/Definitions/my_single_neuron_experiments.json
```

The script will:
- Analyze the model structure to determine layer sizes
- Generate an experiment for each neuron in the model
- Add experiments for patching output logits
- Save the experiments to the specified JSON file

#### Layer-wise Patching

```bash
# Generate experiments to patch each layer completely
python ExperimentTooling/generate_experiment.py \
    --type layer \
    --agent_path path/to/your/agent
```

This will create a file at `ExperimentTooling/Definitions/layer_experiment.json` with experiments for patching each layer completely.

#### Neuron Group Patching

```bash
# Generate experiments to patch neurons in groups of 10
python ExperimentTooling/generate_experiment.py \
    --type neuron_groups \
    --group_size 10 \
    --agent_path path/to/your/agent
```

This approach is useful for efficient search when there are many neurons to patch.

### 2. Running the Experiments

Once you have generated the experiment definitions, you can run them:

```bash
# Create a directory for results
mkdir -p results/single_neuron_patching

# Run the single neuron patching experiments
python ExperimentTooling/run_experiments.py \
    --agent_path path/to/your/agent \
    --definition_file ExperimentTooling/Definitions/my_single_neuron_experiments.json \
    --output_dir results/single_neuron_patching
```

For large experiment sets, you might want to run them in batches:

```bash
# Run experiments 0-50
python ExperimentTooling/run_experiments.py \
    --agent_path path/to/your/agent \
    --definition_file ExperimentTooling/Definitions/my_single_neuron_experiments.json \
    --output_dir results/single_neuron_patching \
    --start_index 0 \
    --end_index 50

# Run experiments 50-100
python ExperimentTooling/run_experiments.py \
    --agent_path path/to/your/agent \
    --definition_file ExperimentTooling/Definitions/my_single_neuron_experiments.json \
    --output_dir results/single_neuron_patching \
    --start_index 50 \
    --end_index 100
```

### 3. Analyzing the Results

After running the experiments, you can use the AnalysisTooling package to analyze the results:

```bash
# Analyze all results in the directory
python AnalysisTooling/patching_analysis.py \
    --agent_path path/to/your/agent \
    --input_dir results/single_neuron_patching
```

## Example Use Cases

### Finding Important Neurons

1. Generate single neuron patching experiments:
   ```bash
   python ExperimentTooling/generate_experiment.py --type single_neuron --agent_path path/to/agent
   ```

2. Run all experiments:
   ```bash
   python ExperimentTooling/run_experiments.py \
   --agent_path path/to/agent \
   --definition_file ExperimentTooling/Definitions/single_neuron_experiment.json \
   --output_dir results/neuron_importance
   ```

3. Analyze results to find neurons with the largest impact:
   ```bash
   python AnalysisTooling/patching_analysis.py --agent_path path/to/agent --input_dir results/neuron_importance
   ```

### Comparing Layer Importance

1. Generate layer-wise patching experiments:
   ```bash
   python ExperimentTooling/generate_experiment.py --type layer --agent_path path/to/agent
   ```

2. Run the experiments:
   ```bash
   python ExperimentTooling/run_experiments.py \
   --agent_path path/to/agent \
   --definition_file ExperimentTooling/Definitions/layer_experiment.json \
   --output_dir results/layer_importance
   ```

3. Analyze to compare the importance of different layers:
   ```bash
   python AnalysisTooling/patching_analysis.py --agent_path path/to/agent --input_dir results/layer_importance
   ```

## Tips

- For very large models, start with layer-wise or neuron group patching to identify regions of interest
- Then use single neuron patching on those specific layers or groups
- Use the `--exclude_output_logits` flag if you're only interested in internal representations
- Save experiment definitions with descriptive names to keep track of different experiment sets 