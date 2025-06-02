# Saliency Tooling

This directory contains tools for gradient-based saliency analysis of DQN models to identify the most important weights for perturbation experiments.

## Structure

```
SaliencyTooling/
├── initial_test.py          # Master script that runs the complete pipeline
├── README.md               # This file
└── InitialGradients/       # Package containing individual analysis tools
    ├── __init__.py         # Package initialization
    ├── gradients.py        # Gradient computation tool
    ├── average.py          # Average gradient magnitude computation
    └── extract.py          # Candidate weight extraction tool
```

## Master Script: initial_test.py

The `initial_test.py` script runs the complete gradient analysis pipeline in three sequential steps:

1. **Gradient Computation**: Computes gradients of all model weights with respect to the highest output logit for all clean input examples
2. **Average Gradients**: Computes average gradient magnitudes across all input examples for each neuron
3. **Weight Extraction**: Extracts the most important candidate weights based on gradient analysis

### Usage

```bash
python initial_test.py \
    --agent_path "path/to/agent/directory" \
    --useful_neurons_path "path/to/useful_neurons.json" \
    --k 2 \
    --m 4 \
    --device cpu
```

### Parameters

- `--agent_path`: Path to the agent directory (must contain `agent.zip`, `activation_inputs/clean_inputs.json`, and `patching_results/cross_metric_summary.json`)
- `--useful_neurons_path`: Path to the useful neurons definitions (default: `Neuron_Selection/ExperimentTooling/Definitions/useful_neurons.json`)
- `--k`: Top k weights per neuron to select (default: 2)
- `--m`: Additional weights to select globally (default: 4)
- `--device`: Device to run on (default: cpu)
- `--output_dir`: Output directory (default: `{agent_path}/gradient_analysis`)

Note: The cross-metric analysis file is automatically located at `{agent_path}/patching_results/cross_metric_summary.json`.

### Output Files

The pipeline generates three files in the output directory:

1. **weight_gradients.json**: Raw gradients for all weights across all input examples
2. **average_gradients.json**: Average gradient magnitudes for each layer/neuron
3. **candidate_weights.json**: Final candidate weights with targeting information

### Example Output Structure

The final `candidate_weights.json` contains:

```json
{
  "weight_targets": [
    {
      "layer_name": "q_net.0.weight",
      "neuron_index": 61,
      "weight_index": 5,
      "gradient_magnitude": 0.126,
      "source": "top_1_of_exp_62_q_net.0",
      "priority": "guaranteed_k_1"
    }
  ],
  "metadata": {
    "k_per_neuron": 2,
    "additional_m": 4,
    "num_candidate_neurons": 8,
    "total_weights_selected": 20,
    "description": "Top 2 weights per neuron plus top 4 additional weights globally"
  }
}
```

## Individual Tools

The individual tools in `InitialGradients/` can also be run independently:

- `python InitialGradients/gradients.py --agent_path ... --output_dir ...`
- `python InitialGradients/average.py --gradients_path ... --output_dir ...`
- `python InitialGradients/extract.py --cross_metric_path ... --average_gradients_path ...`

## Integration with Perturbation Framework

The output `candidate_weights.json` is designed to work seamlessly with the perturbation tooling in `Parameter_Selection/PerturbationTooling/`. The `weight_targets` section provides exact specifications for which weights to modify in perturbation experiments. 