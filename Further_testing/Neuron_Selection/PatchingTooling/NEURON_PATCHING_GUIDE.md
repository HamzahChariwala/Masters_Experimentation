# Guide to Neuron Patching

This guide explains how to use the activation patching system to analyze neural network behavior by patching specific neurons or layers.

## Patch Specification Formats

The patching system supports two formats for specifying which neurons to patch:

### 1. Command-line String Format

```
"layer1:neurons;layer2:neurons;layer3:neurons"
```

Where:
- `layer1`, `layer2`, etc. are the names of layers in the model
- `neurons` can be either:
  - `all` - to patch the entire layer
  - A comma-separated list of neuron indices (e.g., `5,7,10`) to patch specific neurons

Examples:
- `"q_net.0:all"` - Patch the entire first layer of the q-network
- `"q_net.2:5,6,7"` - Patch only neurons 5, 6, and 7 in the third layer of the q-network
- `"q_net.0:10,11;q_net.2:5"` - Patch neurons 10 and 11 in the first layer, and neuron 5 in the third layer

### 2. JSON File Format for Batch Experiments

For more complex experiments or to run multiple patching configurations in sequence, you can use a JSON file:

```json
[
  {"q_net.0": [2, 4]},
  {"q_net.2": [10, 11]},
  {"q_net.0": [5, 6, 7], "q_net.2": [8, 9]},
  {"q_net.2": "all"}
]
```

Each object in the array represents a separate patching experiment:
- Keys are layer names
- Values are either a list of neuron indices or the string "all"

This format allows for:
- Running multiple experiments in sequence
- Patching different combinations of neurons across layers
- Comparing results across different patching strategies

## How to Identify Neurons to Patch

Finding the right neurons to patch requires analysis of activations. Here are several approaches:

### 2. Comparing Clean and Corrupted Activations Manually

To identify neurons that behave differently in clean vs. corrupted inputs:

1. Extract activations for both clean and corrupted inputs:
   ```
   python activation_extraction.py --agent_path <agent_path> --inputs clean_inputs.json
   python activation_extraction.py --agent_path <agent_path> --inputs corrupted_inputs.json
   ```

2. Analyze the activation differences:
   - Look for neurons with the largest differences in activation values
   - Focus on neurons that change from active to inactive (or vice versa)
   - Pay attention to neurons that show consistent differences across multiple inputs

3. Example analysis script:
   ```python
   import numpy as np
   import json
   
   # Load clean and corrupted activations
   with open("path/to/clean_activations_readable.json", "r") as f:
       clean = json.load(f)
   
   with open("path/to/corrupted_activations_readable.json", "r") as f:
       corrupted = json.load(f)
   
   # For a specific input and layer, find neurons with largest differences
   input_id = "your-input-id"
   layer = "q_net.2"
   
   clean_values = np.array(clean[input_id][layer])
   corrupted_values = np.array(corrupted[input_id][layer])
   
   # Calculate absolute differences
   differences = np.abs(clean_values - corrupted_values)
   
   # Get indices of neurons with largest differences
   top_different_neurons = np.argsort(differences[0])[-10:]  # Top 10 different neurons
   print(f"Top different neurons in {layer}: {top_different_neurons}")
   ```

### 3. Analyzing ReLU Activations

Identify neurons that are "dead" (outputting zero) in one condition but active in another:

```python
# Find neurons that are active in clean but dead in corrupted
clean_active = clean_values[0] > 0
corrupted_dead = corrupted_values[0] == 0

neurons_of_interest = np.where(clean_active & corrupted_dead)[0]
print(f"Neurons active in clean but dead in corrupted: {neurons_of_interest}")
```

### 4. Systematic Exploration

Try patching groups of neurons systematically to identify important ones:

1. Start with patching the entire layer to see if it has an effect
2. If it does, try patching half the neurons at a time to narrow down
3. Continue narrowing down to find the minimal set of neurons that cause the effect

## Running Patching Experiments

### Single Experiment with Command-line Format

```bash
python activation_patching.py \
    --agent_path <path_to_agent> \
    --target_input <input_file.json> \
    --source_activations <activation_file.npz> \
    --patch_spec "<layer>:<neurons>;..." \
    --output_prefix <prefix_for_output>
```

### Batch Experiments with JSON File

```bash
python activation_patching.py \
    --agent_path <path_to_agent> \
    --target_input <input_file.json> \
    --source_activations <activation_file.npz> \
    --patches_file <path_to_patches.json> \
    --output_prefix <prefix_for_output>
```

### Examples

1. Patch specific neurons in multiple layers:
   ```bash
   python activation_patching.py \
       --agent_path ../Agent_Storage/SpawnTests/biased/biased-v1 \
       --target_input corrupted_inputs.json \
       --source_activations clean_activations.npz \
       --patch_spec "q_net.0:5,6,7;q_net.2:10,11" \
       --output_prefix experiment1
   ```

2. Patch an entire layer:
   ```bash
   python activation_patching.py \
       --agent_path ../Agent_Storage/SpawnTests/biased/biased-v1 \
       --target_input corrupted_inputs.json \
       --source_activations clean_activations.npz \
       --patch_spec "q_net.2:all" \
       --output_prefix experiment2
   ```

3. Focus on a specific input by providing its ID:
   ```bash
   python activation_patching.py \
       --agent_path ../Agent_Storage/SpawnTests/biased/biased-v1 \
       --target_input corrupted_inputs.json \
       --source_activations clean_activations.npz \
       --patch_spec "q_net.0:5,6,7" \
       --input_ids "MiniGrid-LavaCrossingS11N5-v0-81110-6,4,0-0763" \
       --output_prefix experiment3
   ```

4. Run a batch of experiments using a JSON file:
   ```bash
   python activation_patching.py \
       --agent_path ../Agent_Storage/SpawnTests/biased/biased-v1 \
       --target_input corrupted_inputs.json \
       --source_activations clean_activations.npz \
       --patches_file PatchingTooling/patches_sample.json \
       --output_prefix batch_experiments
   ```

## Interpreting Results

The results of patching experiments are saved in JSON files in the `patching_results` directory with the following information:

- `baseline_output`: The output values before patching
- `baseline_action`: The action chosen before patching
- `patched_output`: The output values after patching
- `patched_action`: The action chosen after patching
- `action_changed`: Whether the action changed due to patching
- `patch_configuration`: The exact patch configuration used (added for reference)

A successful patch typically results in the patched action matching the action from the source activations, indicating that those neurons are causally responsible for the decision.

## Tips for Effective Patching

1. **Start broad, then narrow down**: First patch entire layers, then narrow down to specific neurons.

2. **Compare multiple inputs**: Look for patterns across different inputs to identify consistently important neurons.

3. **Check layer relevance**: Some layers may have more influence on the final output than others.

4. **Analyze activation patterns**: Look for neurons that show distinct patterns in clean vs. corrupted inputs.

5. **Consider ablation studies**: Try "zeroing out" specific neurons instead of patching them to see their importance.

6. **Use batch experiments**: For systematic exploration, create a JSON file with multiple patch configurations to test different hypotheses in sequence.

## Advanced Analysis

For more sophisticated analysis, you can use the Python classes directly in your own scripts:

```python
from Neuron_Selection.PatchingTooling import PatchingExperiment

experiment = PatchingExperiment(agent_path="path/to/agent")

# Single patch experiment
results1 = experiment.run_patching_experiment(
    target_input_file="inputs.json",
    source_activation_file="activations.npz",
    patch_spec={
        "q_net.0": [5, 10, 15],  # Specific neurons
        "q_net.2": "all"         # Entire layer
    }
)

# Multiple patches in a loop
patch_configs = [
    {"q_net.0": [2, 4]},
    {"q_net.2": [10, 11]},
    {"q_net.0": [5, 6, 7], "q_net.2": [8, 9]}
]

for i, patch in enumerate(patch_configs):
    results = experiment.run_patching_experiment(
        target_input_file="inputs.json",
        source_activation_file="activations.npz",
        patch_spec=patch
    )
    # Analyze results...
    experiment.save_results(results, f"experiment_{i+1}_results.json")
```

This allows for more customized experiments and analyses. 