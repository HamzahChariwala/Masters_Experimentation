# Activation Patching System

This system allows for detailed analysis of neural network behavior by selectively replacing (patching) activations during forward passes. This enables identification of causal relationships between specific neuron activations and model outputs.

## Core Functionality

1. **Flexible Activation Patching**
   - Patch activations at specific layers and neurons
   - Support for patching multiple layers simultaneously
   - Allow for patching individual neurons or entire layers
   - Support for batch experiments with multiple patch configurations

2. **Source and Target Configuration**
   - Ability to specify a source file for donor activations
   - Ability to specify a target file for the activations to be patched
   - Support various file formats (npz, json) from the activation_logging directory

3. **Comprehensive Logging**
   - Log all activation data during patched runs
   - Record original and patched outputs for comparison
   - Save results in the same format as the activation extraction for consistency

4. **Experimental Control**
   - Compare patched behavior against baseline (unpatched) behavior
   - Measure the causal effect of specific neurons on model outputs
   - Support for different types of patching experiments beyond just clean vs. corrupted inputs

5. **Integration with Existing Tools**
   - Work seamlessly with activation data from the activation_extraction.py script
   - Compatible with the existing agent models and observation formats
   - Utilize the agent's activation_logging directory structure

## Patch Specification Formats

The system supports two ways to specify which neurons to patch:

1. **Command-line string format**:
   ```
   --patch_spec "layer1:neuron1,neuron2;layer2:all"
   ```

2. **JSON file format** (for batch experiments):
   ```json
   [
     {"layer1": [1, 2, 3], "layer2": [4, 5]},
     {"layer3": "all"},
     {"layer1": [1], "layer3": [7, 8]}
   ]
   ```
   Each object in the array represents a separate patching experiment.

## Use Cases

1. **Corrupted Input Analysis**
   - Identify which neurons are responsible for different decisions when inputs are corrupted
   - Restore proper functionality by patching critical neurons with clean activations

2. **Ablation Studies**
   - Systematically "knock out" neurons or layers to measure their impact
   - Identify the minimal set of neurons required for specific behaviors

3. **Causal Tracing**
   - Track the flow of information through the network
   - Identify which neurons in earlier layers affect specific neurons in later layers

4. **Feature Visualization**
   - Understand what specific neurons or groups of neurons are detecting
   - Correlate neuron activations with input features

5. **Robustness Testing**
   - Test how the model responds to targeted activation changes
   - Identify potential vulnerabilities in the decision-making process

## Implementation Components

1. **Core Patching Classes**
   - `NeuronPatcher`: Basic patching functionality
   - `AdvancedPatcher`: Enhanced patching with additional features
   - `PatchingExperiment`: High-level class for running patching experiments

2. **Data Loading and Management**
   - `ActivationLoader`: Utilities for loading and processing activation data

3. **Helper Tools**
   - `find_different_neurons.py`: Utility for identifying neurons with significant differences

## Success Criteria

The system will be considered successful if it can:

1. Load activation data from specified files in the activation_logging directory
2. Selectively patch activations across multiple layers in a single forward pass
3. Log all activation data during patched runs in the same format as the activation extraction
4. Produce clear, interpretable results showing the effect of patching
5. Handle various input formats and agent architectures
6. Be extendable for future patching experiments beyond clean vs. corrupted inputs 