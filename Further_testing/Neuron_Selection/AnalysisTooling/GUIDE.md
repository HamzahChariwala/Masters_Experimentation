# AnalysisTooling Guide

## Purpose

The AnalysisTooling module provides a framework for analyzing the results of activation patching experiments. Its primary goals are:

1. **Quantify Patching Effects**: Provide numerical metrics that measure how patching specific neurons affects model behavior.
2. **Standardize Analysis**: Establish a common set of metrics to compare different patching experiments.
3. **Automate Processing**: Process large batches of result files efficiently and consistently.
4. **Support Extensibility**: Make it easy to add new analysis metrics as research needs evolve.

## Architecture

The module is organized into several components:

1. **Metrics** (`metrics.py`): Contains individual metric functions that calculate specific measures of patching effects.
2. **Result Processor** (`result_processor.py`): Handles loading, processing, and saving patching result files.
3. **Main Interface** (`patching_analysis.py`): Command-line tool for running analyses on patching results.

### Data Flow

1. Patching experiments produce JSON result files containing baseline and patched model outputs.
2. The analysis script loads these files and extracts relevant data.
3. Metric functions calculate specific measures using the extracted data.
4. Results are added to the original JSON files under a `patch_analysis` section.

## Adding New Metrics

To add a new metric to the system:

1. **Define the metric function** in `metrics.py`:
   ```python
   def your_new_metric(baseline_output: List[List[float]], 
                       patched_output: List[List[float]], 
                       [other_params]) -> Any:
       """
       Description of what the metric calculates.
       
       Args:
           baseline_output: Baseline model output logits
           patched_output: Patched model output logits
           [other_params]: Any additional parameters needed
           
       Returns:
           Calculated metric value
       """
       # Implementation...
       return result
   ```

2. **Add the function to the `METRIC_FUNCTIONS` dictionary** at the bottom of `metrics.py`:
   ```python
   METRIC_FUNCTIONS = {
       # Existing metrics...
       "your_new_metric": your_new_metric,
   }
   ```

3. **Update the imports in `__init__.py`** to expose the new metric:
   ```python
   from .metrics import (
       # Existing imports...
       your_new_metric,
   )
   
   __all__ = [
       # Existing exports...
       'your_new_metric',
   ]
   ```

4. If your metric requires special handling in `analyze_experiment_results()`, update the function in `result_processor.py` to accommodate it.

### Metric Function Guidelines

- **Input Format**: Metric functions should expect baseline and patched outputs as lists of lists (shape typically `[1, n_actions]`).
- **Additional Parameters**: If your metric needs additional information (e.g., action index), include it as a parameter.
- **Error Handling**: Include validation to handle edge cases (empty inputs, etc.).
- **Return Type**: Return values that are JSON-serializable (numbers, strings, lists, dictionaries).
- **Documentation**: Include thorough docstrings explaining the metric's purpose, inputs, and outputs.

## Integration with patching_analysis.py

The main script `patching_analysis.py` provides a command-line interface for running analyses. New metrics are automatically integrated if they follow the pattern above. Users can specify metrics via the `--metrics` argument:

```bash
python patching_analysis.py --agent_path <path> --metrics your_new_metric,existing_metric
```

To process all metrics, users can omit the `--metrics` argument.

## Result Structure

Metrics are stored in the `patch_analysis` section of each experiment in the result files:

```json
{
  "exp_name": {
    "patch_configuration": { ... },
    "results": { ... },
    "patch_analysis": {
      "output_logit_delta": 0.123,
      "your_new_metric": 0.456,
      ...
    }
  }
}
```

## Types of Metrics to Consider

When designing new metrics, consider these categories:

1. **Output-focused metrics**: Measure changes in the model's outputs (logits, probabilities).
2. **Decision-focused metrics**: Analyze how patching affects decision-making (action selection, confidence).
3. **Distribution metrics**: Compare probability distributions (KL divergence, Jensen-Shannon distance).
4. **Activation-focused metrics**: Examine changes in internal activations beyond the final output.
5. **Task-specific metrics**: Measure effects relevant to specific tasks or domains.

## Future Directions

Potential enhancements to consider:

1. **Visualization tools**: Create plots and visualizations of metric results.
2. **Statistical analysis**: Perform significance testing across multiple experiments.
3. **Comparative metrics**: Compare patching effects across different models or configurations.
4. **Automated interpretation**: Provide automated insights into what metrics reveal about model behavior.
5. **Layer-wise metrics**: Analyze how patching effects propagate through network layers.

## Best Practices

1. **Test thoroughly**: Ensure new metrics work correctly with various input formats and edge cases.
2. **Document clearly**: Explain what each metric measures and how to interpret its values.
3. **Consider performance**: For computationally intensive metrics, include performance optimizations.
4. **Maintain backward compatibility**: Avoid breaking changes to existing metric functions.
5. **Use consistent naming**: Follow the established naming patterns for new metrics.

## Troubleshooting

Common issues and solutions:

1. **Missing metrics in output**: Check that your metric is properly added to the `METRIC_FUNCTIONS` dictionary.
2. **Serialization errors**: Ensure your metric returns JSON-serializable values.
3. **Processing errors**: Debug by adding print statements to trace data flow through the system.
4. **NaN or unexpected values**: Add validation to handle edge cases in your metric function.

## Conclusion

The AnalysisTooling module provides a flexible framework for analyzing activation patching results. By following the patterns established here, you can easily extend the system with new metrics to deepen your understanding of neural network behavior through activation patching. 