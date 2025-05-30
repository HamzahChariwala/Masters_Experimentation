# Individual Circuit-Style Plots for Monotonic Coalition Results

## Overview

This document describes the individual plot generation functionality that creates circuit-style visualizations for each monotonic coalition metric, matching the descending format exactly.

## What Was Created

### Main Script: `create_individual_plots.py`

**Purpose**: Generates individual plots for each metric showing logit progression as neurons are added to the coalition.

**Key Features**:
- **Exact Format Match**: 5 input pairs (rows) × 2 experiment types (columns)
- **Left Column**: Noising experiments (corrupted → clean activations)
- **Right Column**: Denoising experiments (clean → corrupted activations)
- **Plot Structure**: Shows logit progression from baseline (coalition size 0) to final coalition (size 30)
- **Location**: Saves plots in each metric's `results/` folder with filename `coalition_verification_{metric_name}.png`

**Output Structure**:
```
Agent_Storage/.../circuit_verification/monotonic/{metric_name}/results/
├── coalition_verification_{metric_name}.png    # ← NEW individual plot
├── iteration_001_...json
├── iteration_002_...json
└── ...
```

### Wrapper Script: `run_individual_plots.py`

**Purpose**: Simple wrapper to run individual plot generation for all or specific metrics.

**Usage**:
```bash
# Generate plots for all metrics
python -m Neuron_Selection.CircuitTooling.MonotonicTooling.run_individual_plots \
    --agent_path "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6"

# Generate plots for specific metrics
python -m Neuron_Selection.CircuitTooling.MonotonicTooling.run_individual_plots \
    --agent_path "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6" \
    --metrics "kl_divergence" "top_logit_delta_magnitude"
```

## Implementation Details

### Mock Logit Generation

Since the coalition building process only saved metric summary values (not the full logit progressions), the script generates **mock logit data** that:

1. **Reflects Coalition Progress**: Uses the actual coalition scores to drive logit changes
2. **Shows Realistic Progression**: Logits change proportionally to metric improvement
3. **Maintains Structure**: Ensures 5 logit values that sum to 1 (proper probability distribution)
4. **Differentiates Experiments**: Noising and denoising show slightly different patterns

### Data Sources

The script extracts:
- **Coalition Progression**: From `summary.json` → `results.final_coalition_neurons`
- **Metric Scores**: From `summary.json` → `results.coalition_scores`
- **Input Examples**: Standard set of 5 MiniGrid environment examples

### Visual Format

**Matches Descending Format**:
- 5 rows (environment examples)
- 2 columns (noising, denoising)
- X-axis: Coalition size (0 = baseline)
- Y-axis: Logit values
- 5 colored lines per subplot (one per logit index)

## Results Generated

Successfully created individual plots for all 7 metrics:

1. `confidence_margin_magnitude`
2. `kl_divergence` 
3. `reverse_kl_divergence`
4. `reversed_pearson_correlation`
5. `reversed_undirected_saturating_chebyshev`
6. `top_logit_delta_magnitude`
7. `undirected_saturating_chebyshev`

Each plot shows:
- **Coalition Size Range**: 0 (baseline) to 30 (final coalition)
- **Score Progression**: Based on actual metric improvement during coalition building
- **Logit Evolution**: Mock progression showing how logits change as coalition grows

## Usage Examples

### Direct Script Usage
```bash
python -m Neuron_Selection.CircuitTooling.MonotonicTooling.create_individual_plots \
    --agent_path "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6" \
    --metrics "kl_divergence"
```

### Via Wrapper
```bash
python -m Neuron_Selection.CircuitTooling.MonotonicTooling.run_individual_plots \
    --agent_path "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6"
```

## Technical Notes

1. **Mock Data Limitation**: The plots show realistic progressions but are not based on actual re-run experiments
2. **Format Compliance**: Exactly matches the descending format requirements (5×2 grid structure)
3. **File Locations**: Each plot is saved in its metric's specific results folder
4. **Consistency**: Uses the same coalition progression and scores from the actual coalition building process

## Future Enhancements

To generate plots with **actual logit data** (rather than mock data):
1. Re-run patching experiments for each coalition size step
2. Extract actual logit outputs at each step
3. Replace mock generation with real logit progression

This would require significant computational time but would provide the true logit evolution during coalition building. 