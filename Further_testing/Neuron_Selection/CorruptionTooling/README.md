# State Corruption Tool

A simple GUI tool for selecting and corrupting neural network input states from filtered evaluation logs.

## Overview

This tool allows you to:
1. Load filtered states from an agent's evaluation logs
2. View the first 8 values of each state and the corresponding key
3. Visualize the remaining values as two square grids
4. Select cells to corrupt (flip 0 to 1 or 1 to 0)
5. Save both the original (clean) and corrupted states to separate files

## Usage

```bash
python corruption_tool.py --path [AGENT_PATH] --seed [RANDOM_SEED]
```

- `--path`: Path to the agent directory containing the filtered_states.json file (required)
- `--seed`: Random seed for state sampling (default: 42)

## GUI Interface

- The top section displays the state key and first 8 values of the array
- The middle section shows two grids representing the remaining values split into two square grids
  - White cells represent 0s
  - Black cells represent 1s
  - Click on cells to select them for corruption (they will turn grey)
- The bottom section contains three buttons:
  - **Skip**: Move to the next state without saving
  - **Terminate**: End the session and save any unsaved changes
  - **✓** (green checkmark): Save the current state and its corrupted version, then move to the next state

## Output Files

The tool generates two JSON files in the agent directory:
- `clean_inputs.json`: Original state arrays
- `corrupted_inputs.json`: Corrupted state arrays (with selected cells flipped)

Each entry in these files uses the original state key and maintains the original input format.

## Requirements

- Python 3.6+
- NumPy
- Tkinter (included in most Python installations) 