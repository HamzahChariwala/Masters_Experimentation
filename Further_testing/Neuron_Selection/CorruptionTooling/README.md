# State Corruption Tool

A simple GUI tool for selecting and corrupting neural network input states from filtered evaluation logs.

## Overview

This tool allows you to:
1. Load filtered states from an agent's evaluation logs
2. View the state key, first 8 values, and additional properties (action taken, risky_diagonal, next_cell_is_lava)
3. Visualize the input as two square grids
4. Select cells to corrupt (flip 0 to 1 or 1 to 0)
5. Save both the original (clean) and corrupted states to separate files in the activation_inputs folder

## Running the Tool

To run the tool with your agent data:

```bash
# Run from the project root directory:
python Neuron_Selection/CorruptionTooling/run.py --path Agent_Storage/SpawnTests/biased/biased-v1
```

You can also specify a random seed for state selection:

```bash
python Neuron_Selection/CorruptionTooling/run.py --path [AGENT_PATH] --seed [RANDOM_SEED]
```

## GUI Interface

- The top section displays:
  - The state key and first 8 values of the input array
  - Additional properties: Action taken, Risky Diagonal status, and Next Cell is Lava status
- The middle section shows two grids representing the input values split into two square grids
  - White cells represent 0s
  - Black cells represent 1s
  - Click on cells to select them for corruption (they will turn grey)
- The bottom section contains:
  - **Skip**: Move to the next state without saving
  - **Terminate**: End the session and save any unsaved changes
  - **✓** (green checkmark): Save the current state and its corrupted version, then move to the next state
  - A progress counter showing states processed and saved

## Output Files

The tool generates two JSON files in the `activation_inputs` directory within the agent directory:
- `clean_inputs.json`: Original state arrays
- `corrupted_inputs.json`: Corrupted state arrays (with selected cells flipped)

Each entry in these files uses the original state key and maintains the original input format.

## Requirements

- Python 3.6+
- NumPy
- Tkinter (included in most Python installations)

## Troubleshooting

If the GUI window doesn't appear immediately, check:
1. Other workspaces/virtual desktops
2. Behind other windows
3. Minimized windows in your taskbar/dock

If you encounter "File Not Found" errors, make sure your agent directory contains a `filtered_states.json` file. 