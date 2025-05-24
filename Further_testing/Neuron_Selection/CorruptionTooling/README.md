# State Corruption Tool

A simple GUI tool for selecting and corrupting neural network input states from filtered evaluation logs.

## Overview

This tool allows you to:
1. Load filtered states from an agent's evaluation logs
2. View the state key and action number for each state
3. Visualize the input as two square grids
4. Select cells to corrupt (flip 0 to 1 or 1 to 0)
5. Save both the original (clean) and corrupted states to separate files

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
  - The state key
  - The action number taken by the agent
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

The tool generates two JSON files in the agent directory:
- `clean_inputs.json`: Original state arrays
- `corrupted_inputs.json`: Corrupted state arrays (with selected cells flipped)

Each entry in these files uses the original state key and maintains the original input format.

## Verifying Results

After using the tool, you can verify the differences between the clean and corrupted inputs:

```bash
python Neuron_Selection/CorruptionTooling/verify_outputs.py --path [AGENT_PATH]
```

This will show which cells were modified for each state.

## Troubleshooting

If the GUI window doesn't appear immediately, check:
1. Other workspaces/virtual desktops
2. Behind other windows
3. Minimized windows in your taskbar/dock

## Requirements

- Python 3.6+
- NumPy
- Tkinter (included in most Python installations)

## Alternative Usage Methods

For testing with sample data:

```bash
# Generate sample data and run the tool
python Neuron_Selection/CorruptionTooling/test_tool.py --output test_agent
```

## Requirements

- Python 3.6+
- NumPy
- Tkinter (included in most Python installations)

## Troubleshooting

If you're having trouble with the tool:

1. **GUI Not Appearing**: Try running the `tk_test.py` script to verify Tkinter is working:
   ```bash
   python Neuron_Selection/CorruptionTooling/tk_test.py
   ```

2. **File Not Found Errors**: Make sure your agent directory contains a `filtered_states.json` file

3. **JSON Format Issues**: Ensure your filtered_states.json follows the expected format with each entry containing an "input" array 