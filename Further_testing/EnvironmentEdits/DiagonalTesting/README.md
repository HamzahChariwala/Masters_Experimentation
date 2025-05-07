# Diagonal Movement Testing

This folder contains scripts and test results related to the diagonal movement feature implemented in the custom MinGrid environment.

## Test Scripts

- **test_new_image_in_info.py**: Tests the diagonal movement and extracts the 'new_image' observation to verify position changes
- **test_new_image_diagonal.py**: Tests diagonal movement from the center of the grid to better demonstrate position changes
- **visualize_diagonal_movement.py**: Creates visualizations of diagonal movement using the renderer
- **check_action_implementation.py**: Inspects the internal implementation of diagonal movements
- **comprehensive_move_test.py**: Tests various movement types including regular and diagonal movements
- **diagonal_with_training_wrappers.py**: Tests diagonal movement with the exact same wrappers used in training
- **test_diagonal_move.py**: Basic diagonal movement test
- **test_diagonal_move_with_capture.py**: Tests diagonal movement with image capture

## Test Results

The `test_results` directory contains outputs from the various test scripts:

### new_image_test_results
Contains JSON files showing the 'new_image' observations before and after diagonal moves.

### center_diagonal_test_results
Contains detailed test results from diagonal movement tests starting from the center of the grid.

### diagonal_visualization
Contains images and GIFs showing the visual representation of diagonal movements.

### comprehensive_test_results
Contains test results comparing regular moves with diagonal moves, including movement_test_results.json and visualizations.

### diagonal_move_tests
Contains test results and grid state visualizations from observe_grid_state.py.

### diagonal_test_results
Contains visualization images of diagonal movements from all four directions.

### training_wrappers_test_results
Contains test results when using the exact same wrappers as in training.

## Key Findings

1. **Internal State Behavior**: Diagonal moves correctly update the agent's position in the internal state according to the agent's orientation:
   - When facing Right: diagonal_left moves [1,1], diagonal_right moves [1,-1]
   - When facing Down: diagonal_left moves [-1,1], diagonal_right moves [1,1]
   - When facing Left: diagonal_left moves [-1,-1], diagonal_right moves [-1,1]
   - When facing Up: diagonal_left moves [1,-1], diagonal_right moves [-1,-1]

2. **Observation Representation**: In the 'new_image' observation, the agent stays centered at a fixed position regardless of movement, as the observation is centered on the agent.

3. **Visual Rendering**: The renderer doesn't show a smooth diagonal animation path, but rather updates the agent to the final position in a single render frame.

4. **Implementation Details**: Diagonal movements are implemented by combining a forward vector with a lateral vector based on the agent's orientation. 