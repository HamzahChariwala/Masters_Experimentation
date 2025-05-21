# MiniGrid Coordinate System Analysis

This document summarizes the analysis of coordinate system issues in the MiniGrid environment and provides recommendations for more robust mask generation.

## Key Findings

1. **Grid Coordinate System**: 
   - MiniGrid's grid is indexed with (x, y) coordinates
   - The correct parameter order for `grid.get()` is `grid.get(x, y)`, not `grid.get(y, x)`
   - When accessing the grid for display, remember that visualization typically uses (row, col) which becomes (y, x)

2. **Diagonal Actions**:
   - Actions 3 (diagonal left) and 4 (diagonal right) are relative to the agent's orientation
   - For a right-facing agent (dir=0):
     - Action 3 moves northeast (forward + left)
     - Action 4 moves southeast (forward + right)
   - As the agent's orientation changes, the directions rotate accordingly
   - Diagonal movement properly respects collision detection and won't allow "cutting corners"

3. **Mask Generation Issues**:
   - The original `PartialObsWrapper` uses a complex rotation scheme that varies by direction
   - This can lead to confusion when trying to understand the relationship between world coordinates and mask indices
   - Our testing revealed inconsistencies in how objects appear in masks with different agent orientations

## RobustMaskGenerator Implementation

We've implemented a more robust approach to mask generation that addresses these issues:

```python
class RobustMaskGenerator(ObservationWrapper):
    """
    A more robust implementation of partial observation mask generation.
    This wrapper addresses coordinate system issues by:
    
    1. Extracting a window of the grid centered on the agent
    2. Rotating the grid based on the agent's orientation
    3. Generating masks with a consistent coordinate system
    
    All masks are generated in agent-centric coordinates, where:
    - Forward is always at the top of the mask
    - The agent is always at the center of the mask
    """
```

### Key Advantages:

1. **Consistent Orientation**: 
   - The agent is always at the center of the view
   - The agent's forward direction is always toward the top of the mask
   - This makes it much easier to interpret the masks regardless of agent orientation

2. **Correct Coordinate Transformation**:
   - The implementation carefully transforms between world coordinates and mask coordinates
   - It uses the right parameter order for `grid.get(x, y)`

3. **Stability Across Orientations**:
   - Objects appear in consistent relative positions regardless of agent orientation
   - This makes it easier for both humans and algorithms to interpret the masks

## Recommendations

1. **Replace PartialObsWrapper with RobustMaskGenerator**:
   - Use the more robust implementation for more consistent behavior
   - This will make it easier to debug and understand agent behavior

2. **Be Explicit About Coordinate Systems**:
   - Always document which coordinate system you're using (world, mask, or display)
   - Use comments to clarify parameter ordering, especially for functions like `grid.get()`

3. **Test Thoroughly**:
   - Run the provided test script to validate your environment's behavior
   - Visualize masks at different agent orientations to ensure proper rotation

4. **Understanding Diagonal Actions**:
   - Remember that diagonal actions are relative to the agent's orientation
   - Pay attention to collision detection when using diagonal actions near walls

## Test Visualizations

Our tests generate several visualizations that help understand the coordinate systems:

- `mask_test.png`: Basic mask generation visualization
- `mask_test_PartialObsWrapper_dir*.png`: Masks from different orientations with original implementation
- `mask_test_RobustMaskGenerator_dir*.png`: Masks from different orientations with robust implementation
- `action_visualization.png`: Results of different actions from different orientations
- `coordinate_test.png`: Comparison between grid and mask coordinate systems

By examining these visualizations and reading the test output, you can better understand how the coordinate systems work in MiniGrid. 