# State Filter Criteria Reference Guide

This guide provides a comprehensive reference for all available filtering criteria that can be used with the `state_filter.py` script.

## Basic Format

Criteria follow this format:
```
"property:operator:value"
```

For boolean properties:
```
"property:is:true"  or  "property:is:false"
```

Multiple criteria can be combined in a list:
```python
CRITERIA = ["property1:operator:value", "property2:is:true"]
```

## Available Properties

### State Array Properties
These properties are available directly from the state array:

| Property | Index | Description | Example Values |
|----------|-------|-------------|---------------|
| `cell_type` | 0 | Type of cell the agent is on | "floor", "lava" |
| `path_length` | 1 | Length of the current path | 0, 1, 2, ... |
| `lava_steps` | 2 | Number of lava steps taken | 0, 1, 2, ... |
| `reaches_goal` | 3 | Whether the agent reaches the goal | true, false |
| `next_cell_is_lava` | 4 | Whether the next cell is lava | true, false |
| `risky_diagonal` | 5 | Whether the move is a risky diagonal | true, false |
| `target_state` | 6 | Target state identifier | Various values |
| `action_taken` | 7 | Action taken by the agent | 0, 1, 2, 3, 4 |

### Nested Properties
You can also access nested properties using dot notation:

| Property | Description |
|----------|-------------|
| `model_inputs.raw_input` | Raw input array sent to the model |
| `agent.position.x` | Agent's x position (if available) |
| `agent.position.y` | Agent's y position (if available) |
| `agent.direction` | Agent's direction (if available) |

## Available Operators

### Comparison Operators
For numeric and string comparisons:

| Operator | Description | Example |
|----------|-------------|---------|
| `eq` | Equals | `"path_length:eq:10"` |
| `gt` | Greater than | `"lava_steps:gt:5"` |
| `lt` | Less than | `"path_length:lt:100"` |
| `gte` | Greater than or equal | `"path_length:gte:10"` |
| `lte` | Less than or equal | `"lava_steps:lte:3"` |

### Boolean Operators
For boolean properties:

| Operator | Description | Example |
|----------|-------------|---------|
| `is:true` | Property is true | `"reaches_goal:is:true"` |
| `is:false` | Property is false | `"next_cell_is_lava:is:false"` |

## Common Use Cases

### Find states where agent can reach goal without stepping in lava
```python
CRITERIA = ["reaches_goal:is:true", "lava_steps:eq:0"]
```

### Find states where agent is on floor but next cell is lava
```python
CRITERIA = ["cell_type:eq:floor", "next_cell_is_lava:is:true"]
```

### Find states with risky diagonal moves near lava
```python
CRITERIA = ["risky_diagonal:is:true", "next_cell_is_lava:is:true"]
```

### Find states where agent reaches goal despite stepping in lava
```python
CRITERIA = ["reaches_goal:is:true", "lava_steps:gt:0"]
```

### Find states where agent can reach goal without stepping in lava and next step is lava
```python
CRITERIA = ["reaches_goal:is:true", "lava_steps:eq:0", "next_cell_is_lava:is:true", "cell_type:eq:floor"]
```

### Find states with short paths to goal
```python
CRITERIA = ["reaches_goal:is:true", "path_length:lt:15"]
```

### Find states where the agent took a specific action
```python
CRITERIA = ["action_taken:eq:2"]  # Assuming 2 represents the forward action
```

## Tips for Effective Filtering

1. **Combine multiple criteria** to narrow down specific scenarios.
2. **Start with broader filters** and then refine to find edge cases.
3. **Check both positive and negative cases** for comprehensive analysis.
4. **Filter for short paths first** if you're looking for efficient solutions.
5. **Use `cell_type:eq:floor`** to exclude states where the agent is already on lava. 