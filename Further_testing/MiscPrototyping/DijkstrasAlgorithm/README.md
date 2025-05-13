# Dijkstra's Algorithm for MiniGrid Environments

This module provides tools to convert MiniGrid environments into graph representations and find optimal paths using Dijkstra's algorithm. It also includes visualization tools to produce high-quality figures for academic reports.

## Features

- Convert MiniGrid environments to directed graph representations
- Calculate optimal paths from any cell and orientation to the goal
- Support different handling modes for lava tiles:
  - **Normal**: Lava tiles are treated as regular floor tiles (cost = 1)
  - **Costly**: Lava tiles have an increased cost (configurable multiplier)
  - **Blocked**: Lava tiles cannot be traversed
- Generate JSON output with optimal costs for each cell and orientation
- Create high-quality visualizations of the environment and optimal paths

## Requirements

- Python 3.7+
- Required packages:
  - gymnasium
  - minigrid
  - numpy
  - matplotlib
  - networkx
  - seaborn

## Files

- `minigrid_graph.py`: Core module for creating graph representations and running Dijkstra's algorithm
- `run_analysis.py`: Script for analyzing multiple environments and seeds
- `visualize_optimal_paths.py`: Script for creating high-quality visualizations
- `README.md`: This documentation file

## Usage

### Basic Usage

1. Run the analysis on one or more environments:

```bash
# Create a results directory
mkdir -p results

# Run the analysis
python run_analysis.py --envs MiniGrid-Empty-8x8-v0 MiniGrid-LavaCrossingS9N1-v0 --output-dir results
```

2. Create visualizations from the results:

```bash
# Create a visualizations directory
mkdir -p visualizations

# Generate visualizations
python visualize_optimal_paths.py --results-dir results --output-dir visualizations
```

### Advanced Usage

#### Running for Specific Environments or Seeds

```bash
# Run for a specific environment and seed
python run_analysis.py --envs MiniGrid-LavaCrossingS9N1-v0 --seeds 12345 --output-dir results

# Analyze only with specific lava modes
python run_analysis.py --envs MiniGrid-LavaCrossingS9N1-v0 --lava-modes normal costly --output-dir results
```

#### Visualizing Specific Results

```bash
# Visualize results for a specific environment
python visualize_optimal_paths.py --results-dir results --output-dir visualizations --env-id MiniGrid-LavaCrossingS9N1-v0

# Visualize results for a specific lava mode
python visualize_optimal_paths.py --results-dir results --output-dir visualizations --lava-mode blocked
```

## API Reference

### `MiniGridGraphConverter` Class

The main class for converting MiniGrid environments to graphs and running Dijkstra's algorithm.

#### Methods

- `create_graph_from_env(env_id, seed=None, lava_mode="blocked", lava_cost_multiplier=5.0)`: Create a graph representation of a MiniGrid environment
- `run_dijkstra(goal_pos=None)`: Run Dijkstra's algorithm to find optimal paths to a goal
- `export_optimal_costs(output_file)`: Export the optimal costs to a JSON file
- `visualize_graph(output_file=None, show_costs=True)`: Visualize the graph and optionally the optimal costs

### `generate_optimal_paths_for_env` Function

A helper function that runs the entire analysis pipeline for a given environment.

```python
generate_optimal_paths_for_env(env_id, seed=None, lava_modes=None, output_dir="./", visualize=True)
```

## Output Format

The output JSON files contain the optimal costs from each cell and orientation to the goal. The format is:

```json
{
  "MiniGrid-LavaCrossingS9N1-v0_12345": {
    "costs": {
      "1,1": {
        "0": 5.0,
        "1": 4.0,
        "2": 5.0,
        "3": 6.0
      },
      ...
    },
    "width": 11,
    "height": 11
  }
}
```

Where:
- The key is the environment ID and seed
- `costs` contains a dictionary mapping cell positions (x,y) to costs for each orientation
- Orientations are: 0 (right), 1 (down), 2 (left), 3 (up)
- `width` and `height` are the dimensions of the environment

## Visualizations

Two types of visualizations are generated:

1. **Cost Map**: Shows the minimum cost from each cell to the goal across all orientations
2. **Directional Map**: Shows the optimal orientation at each cell and its cost to reach the goal

## Examples

### Example: Calculate optimal paths for a lava crossing environment

```python
from minigrid_graph import generate_optimal_paths_for_env

# Generate optimal paths for all lava modes
results = generate_optimal_paths_for_env(
    env_id="MiniGrid-LavaCrossingS9N1-v0", 
    seed=12345,
    lava_modes=["normal", "costly", "blocked"],
    output_dir="./results",
    visualize=True
)
```

### Example: Custom graph creation and analysis

```python
from minigrid_graph import MiniGridGraphConverter

# Create converter
converter = MiniGridGraphConverter()

# Create graph from environment
graph = converter.create_graph_from_env(
    env_id="MiniGrid-Empty-8x8-v0",
    seed=42,
    lava_mode="normal"
)

# Run Dijkstra's algorithm
costs = converter.run_dijkstra()

# Export the results
converter.export_optimal_costs("empty_8x8_costs.json")

# Visualize the graph
converter.visualize_graph("empty_8x8_graph.png", show_costs=True)
``` 