# Visualisation Tools

This directory contains custom visualization tools for MiniGrid environments.

## LavaCrossing Environment Visualization

The `env_plots.py` module provides custom visualization functions for LavaCrossing S11N5 environments.

### Features

- **Custom styling**: 
  - Walls: Dark grey (#5d5d5d)
  - Floors: Light gray (#f0f0f0)  
  - Lava: Medium grey (#a0a0a0)
  - Grid borders: Faint black (#000000)
  - Goal: Black circle (#000000)

### Functions

#### 1. Basic Environment Visualization

```python
from env_plots import plot_lavacrossing_environment

# Generate visualization for a specific seed
plot_lavacrossing_environment(81102)
```

#### 2. Agent Visualization with Partial Observability

```python
from env_plots import plot_lavacrossing_with_agent

# Generate visualization with agent at position (5,5) facing North (theta=0)
plot_lavacrossing_with_agent(seed=81102, agent_x=5, agent_y=5, agent_theta=0)

# Different orientations
plot_lavacrossing_with_agent(81102, 3, 7, 1)  # Facing East
plot_lavacrossing_with_agent(81102, 8, 3, 2)  # Facing South  
plot_lavacrossing_with_agent(81102, 2, 8, 3)  # Facing West
```

### Agent Visualization Features

- **Black triangular arrowhead**: Shows agent position and orientation
- **Partial observability**: 7x7 window around agent
- **Realistic visibility**: Agent positioned on edge of observation window (can't see behind)
- **Darkened areas**: Cells outside observation window have black overlay (40% opacity)
- **Orientation system**: 
  - 0 = North (up)
  - 1 = East (right)
  - 2 = South (down)  
  - 3 = West (left)

### Function Parameters

#### `plot_lavacrossing_environment()`
- `seed` (int): Seed for environment generation
- `env_id` (str, optional): Environment ID (default: "MiniGrid-LavaCrossingS11N5-v0")
- `save_path` (str, optional): Custom save path. If None, saves to `raw_envs/`

#### `plot_lavacrossing_with_agent()`
- `seed` (int): Seed for environment generation
- `agent_x` (int): Agent x position
- `agent_y` (int): Agent y position
- `agent_theta` (int): Agent orientation (0=North, 1=East, 2=South, 3=West)
- `env_id` (str, optional): Environment ID (default: "MiniGrid-LavaCrossingS11N5-v0")
- `save_path` (str, optional): Custom save path. If None, saves to `partial/`
- `window_size` (int, optional): Size of observation window (default: 7)

### Testing

Run test scripts to generate visualizations:

```bash
cd Visualisation_Tools

# Test basic environment visualization
python render_training_seeds.py

# Test agent visualization
python test_agent_visualization.py
```

### Output

Visualizations are saved as high-resolution PNG files (300 DPI).

**Save locations:**
- Basic environments: `raw_envs/` directory
- Agent visualizations: `partial/` directory

**File naming conventions:**
- Basic: `lavacrossing_s11n5_seed_{seed}.png`
- With agent: `lavacrossing_s11n5_seed_{seed}_agent_{x}_{y}_{theta}.png` 