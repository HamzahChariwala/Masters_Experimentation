# Flexible Spawn Distributions for MiniGrid

This module provides a flexible system for controlling agent spawn locations in MiniGrid environments. It allows for various probability distributions and curriculum learning through temporal transitions.

## Features

- **Multiple distribution types**: Uniform, Poisson, Gaussian, distance-based
- **Point-centric distributions**: Create distributions centered on the goal, lava, or any other point
- **Temporal transitions**: Gradually shift between distributions during training
- **Masking**: Exclude occupied cells or cells adjacent to the goal
- **Visualization tools**: Visualize distributions and spawn frequencies

## Project Structure

The code is organized as follows:

- Core functionality (`DistributionMap` and `FlexibleSpawnWrapper` classes) is in `EnvironmentEdits/BespokeEdits/SpawnDistribution.py`
- Visualization and testing utilities are in the `SpawnDistributions` directory

## Components

### `DistributionMap`

A class for generating and managing probability distributions over a grid:

- `uniform_distribution()`: Create a uniform distribution
- `poisson_from_point()`: Create a Poisson-like distribution from a point
- `gaussian_from_point()`: Create a Gaussian distribution from a point
- `distance_based_from_point()`: Create a distance-based distribution
- `multi_point_distribution()`: Create distributions based on multiple points
- `mask_cells()`: Apply a binary mask to exclude certain cells
- `invert()`: Invert the distribution (far becomes near, near becomes far)
- `temporal_interpolation()`: Interpolate between two distributions
- `sample()`: Sample a position from the distribution

### `FlexibleSpawnWrapper`

A Gymnasium wrapper that applies the distributions to MiniGrid environments:

```python
wrapper = FlexibleSpawnWrapper(
    env,
    distribution_type="poisson_goal",
    distribution_params={"lambda_param": 0.5, "favor_near": True},
    total_timesteps=100000,
    temporal_transition={
        "target_type": "poisson_goal",
        "target_params": {"lambda_param": 0.5, "favor_near": False},
        "rate": 1.0
    }
)
```

## Distribution Types

1. **Uniform**: Equal probability for all valid cells
2. **Poisson from goal**: Exponentially decreasing probability with distance from goal
3. **Gaussian from goal**: Normal distribution centered on goal
4. **Distance-based**: Probability based on distance from goal, with customizable power
5. **Multi-point**: Distributions based on multiple points (e.g., all lava cells)

## Temporal Transitions

The system supports curriculum learning by gradually changing the spawn distribution over time:

1. Start with an easier distribution (e.g., spawning near the goal)
2. Gradually transition to a harder distribution (e.g., spawning far from the goal)
3. Control the transition rate and curve

## Example Usage

### Basic Usage

```python
import gymnasium as gym
from EnvironmentEdits.BespokeEdits.SpawnDistribution import FlexibleSpawnWrapper

# Create environment
env = gym.make("MiniGrid-Empty-8x8-v0")

# Apply the wrapper with a uniform distribution
env = FlexibleSpawnWrapper(
    env,
    distribution_type="uniform"
)

# Use the environment normally
obs, info = env.reset()
```

### Advanced: Temporal Transition

```python
# Start with near-goal spawns, transition to far-from-goal spawns
env = FlexibleSpawnWrapper(
    env,
    distribution_type="poisson_goal",
    distribution_params={"lambda_param": 0.5, "favor_near": True},
    total_timesteps=1000000,  # Total expected timesteps in training
    temporal_transition={
        "target_type": "poisson_goal",
        "target_params": {"lambda_param": 0.5, "favor_near": False},
        "rate": 1.0  # Control transition speed (higher = faster transition)
    }
)
```

## Demonstration and Testing

The module includes:

1. `SpawnDistributions/test_distributions.py`: Tests the different distribution types and visualizes them
2. `SpawnDistributions/demo.py`: Demonstrates how to use the module with MiniGrid environments

Run the demo:

```bash
python SpawnDistributions/demo.py
```

This will create various visualizations in the `demo_outputs` directory.

## Integration with Training

This module can replace the standard `RandomSpawnWrapper` in your training loops:

```python
# Replace this:
env = RandomSpawnWrapper(env, exclude_goal_adjacent=True)

# With this:
env = FlexibleSpawnWrapper(
    env,
    distribution_type="poisson_goal",
    distribution_params={"lambda_param": 0.5, "favor_near": True},
    total_timesteps=total_training_steps,
    temporal_transition={
        "target_type": "poisson_goal",
        "target_params": {"lambda_param": 0.5, "favor_near": False},
        "rate": 0.5
    }
)
```