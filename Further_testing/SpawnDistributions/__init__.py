"""
Flexible Spawn Distributions for MiniGrid Environments.

This module provides a system for controlling spawn positions in grid-based environments 
using various probability distributions, including support for curriculum learning 
through temporal transitions between distributions.

Main components:
- DistributionMap: Creates and manages probability distributions over a grid
- FlexibleSpawnWrapper: Gymnasium wrapper that applies distributions to MiniGrid environments
- SpawnDistributionCallback: Callback for visualizing spawn distributions during training
- EnhancedSpawnDistributionCallback: Improved callback with better wrapper finding abilities
- visualization: Module for visualizing and monitoring spawn distributions
- visualize_distributions: Script for visualizing different types of distributions
- visualize_curriculum: Script for visualizing curriculum learning approaches
"""

# Import from the new location in BespokeEdits
from EnvironmentEdits.BespokeEdits.SpawnDistribution import DistributionMap, FlexibleSpawnWrapper
from SpawnDistributions.visualization import SpawnDistributionCallback, EnhancedSpawnDistributionCallback, generate_final_visualizations

__all__ = [
    'DistributionMap', 
    'FlexibleSpawnWrapper',
    'SpawnDistributionCallback',
    'EnhancedSpawnDistributionCallback',
    'generate_final_visualizations'
]