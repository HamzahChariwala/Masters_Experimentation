"""
Flexible Spawn Distributions for MiniGrid Environments.

This module provides a system for controlling spawn positions in grid-based environments 
using various probability distributions, including support for curriculum learning 
through temporal transitions between distributions.

Main components:
- DistributionMap: Creates and manages probability distributions over a grid
- FlexibleSpawnWrapper: Gymnasium wrapper that applies distributions to MiniGrid environments
"""

from .spawn_distributions import DistributionMap, FlexibleSpawnWrapper

__all__ = ['DistributionMap', 'FlexibleSpawnWrapper']