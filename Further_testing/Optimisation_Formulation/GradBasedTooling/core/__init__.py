"""
Core optimization components for gradient-based neural network optimization.

This module provides the base classes and interfaces for implementing
various optimization methods and objective functions.
"""

from .base_solver import BaseSolver
from .base_objective import BaseObjective

__all__ = ['BaseSolver', 'BaseObjective'] 