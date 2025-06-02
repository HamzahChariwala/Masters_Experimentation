"""
Abstract base class for optimization objective functions.

Defines the interface for computing objective values and gradients.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch


class BaseObjective(ABC):
    """
    Abstract base class for optimization objective functions.
    
    All objective function implementations must inherit from this class.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the objective function.
        
        Args:
            config: Configuration parameters for the objective function
        """
        self.config = config or {}
        
    @abstractmethod
    def compute_objective(self, 
                         weight_perturbations: np.ndarray,
                         **kwargs) -> float:
        """
        Compute the objective function value.
        
        Args:
            weight_perturbations: Vector of weight changes (Δw)
            **kwargs: Additional arguments specific to objective type
            
        Returns:
            Scalar objective function value
        """
        pass
    
    @abstractmethod
    def compute_gradient(self, 
                        weight_perturbations: np.ndarray,
                        **kwargs) -> np.ndarray:
        """
        Compute the gradient of the objective function.
        
        Args:
            weight_perturbations: Vector of weight changes (Δw)
            **kwargs: Additional arguments specific to objective type
            
        Returns:
            Gradient vector with same shape as weight_perturbations
        """
        pass
    
    def compute_objective_and_gradient(self, 
                                     weight_perturbations: np.ndarray,
                                     **kwargs) -> tuple:
        """
        Compute both objective value and gradient.
        
        Default implementation calls both methods separately.
        Can be overridden for efficiency if both can be computed together.
        
        Args:
            weight_perturbations: Vector of weight changes (Δw)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (objective_value, gradient_vector)
        """
        obj = self.compute_objective(weight_perturbations, **kwargs)
        grad = self.compute_gradient(weight_perturbations, **kwargs)
        return obj, grad
    
    def validate_inputs(self, 
                       weight_perturbations: np.ndarray,
                       **kwargs) -> bool:
        """
        Validate inputs to the objective function.
        
        Args:
            weight_perturbations: Weight change vector
            **kwargs: Additional arguments
            
        Returns:
            True if inputs are valid, False otherwise
        """
        if not isinstance(weight_perturbations, np.ndarray):
            return False
            
        if weight_perturbations.ndim != 1:
            return False
            
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration parameters."""
        self.config.update(new_config) 