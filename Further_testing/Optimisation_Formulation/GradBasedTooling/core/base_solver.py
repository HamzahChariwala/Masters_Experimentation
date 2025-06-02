"""
Abstract base class for optimization solvers.

Defines the interface that all concrete solver implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BaseSolver(ABC):
    """
    Abstract base class for optimization solvers.
    
    All concrete solver implementations must inherit from this class and
    implement the required methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the solver with configuration parameters.
        
        Args:
            config: Dictionary of solver-specific configuration parameters
        """
        self.config = config or {}
        self.solution = None
        self.convergence_info = {}
        
    @abstractmethod
    def solve(self, 
              objective_func: callable,
              initial_guess: np.ndarray,
              bounds: Optional[Tuple] = None,
              constraints: Optional[list] = None) -> Dict[str, Any]:
        """
        Solve the optimization problem.
        
        Args:
            objective_func: Function to minimize (takes parameters, returns scalar)
            initial_guess: Starting point for optimization
            bounds: Bounds on variables (min, max) pairs
            constraints: List of constraint functions/dictionaries
            
        Returns:
            Dictionary containing:
                - 'success': Boolean indicating if optimization succeeded
                - 'x': Optimal solution vector
                - 'fun': Objective function value at solution
                - 'message': Status message
                - 'nit': Number of iterations
        """
        pass
    
    @abstractmethod
    def get_solution(self) -> Optional[np.ndarray]:
        """
        Get the current solution vector.
        
        Returns:
            Solution vector if available, None otherwise
        """
        pass
    
    @abstractmethod
    def get_convergence_info(self) -> Dict[str, Any]:
        """
        Get convergence information from the last optimization run.
        
        Returns:
            Dictionary with convergence details (iterations, function calls, etc.)
        """
        pass
    
    def validate_inputs(self, 
                       objective_func: callable,
                       initial_guess: np.ndarray,
                       bounds: Optional[Tuple] = None,
                       constraints: Optional[list] = None) -> bool:
        """
        Validate inputs to the solver.
        
        Args:
            objective_func: Objective function to validate
            initial_guess: Initial parameter vector
            bounds: Variable bounds
            constraints: Constraint specifications
            
        Returns:
            True if inputs are valid, False otherwise
        """
        if not callable(objective_func):
            return False
            
        if not isinstance(initial_guess, np.ndarray):
            return False
            
        if initial_guess.ndim != 1:
            return False
            
        return True 