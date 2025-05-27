"""
InitialGradients package for gradient-based saliency analysis.

This package contains tools for:
1. Computing gradients of model weights with respect to output logits
2. Averaging gradient magnitudes across input examples  
3. Extracting candidate weights based on gradient analysis

Modules:
- gradients: Compute gradients for all model weights
- average: Compute average gradient magnitudes
- extract: Extract top candidate weights for perturbation
"""

from .gradients import compute_gradients_for_agent, save_gradients
from .average import compute_average_gradients, save_average_gradients
from .extract import extract_candidate_weights, save_candidate_weights

__all__ = [
    'compute_gradients_for_agent',
    'save_gradients', 
    'compute_average_gradients',
    'save_average_gradients',
    'extract_candidate_weights',
    'save_candidate_weights'
] 