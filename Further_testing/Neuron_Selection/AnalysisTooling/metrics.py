#!/usr/bin/env python3
"""
Metrics for analyzing the effects of activation patching.

This module contains functions that calculate various metrics to quantify
the effects of patching on model behavior.
"""
import numpy as np
from typing import Dict, List, Any, Union, Tuple


def output_logit_delta(baseline_output: List[List[float]], 
                      patched_output: List[List[float]], 
                      action: int) -> float:
    """
    Calculate the change in the logit value for the given action between
    baseline and patched outputs.
    
    Args:
        baseline_output: Baseline model output logits (typically shape [1, n_actions])
        patched_output: Patched model output logits (same shape as baseline)
        action: Action index to analyze
        
    Returns:
        Change in logit value (patched - baseline) for the specified action
    """
    # Ensure we have valid data
    if not baseline_output or not patched_output:
        print("Warning: Empty output arrays provided to output_logit_delta")
        return 0.0
    
    # Extract logit values for the specified action
    baseline_logit = baseline_output[0][action]
    patched_logit = patched_output[0][action]
    
    # Calculate the difference
    delta = patched_logit - baseline_logit
    
    return delta


def logit_difference_norm(baseline_output: List[List[float]], 
                         patched_output: List[List[float]]) -> float:
    """
    Calculate the L2 norm of the difference between baseline and patched logits.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        
    Returns:
        L2 norm of the difference vector
    """
    # Convert to numpy arrays
    baseline_array = np.array(baseline_output)
    patched_array = np.array(patched_output)
    
    # Calculate the difference vector
    diff_vector = patched_array - baseline_array
    
    # Calculate the L2 norm
    l2_norm = np.linalg.norm(diff_vector)
    
    return float(l2_norm)


def action_probability_delta(baseline_output: List[List[float]], 
                            patched_output: List[List[float]], 
                            action: int) -> float:
    """
    Calculate the change in probability (using softmax) for the given action
    between baseline and patched outputs.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        action: Action index to analyze
        
    Returns:
        Change in action probability (patched - baseline)
    """
    # Convert to numpy arrays
    baseline_array = np.array(baseline_output[0])
    patched_array = np.array(patched_output[0])
    
    # Apply softmax to get probabilities
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()
    
    baseline_probs = softmax(baseline_array)
    patched_probs = softmax(patched_array)
    
    # Calculate probability difference for the specified action
    prob_delta = patched_probs[action] - baseline_probs[action]
    
    return float(prob_delta)


def kl_divergence(baseline_output: List[List[float]], 
                 patched_output: List[List[float]]) -> float:
    """
    Calculate the KL divergence between the baseline and patched output distributions.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        
    Returns:
        KL divergence from baseline to patched distribution
    """
    # Convert to numpy arrays
    baseline_array = np.array(baseline_output[0])
    patched_array = np.array(patched_output[0])
    
    # Apply softmax to get probabilities
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()
    
    baseline_probs = softmax(baseline_array)
    patched_probs = softmax(patched_array)
    
    # Calculate KL divergence: sum(p_baseline * log(p_baseline / p_patched))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    kl_div = np.sum(baseline_probs * np.log((baseline_probs + epsilon) / (patched_probs + epsilon)))
    
    return float(kl_div)


def top_action_probability_gap(baseline_output: List[List[float]], 
                              patched_output: List[List[float]]) -> Tuple[float, int, int]:
    """
    Calculate the probability gap between the top action in baseline and patched outputs.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        
    Returns:
        Tuple of (probability gap, baseline top action, patched top action)
    """
    # Convert to numpy arrays
    baseline_array = np.array(baseline_output[0])
    patched_array = np.array(patched_output[0])
    
    # Apply softmax to get probabilities
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()
    
    baseline_probs = softmax(baseline_array)
    patched_probs = softmax(patched_array)
    
    # Get top actions
    baseline_top_action = int(np.argmax(baseline_probs))
    patched_top_action = int(np.argmax(patched_probs))
    
    # Calculate probability difference for the top actions
    top_action_gap = float(patched_probs[patched_top_action] - baseline_probs[baseline_top_action])
    
    return (top_action_gap, baseline_top_action, patched_top_action)


# Dictionary mapping metric names to their functions
METRIC_FUNCTIONS = {
    "output_logit_delta": output_logit_delta,
    "logit_difference_norm": logit_difference_norm,
    "action_probability_delta": action_probability_delta,
    "kl_divergence": kl_divergence,
    "top_action_probability_gap": top_action_probability_gap
} 