#!/usr/bin/env python3
"""
Metrics for analyzing the effects of activation patching.

This module contains functions that calculate various metrics to quantify
the effects of patching on model behavior.
"""
import numpy as np
import scipy.stats
from typing import Dict, List, Any, Union, Tuple
from scipy.spatial.distance import euclidean, chebyshev, cosine


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
    KL(baseline || patched) measures how much information is lost when using patched to approximate baseline.
    
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


def reverse_kl_divergence(baseline_output: List[List[float]], 
                         patched_output: List[List[float]]) -> float:
    """
    Calculate the reverse KL divergence between the patched and baseline distributions.
    KL(patched || baseline) measures how much information is lost when using baseline to approximate patched.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        
    Returns:
        KL divergence from patched to baseline distribution
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
    
    # Calculate reverse KL divergence: sum(p_patched * log(p_patched / p_baseline))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    reverse_kl_div = np.sum(patched_probs * np.log((patched_probs + epsilon) / (baseline_probs + epsilon)))
    
    return float(reverse_kl_div)


def top_action_probability_gap(baseline_output: List[List[float]], 
                              patched_output: List[List[float]]) -> Dict[str, Any]:
    """
    Calculate the probability gap between the top action in baseline and patched outputs.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        
    Returns:
        Dictionary with gap, baseline top action, and patched top action
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
    
    return {
        "gap": top_action_gap,
        "baseline_top_action": baseline_top_action,
        "patched_top_action": patched_top_action
    }


def logit_proportion_change(baseline_output: List[List[float]], 
                           patched_output: List[List[float]], 
                           action: int) -> float:
    """
    Calculate the change in the proportion of the original output logit relative to all logits.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        action: Action index to analyze
        
    Returns:
        Change in proportion of the specified logit relative to all logits
    """
    # Convert to numpy arrays
    baseline_array = np.array(baseline_output[0])
    patched_array = np.array(patched_output[0])
    
    # Calculate proportions (using absolute values to handle negative logits)
    baseline_abs = np.abs(baseline_array)
    patched_abs = np.abs(patched_array)
    
    baseline_sum = np.sum(baseline_abs)
    patched_sum = np.sum(patched_abs)
    
    # Calculate the proportions
    baseline_proportion = baseline_abs[action] / baseline_sum if baseline_sum != 0 else 0
    patched_proportion = patched_abs[action] / patched_sum if patched_sum != 0 else 0
    
    # Calculate the change in proportion
    proportion_change = patched_proportion - baseline_proportion
    
    return float(proportion_change)


def euclidean_distance(baseline_output: List[List[float]], 
                      patched_output: List[List[float]]) -> float:
    """
    Calculate the Euclidean distance between baseline and patched logit vectors.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        
    Returns:
        Euclidean distance between the vectors
    """
    # Convert to numpy arrays
    baseline_array = np.array(baseline_output[0])
    patched_array = np.array(patched_output[0])
    
    # Calculate Euclidean distance
    distance = euclidean(baseline_array, patched_array)
    
    return float(distance)


def chebyshev_distance_excluding_top(baseline_output: List[List[float]], 
                                    patched_output: List[List[float]]) -> Dict[str, Any]:
    """
    Calculate the Chebyshev distance (maximum absolute difference) between baseline and patched logits,
    excluding the original top action. This measures the maximum change in any non-winning logit.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        
    Returns:
        Dictionary with the distance and the action that showed the maximum change
    """
    # Convert to numpy arrays
    baseline_array = np.array(baseline_output[0])
    patched_array = np.array(patched_output[0])
    
    # Find the top action in the baseline
    top_action = np.argmax(baseline_array)
    
    # Create copies of the arrays without the top action
    actions = list(range(len(baseline_array)))
    actions.remove(top_action)
    
    # Extract the values without the top action
    baseline_filtered = baseline_array[actions]
    patched_filtered = patched_array[actions]
    
    # Calculate absolute differences
    abs_diffs = np.abs(patched_filtered - baseline_filtered)
    
    # Find the maximum difference and its index
    if len(abs_diffs) > 0:
        max_diff = float(np.max(abs_diffs))
        max_diff_idx = int(np.argmax(abs_diffs))
        max_diff_action = actions[max_diff_idx]
    else:
        # Handle edge case where there's only one action
        max_diff = 0.0
        max_diff_action = -1
    
    return {
        "distance": max_diff,
        "action": max_diff_action
    }


def cosine_similarity(baseline_output: List[List[float]], 
                     patched_output: List[List[float]]) -> float:
    """
    Calculate the cosine similarity between baseline and patched logit vectors.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        
    Returns:
        Cosine similarity (1 - cosine distance) between the vectors
    """
    # Convert to numpy arrays
    baseline_array = np.array(baseline_output[0])
    patched_array = np.array(patched_output[0])
    
    # Calculate cosine similarity
    similarity = 1.0 - cosine(baseline_array, patched_array)
    
    return float(similarity)


def confidence_margin_change(baseline_output: List[List[float]], 
                            patched_output: List[List[float]]) -> Dict[str, Any]:
    """
    Calculate the change in the confidence margin (difference between the top and second-highest logits)
    divided by the sum of all logits to provide a normalized metric.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        
    Returns:
        Dictionary with normalized margin change and related information
    """
    # Convert to numpy arrays
    baseline_array = np.array(baseline_output[0])
    patched_array = np.array(patched_output[0])
    
    # Find top action in baseline
    baseline_top_action = np.argmax(baseline_array)
    
    # Sort the baseline and patched arrays to find top and second values
    baseline_sorted = np.sort(baseline_array)[::-1]  # Sort in descending order
    patched_sorted = np.sort(patched_array)[::-1]
    
    # Calculate the margins
    baseline_margin = baseline_sorted[0] - baseline_sorted[1] if len(baseline_sorted) > 1 else baseline_sorted[0]
    patched_margin = patched_sorted[0] - patched_sorted[1] if len(patched_sorted) > 1 else patched_sorted[0]
    
    # Calculate the sum of absolute logits for normalization
    baseline_sum = np.sum(np.abs(baseline_array))
    patched_sum = np.sum(np.abs(patched_array))
    
    # Normalize margins
    baseline_norm_margin = baseline_margin / baseline_sum if baseline_sum != 0 else 0
    patched_norm_margin = patched_margin / patched_sum if patched_sum != 0 else 0
    
    # Calculate change in normalized margin
    margin_change = patched_norm_margin - baseline_norm_margin
    
    return {
        "normalized_margin_change": float(margin_change),
        "baseline_margin": float(baseline_margin),
        "patched_margin": float(patched_margin),
        "baseline_normalized_margin": float(baseline_norm_margin),
        "patched_normalized_margin": float(patched_norm_margin)
    }


def pearson_correlation(baseline_output: List[List[float]], 
                       patched_output: List[List[float]]) -> Dict[str, Any]:
    """
    Calculate the Pearson correlation coefficient between baseline and patched logit vectors.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        
    Returns:
        Dictionary with correlation coefficient and p-value
    """
    # Convert to numpy arrays
    baseline_array = np.array(baseline_output[0])
    patched_array = np.array(patched_output[0])
    
    # Calculate Pearson correlation
    correlation, p_value = scipy.stats.pearsonr(baseline_array, patched_array)
    
    return {
        "correlation": float(correlation),
        "p_value": float(p_value)
    }


def hellinger_distance(baseline_output: List[List[float]], 
                      patched_output: List[List[float]]) -> float:
    """
    Calculate the Hellinger distance between baseline and patched probability distributions.
    This measures the similarity between two probability distributions and is in [0, 1].
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        
    Returns:
        Hellinger distance between the distributions
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
    
    # Calculate Hellinger distance: 1/sqrt(2) * sqrt(sum((sqrt(p) - sqrt(q))^2))
    hellinger = np.sqrt(0.5 * np.sum((np.sqrt(baseline_probs) - np.sqrt(patched_probs))**2))
    
    return float(hellinger)


def mahalanobis_distance(baseline_output: List[List[float]], 
                        patched_output: List[List[float]]) -> float:
    """
    Calculate the Mahalanobis distance between baseline and patched logit vectors.
    This distance takes into account the correlation between different logits.
    Note: Since we usually only have single samples, we use an identity covariance matrix,
    which simplifies to the Euclidean distance. In practice, a pre-computed covariance
    matrix from multiple runs would be needed for a true Mahalanobis distance.
    
    Args:
        baseline_output: Baseline model output logits
        patched_output: Patched model output logits
        
    Returns:
        Mahalanobis distance between the vectors
    """
    # Convert to numpy arrays
    baseline_array = np.array(baseline_output[0])
    patched_array = np.array(patched_output[0])
    
    # Calculate difference vector
    diff = baseline_array - patched_array
    
    # Since we don't have enough samples to estimate covariance, 
    # we use identity matrix, which simplifies to Euclidean distance
    mahalanobis = np.sqrt(np.sum(diff**2))
    
    return float(mahalanobis)


# Dictionary mapping metric names to their functions
METRIC_FUNCTIONS = {
    "output_logit_delta": output_logit_delta,
    "logit_difference_norm": logit_difference_norm,
    "action_probability_delta": action_probability_delta,
    "kl_divergence": kl_divergence,
    "reverse_kl_divergence": reverse_kl_divergence,
    "top_action_probability_gap": top_action_probability_gap,
    "logit_proportion_change": logit_proportion_change,
    "euclidean_distance": euclidean_distance,
    "chebyshev_distance_excluding_top": chebyshev_distance_excluding_top,
    "cosine_similarity": cosine_similarity,
    "confidence_margin_change": confidence_margin_change,
    "pearson_correlation": pearson_correlation,
    "hellinger_distance": hellinger_distance,
    "mahalanobis_distance": mahalanobis_distance
} 