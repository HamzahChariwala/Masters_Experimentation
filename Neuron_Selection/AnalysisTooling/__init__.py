"""
AnalysisTooling package for analyzing activation patching results.

This package provides tools for analyzing and interpreting the results
of activation patching experiments.
"""

from .metrics import (
    output_logit_delta,
    logit_difference_norm,
    action_probability_delta,
    kl_divergence,
    reverse_kl_divergence,
    top_action_probability_gap,
    logit_proportion_change,
    euclidean_distance,
    chebyshev_distance_excluding_top,
    cosine_similarity,
    confidence_margin_change,
    pearson_correlation,
    hellinger_distance,
    mahalanobis_distance,
    directed_saturating_chebyshev,
    undirected_saturating_chebyshev,
    METRIC_FUNCTIONS
)

from .result_processor import (
    load_result_file,
    save_result_file,
    analyze_experiment_results,
    process_result_file,
    process_result_directory
)

__all__ = [
    # Metrics
    'output_logit_delta',
    'logit_difference_norm',
    'action_probability_delta',
    'kl_divergence',
    'reverse_kl_divergence',
    'top_action_probability_gap',
    'logit_proportion_change',
    'euclidean_distance',
    'chebyshev_distance_excluding_top',
    'cosine_similarity',
    'confidence_margin_change',
    'pearson_correlation',
    'hellinger_distance',
    'mahalanobis_distance',
    'directed_saturating_chebyshev',
    'undirected_saturating_chebyshev',
    'METRIC_FUNCTIONS',
    
    # Result processing
    'load_result_file',
    'save_result_file',
    'analyze_experiment_results',
    'process_result_file',
    'process_result_directory'
] 