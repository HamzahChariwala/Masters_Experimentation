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
    top_action_probability_gap,
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
    'top_action_probability_gap',
    'METRIC_FUNCTIONS',
    
    # Result processing
    'load_result_file',
    'save_result_file',
    'analyze_experiment_results',
    'process_result_file',
    'process_result_directory'
] 