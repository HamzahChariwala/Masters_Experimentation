#!/usr/bin/env python3
"""
Monotonic Coalition Builder for Neuron Circuit Analysis

This script implements a monotonic coalition building algorithm that iteratively
builds coalitions of neurons starting from the highest-scoring individual neuron
and progressively adding neurons that maximize disruption to the current coalition's output.

For each metric, the algorithm:
1. Starts with the highest-scoring neuron from filtered results
2. Iteratively tests the next m highest-scoring neurons as potential partners
3. Selects the partner that maximizes the target metric relative to current coalition
4. Adds the selected partner to the coalition
5. Repeats until coalition reaches maximum size k

The algorithm measures metrics against the current coalition's logits, not the original logits,
ensuring monotonic improvement in the target metric.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from datetime import datetime

# Add the Neuron_Selection directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
neuron_selection_dir = os.path.abspath(os.path.join(script_dir, "../.."))
project_root = os.path.abspath(os.path.join(neuron_selection_dir, ".."))

# Add both directories to path if they're not already there
for path in [neuron_selection_dir, project_root]:
    if not path in sys.path:
        sys.path.insert(0, path)

from Neuron_Selection.PatchingTooling.patching_experiment import PatchingExperiment
from Neuron_Selection.AnalysisTooling.metrics import METRIC_FUNCTIONS

# Default parameters
DEFAULT_CANDIDATE_POOL_SIZE = 20
DEFAULT_MAX_COALITION_SIZE = 30
DEFAULT_HIGHEST = True
DEFAULT_INPUT_IDS = None  # Process all available input IDs by default

# Available metrics for coalition building
AVAILABLE_METRICS = [
    "kl_divergence",
    "reverse_kl_divergence", 
    "undirected_saturating_chebyshev",
    "reversed_undirected_saturating_chebyshev",
    "confidence_margin_magnitude",
    "reversed_pearson_correlation",
    "top_logit_delta_magnitude"
]

class MonotonicCoalitionBuilder:
    """
    Implements the monotonic coalition building algorithm for neuron circuit analysis.
    """
    
    def __init__(self, agent_path: str, metric: str, candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
                 max_coalition_size: int = DEFAULT_MAX_COALITION_SIZE, highest: bool = DEFAULT_HIGHEST,
                 input_ids: Optional[List[str]] = None, device: str = "cpu"):
        """
        Initialize the monotonic coalition builder.
        
        Args:
            agent_path: Path to the agent directory
            metric: Target metric to optimize for
            candidate_pool_size: Number of candidate partners to consider each iteration
            max_coalition_size: Maximum coalition size to build
            highest: Whether to use highest values when combining noising/denoising scores
            input_ids: List of input IDs to process (None for all)
            device: Device to run experiments on
        """
        self.agent_path = Path(agent_path)
        self.metric = metric
        self.candidate_pool_size = candidate_pool_size
        self.max_coalition_size = max_coalition_size
        self.highest = highest
        self.input_ids = input_ids
        self.device = device
        
        # Validate metric
        if metric not in AVAILABLE_METRICS:
            raise ValueError(f"Invalid metric: {metric}. Available: {AVAILABLE_METRICS}")
        
        # Setup output directory
        self.output_dir = self.agent_path / "circuit_verification" / "monotonic" / self.metric
        self.experiments_dir = self.output_dir / "experiments"
        self.results_dir = self.output_dir / "results"
        self.plots_dir = self.output_dir / "plots"
        
        # Create output directories
        for dir_path in [self.experiments_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize patching experiment
        self.patching_experiment = PatchingExperiment(str(self.agent_path), device=self.device)
        
        # Load filtered results for the metric
        self.filtered_neurons = self._load_filtered_neurons()
        
        # Initialize tracking variables
        self.coalition = []  # List of (neuron_name, selection_info) tuples
        self.iteration_results = []  # Results for each iteration
        self.total_experiments = 0
        
        print(f"Initialized MonotonicCoalitionBuilder:")
        print(f"  Agent: {self.agent_path}")
        print(f"  Metric: {self.metric}")
        print(f"  Pool size: {self.candidate_pool_size}")
        print(f"  Max coalition size: {self.max_coalition_size}")
        print(f"  Highest mode: {self.highest}")
        print(f"  Available neurons: {len(self.filtered_neurons)}")
    
    def _load_filtered_neurons(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Load filtered neurons for the target metric from the filtered results file.
        
        Returns:
            List of (neuron_name, neuron_data) tuples sorted by importance
        """
        filtered_file = self.agent_path / "patching_results" / "filtered" / f"filtered_{self.metric}.json"
        
        if not filtered_file.exists():
            raise FileNotFoundError(f"Filtered results file not found: {filtered_file}")
        
        with open(filtered_file, 'r') as f:
            filtered_data = json.load(f)
        
        # Extract neurons from the 'averaged' section (should be pre-sorted)
        if 'averaged' not in filtered_data:
            raise KeyError(f"No 'averaged' section found in filtered results for {self.metric}")
        
        averaged_section = filtered_data['averaged']
        neurons = [(neuron_name, neuron_data) for neuron_name, neuron_data in averaged_section.items()]
        
        print(f"Loaded {len(neurons)} filtered neurons for metric '{self.metric}'")
        if neurons:
            print(f"Top neuron: {neurons[0][0]} (score: {neurons[0][1].get('averaged_normalized_value', 'N/A')})")
        
        return neurons
    
    def _create_experiment_config(self, neurons: List[str]) -> Dict[str, List[int]]:
        """
        Create a patch configuration for the given list of neurons.
        
        Args:
            neurons: List of neuron names (e.g., ["q_net.2_neuron_3", "q_net.0_neuron_14"])
            
        Returns:
            Dictionary mapping layer names to neuron indices
        """
        config = {}
        
        for neuron_name in neurons:
            # Parse neuron name to extract layer and index
            # Expected format: "layer_name_neuron_index"
            if "_neuron_" not in neuron_name:
                print(f"Warning: Unexpected neuron name format: {neuron_name}")
                continue
            
            layer_name, neuron_index_str = neuron_name.rsplit("_neuron_", 1)
            try:
                neuron_index = int(neuron_index_str)
                
                if layer_name not in config:
                    config[layer_name] = []
                config[layer_name].append(neuron_index)
            except ValueError:
                print(f"Warning: Could not parse neuron index from: {neuron_name}")
                continue
        
        return config
    
    def _run_coalition_experiment(self, coalition_neurons: List[str], iteration: int, 
                                experiment_name: str) -> Dict[str, Any]:
        """
        Run a patching experiment for the given coalition of neurons.
        
        Args:
            coalition_neurons: List of neuron names in the coalition
            iteration: Current iteration number
            experiment_name: Name identifier for this experiment
            
        Returns:
            Dictionary containing experiment results and metrics
        """
        self.total_experiments += 1
        
        # Create patch configuration
        patch_config = self._create_experiment_config(coalition_neurons)
        
        # Save experiment configuration
        experiment_config = {
            "iteration": iteration,
            "experiment_name": experiment_name,
            "coalition_neurons": coalition_neurons,
            "patch_config": patch_config,
            "metric": self.metric,
            "timestamp": datetime.now().isoformat()
        }
        
        experiment_file = self.experiments_dir / f"iteration_{iteration:03d}_{experiment_name}.json"
        with open(experiment_file, 'w') as f:
            json.dump(experiment_config, f, indent=2)
        
        # Run the actual patching experiment
        try:
            # Run noising experiment (clean -> corrupted)
            noising_results = self.patching_experiment.run_patching_experiment(
                target_input_file="corrupted_inputs.json",
                source_activation_file="clean_activations.npz",
                patch_spec=patch_config,
                input_ids=self.input_ids
            )
            
            # Run denoising experiment (corrupted -> clean)
            denoising_results = self.patching_experiment.run_patching_experiment(
                target_input_file="clean_inputs.json", 
                source_activation_file="corrupted_activations.npz",
                patch_spec=patch_config,
                input_ids=self.input_ids
            )
            
            # Calculate metrics for both experiments
            noising_metrics = self._calculate_metrics(noising_results)
            denoising_metrics = self._calculate_metrics(denoising_results)
            
            # Combine results - make JSON serializable by removing non-serializable data
            experiment_results = {
                "experiment_name": experiment_name,
                "coalition_neurons": coalition_neurons,
                "patch_config": patch_config,
                "noising": {
                    "metrics": noising_metrics,
                    "num_results": len(noising_results)
                },
                "denoising": {
                    "metrics": denoising_metrics,
                    "num_results": len(denoising_results)
                },
                "target_metric_score": self._combine_metric_scores(noising_metrics, denoising_metrics),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save detailed results
            results_file = self.results_dir / f"iteration_{iteration:03d}_{experiment_name}.json"
            with open(results_file, 'w') as f:
                json.dump(experiment_results, f, indent=2)
            
            return experiment_results
            
        except Exception as e:
            print(f"Error running experiment {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "experiment_name": experiment_name,
                "coalition_neurons": coalition_neurons,
                "error": str(e),
                "target_metric_score": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_metrics(self, patching_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate the target metric for the patching results.
        
        Args:
            patching_results: Results from patching experiment
            
        Returns:
            Dictionary with calculated metric values
        """
        metric_function = METRIC_FUNCTIONS[self.metric]
        metric_values = []
        
        for input_id, input_results in patching_results.items():
            if "error" in input_results:
                print(f"Skipping input {input_id} due to error: {input_results.get('error', 'Unknown error')}")
                continue
                
            baseline_output = input_results.get("baseline_output")
            patched_output = input_results.get("patched_output")
            
            if baseline_output is None or patched_output is None:
                print(f"Missing output data for input {input_id}")
                continue
            
            try:
                metric_result = metric_function(baseline_output, patched_output)
                
                # Extract scalar value from metric result
                if isinstance(metric_result, dict):
                    # For metrics that return dictionaries, extract the main value
                    if "ratio" in metric_result:
                        metric_value = metric_result["ratio"]
                    elif "magnitude" in metric_result:
                        metric_value = metric_result["magnitude"]
                    elif "reversed_correlation" in metric_result:
                        metric_value = metric_result["reversed_correlation"]
                    else:
                        # Take the first numeric value found
                        for key, value in metric_result.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                metric_value = value
                                break
                        else:
                            print(f"No numeric value found in metric result for {input_id}: {metric_result}")
                            metric_value = 0.0
                else:
                    metric_value = metric_result
                
                if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                    metric_values.append(float(metric_value))
                    print(f"  Input {input_id}: {self.metric} = {metric_value:.6f}")
                else:
                    print(f"Invalid metric value type for {input_id}: {type(metric_value)} = {metric_value}")
                
            except Exception as e:
                print(f"Error calculating metric {self.metric} for input {input_id}: {e}")
                continue
        
        if not metric_values:
            print(f"Warning: No valid metric values calculated for {self.metric}")
            return {"mean": 0.0, "std": 0.0, "count": 0}
        
        mean_val = float(np.mean(metric_values))
        std_val = float(np.std(metric_values))
        
        print(f"Calculated metrics - mean: {mean_val:.6f}, std: {std_val:.6f}, count: {len(metric_values)}")
        
        return {
            "mean": mean_val,
            "std": std_val,
            "count": len(metric_values),
            "values": metric_values
        }
    
    def _combine_metric_scores(self, noising_metrics: Dict[str, float], 
                             denoising_metrics: Dict[str, float]) -> float:
        """
        Combine noising and denoising scores for the TARGET METRIC according to the 'highest' parameter.
        
        Args:
            noising_metrics: Metrics from noising experiment
            denoising_metrics: Metrics from denoising experiment
            
        Returns:
            Target metric score (not a combination of different metrics)
        """
        noising_score = noising_metrics.get("mean", 0.0)
        denoising_score = denoising_metrics.get("mean", 0.0)
        
        if self.highest:
            # Take the maximum of the two scores for the TARGET METRIC
            return max(noising_score, denoising_score)
        else:
            # Take the average of the two scores for the TARGET METRIC
            return (noising_score + denoising_score) / 2.0
    
    def build_coalition(self) -> Dict[str, Any]:
        """
        Build the monotonic coalition using the specified algorithm.
        
        Returns:
            Dictionary containing the final coalition and building summary
        """
        print(f"\nStarting monotonic coalition building for metric: {self.metric}")
        print(f"Target coalition size: {self.max_coalition_size}")
        print(f"Candidate pool size: {self.candidate_pool_size}")
        
        if not self.filtered_neurons:
            raise ValueError("No filtered neurons available for coalition building")
        
        # Step 1: Start with the highest-scoring neuron
        initial_neuron = self.filtered_neurons[0][0]
        print(f"\nIteration 0: Starting with highest-scoring neuron: {initial_neuron}")
        
        # Run experiment for the initial neuron
        initial_results = self._run_coalition_experiment(
            coalition_neurons=[initial_neuron],
            iteration=0,
            experiment_name="initial_neuron"
        )
        
        initial_score = initial_results.get("target_metric_score", 0.0)
        self.coalition.append((initial_neuron, {
            "iteration": 0,
            "score": initial_score,
            "selection_reason": "highest_scoring_individual",
            "experiments_run": 1
        }))
        
        print(f"Initial coalition score: {initial_score:.6f}")
        
        # Step 2: Iteratively add neurons
        available_neurons = [name for name, _ in self.filtered_neurons[1:]]  # Skip the first (already used)
        
        for iteration in range(1, self.max_coalition_size):
            if len(available_neurons) < self.candidate_pool_size:
                print(f"Warning: Only {len(available_neurons)} neurons remaining, less than pool size {self.candidate_pool_size}")
            
            # Select candidate pool
            candidate_pool = available_neurons[:self.candidate_pool_size]
            if not candidate_pool:
                print("No more candidate neurons available. Stopping coalition building.")
                break
            
            print(f"\nIteration {iteration}: Testing {len(candidate_pool)} candidates")
            
            # Test each candidate with the current coalition
            candidate_scores = []
            current_coalition_neurons = [neuron for neuron, _ in self.coalition]
            
            for candidate in candidate_pool:
                test_coalition = current_coalition_neurons + [candidate]
                candidate_results = self._run_coalition_experiment(
                    coalition_neurons=test_coalition,
                    iteration=iteration,
                    experiment_name=f"candidate_{candidate.replace('.', '_')}"
                )
                
                candidate_score = candidate_results.get("target_metric_score", 0.0)
                candidate_scores.append((candidate, candidate_score, candidate_results))
                
                print(f"  Candidate {candidate}: score = {candidate_score:.6f}")
            
            # Select the best candidate
            if not candidate_scores:
                print("No valid candidate scores. Stopping coalition building.")
                break
            
            best_candidate, best_score, best_results = max(candidate_scores, key=lambda x: x[1])
            
            # Add the best candidate to the coalition
            self.coalition.append((best_candidate, {
                "iteration": iteration,
                "score": best_score,
                "selection_reason": f"best_of_{len(candidate_pool)}_candidates",
                "experiments_run": len(candidate_pool),
                "improvement": best_score - initial_score if iteration == 1 else best_score - self.coalition[-2][1]["score"]
            }))
            
            # Remove the selected candidate from available neurons
            available_neurons.remove(best_candidate)
            
            print(f"Selected: {best_candidate} (score: {best_score:.6f})")
            print(f"Coalition size: {len(self.coalition)}")
        
        # Generate final summary
        summary = self._generate_summary()
        
        # Save summary
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nCoalition building complete!")
        print(f"Final coalition size: {len(self.coalition)}")
        print(f"Total experiments run: {self.total_experiments}")
        print(f"Summary saved to: {summary_file}")
        
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the coalition building process.
        
        Returns:
            Dictionary containing coalition building summary
        """
        coalition_neurons = [neuron for neuron, _ in self.coalition]
        coalition_scores = [info["score"] for _, info in self.coalition]
        
        summary = {
            "algorithm": "monotonic_coalition_builder",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "agent_path": str(self.agent_path),
                "metric": self.metric,
                "candidate_pool_size": self.candidate_pool_size,
                "max_coalition_size": self.max_coalition_size,
                "highest": self.highest,
                "input_ids": self.input_ids
            },
            "results": {
                "final_coalition_size": len(self.coalition),
                "total_experiments_run": self.total_experiments,
                "final_coalition_neurons": coalition_neurons,
                "coalition_scores": coalition_scores,
                "score_progression": {
                    "initial_score": coalition_scores[0] if coalition_scores else 0.0,
                    "final_score": coalition_scores[-1] if coalition_scores else 0.0,
                    "total_improvement": coalition_scores[-1] - coalition_scores[0] if len(coalition_scores) > 1 else 0.0,
                    "average_improvement_per_neuron": (coalition_scores[-1] - coalition_scores[0]) / (len(coalition_scores) - 1) if len(coalition_scores) > 1 else 0.0
                }
            },
            "coalition_details": [
                {
                    "position": i,
                    "neuron": neuron,
                    "iteration": info["iteration"],
                    "score": info["score"],
                    "selection_reason": info["selection_reason"],
                    "experiments_run": info["experiments_run"],
                    "improvement": info.get("improvement", 0.0)
                }
                for i, (neuron, info) in enumerate(self.coalition)
            ],
            "computational_cost": {
                "total_experiments": self.total_experiments,
                "experiments_per_iteration": [info["experiments_run"] for _, info in self.coalition],
                "average_experiments_per_iteration": np.mean([info["experiments_run"] for _, info in self.coalition]) if self.coalition else 0.0
            }
        }
        
        return summary

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build monotonic coalitions of neurons for circuit analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m Neuron_Selection.CircuitTooling.MonotonicTooling.monotonic_coalition_builder \\
    --agent_path "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6" \\
    --metric "undirected_saturating_chebyshev" \\
    --candidate_pool_size 20 \\
    --max_coalition_size 30 \\
    --highest true

Available metrics:
  {}
        """.format("\n  ".join(AVAILABLE_METRICS))
    )
    
    # Required arguments
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--metric", type=str, required=True, choices=AVAILABLE_METRICS,
                       help="Target metric to optimize for")
    
    # Optional arguments
    parser.add_argument("--candidate_pool_size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE,
                       help=f"Number of candidate partners to consider each iteration (default: {DEFAULT_CANDIDATE_POOL_SIZE})")
    parser.add_argument("--max_coalition_size", type=int, default=DEFAULT_MAX_COALITION_SIZE,
                       help=f"Maximum coalition size to build (default: {DEFAULT_MAX_COALITION_SIZE})")
    parser.add_argument("--highest", type=str, default="true", choices=["true", "false"],
                       help=f"Use highest values when combining noising/denoising scores (default: true)")
    parser.add_argument("--input_ids", type=str, default=None,
                       help="Comma-separated list of input IDs to process (default: all)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run experiments on (default: cpu)")
    
    args = parser.parse_args()
    
    # Parse boolean and list arguments
    highest = args.highest.lower() == "true"
    input_ids = args.input_ids.split(",") if args.input_ids else None
    
    try:
        # Initialize and run coalition builder
        builder = MonotonicCoalitionBuilder(
            agent_path=args.agent_path,
            metric=args.metric,
            candidate_pool_size=args.candidate_pool_size,
            max_coalition_size=args.max_coalition_size,
            highest=highest,
            input_ids=input_ids,
            device=args.device
        )
        
        # Build the coalition
        summary = builder.build_coalition()
        
        print(f"\n{'='*50}")
        print("COALITION BUILDING SUMMARY")
        print(f"{'='*50}")
        print(f"Metric: {args.metric}")
        print(f"Final coalition size: {summary['results']['final_coalition_size']}")
        print(f"Total experiments: {summary['results']['total_experiments_run']}")
        print(f"Initial score: {summary['results']['score_progression']['initial_score']:.6f}")
        print(f"Final score: {summary['results']['score_progression']['final_score']:.6f}")
        print(f"Total improvement: {summary['results']['score_progression']['total_improvement']:.6f}")
        print(f"\nFinal coalition:")
        for i, neuron in enumerate(summary['results']['final_coalition_neurons']):
            score = summary['results']['coalition_scores'][i]
            print(f"  {i+1:2d}. {neuron} (score: {score:.6f})")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 