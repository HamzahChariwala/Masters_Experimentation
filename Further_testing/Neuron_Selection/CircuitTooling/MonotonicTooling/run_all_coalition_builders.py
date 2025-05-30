#!/usr/bin/env python3
"""
Run Monotonic Coalition Builder for All Metrics

This script runs the monotonic coalition building algorithm for all 7 metrics separately,
building a separate coalition for each metric that optimizes only that specific metric.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import subprocess
import time

# Add the Neuron_Selection directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
neuron_selection_dir = os.path.abspath(os.path.join(script_dir, "../.."))
project_root = os.path.abspath(os.path.join(neuron_selection_dir, ".."))

# Add both directories to path if they're not already there
for path in [neuron_selection_dir, project_root]:
    if not path in sys.path:
        sys.path.insert(0, path)

# Available metrics for coalition building
ALL_METRICS = [
    "kl_divergence",
    "reverse_kl_divergence", 
    "undirected_saturating_chebyshev",
    "reversed_undirected_saturating_chebyshev",
    "confidence_margin_magnitude",
    "reversed_pearson_correlation",
    "top_logit_delta_magnitude"
]

def run_coalition_builder_for_metric(agent_path: str, metric: str, candidate_pool_size: int,
                                   max_coalition_size: int, highest: bool, input_ids: Optional[str],
                                   device: str) -> bool:
    """
    Run the monotonic coalition builder for a single metric.
    
    Args:
        agent_path: Path to the agent directory
        metric: Target metric to optimize for
        candidate_pool_size: Number of candidate partners to consider each iteration
        max_coalition_size: Maximum coalition size to build
        highest: Whether to use highest values when combining noising/denoising scores
        input_ids: Comma-separated list of input IDs to process
        device: Device to run experiments on
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"BUILDING COALITION FOR METRIC: {metric}")
    print(f"{'='*60}")
    
    # Construct command
    cmd = [
        sys.executable, "-m", "Neuron_Selection.CircuitTooling.MonotonicTooling.monotonic_coalition_builder",
        "--agent_path", agent_path,
        "--metric", metric,
        "--candidate_pool_size", str(candidate_pool_size),
        "--max_coalition_size", str(max_coalition_size),
        "--highest", "true" if highest else "false",
        "--device", device
    ]
    
    if input_ids:
        cmd.extend(["--input_ids", input_ids])
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        print(f"SUCCESS: Coalition building for {metric} completed in {end_time - start_time:.1f} seconds")
        print("STDOUT:", result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Coalition building for {metric} failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error for {metric}: {e}")
        return False

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run monotonic coalition builder for all 7 metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script builds separate coalitions for each of the 7 metrics:
{}

Each coalition is optimized ONLY for its specific metric.

Example usage:
  python -m Neuron_Selection.CircuitTooling.MonotonicTooling.run_all_coalition_builders \\
    --agent_path "Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6" \\
    --candidate_pool_size 20 \\
    --max_coalition_size 30 \\
    --highest true
        """.format("\n".join(f"  - {metric}" for metric in ALL_METRICS))
    )
    
    # Required arguments
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    
    # Optional arguments
    parser.add_argument("--candidate_pool_size", type=int, default=50,
                       help="Number of candidate partners to consider each iteration (default: 50)")
    parser.add_argument("--max_coalition_size", type=int, default=30,
                       help="Maximum coalition size to build (default: 30)")
    parser.add_argument("--highest", type=str, default="true", choices=["true", "false"],
                       help="Use highest values when combining noising/denoising scores (default: true)")
    parser.add_argument("--input_ids", type=str, default=None,
                       help="Comma-separated list of input IDs to process (default: all)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run experiments on (default: cpu)")
    parser.add_argument("--metrics", type=str, default=None,
                       help="Comma-separated list of metrics to run (default: all 7 metrics)")
    
    args = parser.parse_args()
    
    # Parse arguments
    highest = args.highest.lower() == "true"
    metrics_to_run = args.metrics.split(",") if args.metrics else ALL_METRICS
    
    # Validate metrics
    invalid_metrics = [m for m in metrics_to_run if m not in ALL_METRICS]
    if invalid_metrics:
        print(f"Error: Invalid metrics: {invalid_metrics}")
        print(f"Available metrics: {ALL_METRICS}")
        sys.exit(1)
    
    print(f"Running coalition builder for {len(metrics_to_run)} metrics:")
    for metric in metrics_to_run:
        print(f"  - {metric}")
    print(f"\nAgent path: {args.agent_path}")
    print(f"Coalition parameters:")
    print(f"  - Candidate pool size: {args.candidate_pool_size}")
    print(f"  - Max coalition size: {args.max_coalition_size}")
    print(f"  - Highest mode: {highest}")
    print(f"  - Device: {args.device}")
    if args.input_ids:
        print(f"  - Input IDs: {args.input_ids}")
    
    # Run coalition builder for each metric
    successful_metrics = []
    failed_metrics = []
    overall_start_time = time.time()
    
    for metric in metrics_to_run:
        success = run_coalition_builder_for_metric(
            agent_path=args.agent_path,
            metric=metric,
            candidate_pool_size=args.candidate_pool_size,
            max_coalition_size=args.max_coalition_size,
            highest=highest,
            input_ids=args.input_ids,
            device=args.device
        )
        
        if success:
            successful_metrics.append(metric)
        else:
            failed_metrics.append(metric)
    
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Successful coalitions: {len(successful_metrics)}/{len(metrics_to_run)}")
    
    if successful_metrics:
        print("\nSuccessful metrics:")
        for metric in successful_metrics:
            print(f"  ✓ {metric}")
    
    if failed_metrics:
        print("\nFailed metrics:")
        for metric in failed_metrics:
            print(f"  ✗ {metric}")
        sys.exit(1)
    else:
        print("\nAll coalition builders completed successfully!")
        
        # Generate visualizations automatically if all coalitions were successful
        print(f"\n{'='*60}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        try:
            # Import and run the individual plots script
            cmd = [
                sys.executable, "-m", "Neuron_Selection.CircuitTooling.MonotonicTooling.create_individual_plots",
                "--agent_path", args.agent_path
            ]
            
            if args.metrics:
                cmd.extend(["--metrics"] + metrics_to_run)
            
            start_time = time.time()
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            end_time = time.time()
            
            print(f"✓ Visualizations generated successfully in {end_time - start_time:.1f} seconds!")
            print("STDOUT:", result.stdout)
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error generating visualizations!")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            print("You can manually generate them with:")
            print(f"python -m Neuron_Selection.CircuitTooling.MonotonicTooling.create_individual_plots --agent_path \"{args.agent_path}\"")
        except Exception as e:
            print(f"✗ Unexpected error generating visualizations: {e}")
            print("You can manually generate them with:")
            print(f"python -m Neuron_Selection.CircuitTooling.MonotonicTooling.create_individual_plots --agent_path \"{args.agent_path}\"")

if __name__ == "__main__":
    main() 