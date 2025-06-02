#!/usr/bin/env python3
"""
Initial gradient analysis test script.
Runs the complete pipeline: gradients -> average -> extract.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the InitialGradients subdirectory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
initial_gradients_dir = os.path.join(script_dir, "InitialGradients")
sys.path.insert(0, initial_gradients_dir)

# Import functions from the three scripts
from InitialGradients.gradients import compute_gradients_for_agent, save_gradients
from InitialGradients.average import compute_average_gradients, save_average_gradients
from InitialGradients.extract import extract_candidate_weights, save_candidate_weights


def run_initial_gradient_analysis(agent_path: str,
                                 useful_neurons_path: str = "Neuron_Selection/ExperimentTooling/Definitions/useful_neurons.json",
                                 k: int = 2, m: int = 4, device: str = "cpu",
                                 output_dir: str = None) -> str:
    """
    Run the complete initial gradient analysis pipeline.
    
    Args:
        agent_path: Path to the agent directory
        useful_neurons_path: Path to useful_neurons.json
        k: Top k weights per neuron
        m: Additional weights to extract globally
        device: Device to run on
        output_dir: Output directory (defaults to agent_path/gradient_analysis)
        
    Returns:
        Path to the final candidate weights file
    """
    # Derive cross_metric_path from agent_path
    cross_metric_path = os.path.join(agent_path, "patching_results", "cross_metric_summary.json")
    
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(agent_path, "gradient_analysis")
    
    print("="*60)
    print("INITIAL GRADIENT ANALYSIS PIPELINE")
    print("="*60)
    print(f"Agent: {agent_path}")
    print(f"Cross metric: {cross_metric_path}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: k={k}, m={m}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Compute gradients
    print("STEP 1: Computing gradients...")
    print("-" * 30)
    gradients_results = compute_gradients_for_agent(agent_path, device=device)
    gradients_path = os.path.join(output_dir, "weight_gradients.json")
    save_gradients(gradients_results, gradients_path)
    print(f"✓ Step 1 complete: {len(gradients_results)} inputs processed")
    print()
    
    # Step 2: Compute average gradients
    print("STEP 2: Computing average gradients...")
    print("-" * 30)
    average_results = compute_average_gradients(gradients_path)
    average_path = os.path.join(output_dir, "average_gradients.json")
    save_average_gradients(average_results, average_path)
    print(f"✓ Step 2 complete: Averaged {len(average_results['average_gradients'])} layers")
    print()
    
    # Step 3: Extract candidate weights
    print("STEP 3: Extracting candidate weights...")
    print("-" * 30)
    extract_results = extract_candidate_weights(
        cross_metric_path, average_path, useful_neurons_path, k, m
    )
    candidate_path = os.path.join(output_dir, "candidate_weights.json")
    save_candidate_weights(extract_results, candidate_path)
    print(f"✓ Step 3 complete: {extract_results['metadata']['total_weights_selected']} weights selected")
    print()
    
    # Final summary
    print("="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("Generated files:")
    print(f"  1. {gradients_path}")
    print(f"  2. {average_path}")
    print(f"  3. {candidate_path}")
    print()
    print("Final results:")
    print(f"  • Processed {len(gradients_results)} input examples")
    print(f"  • Analyzed {len(average_results['average_gradients'])} parameter layers")
    print(f"  • Identified {extract_results['metadata']['num_candidate_neurons']} candidate neurons")
    print(f"  • Selected {extract_results['metadata']['total_weights_selected']} target weights")
    print(f"    - {extract_results['metadata']['guaranteed_per_neuron']} guaranteed (top {k} per neuron)")
    print(f"    - {extract_results['metadata']['additional_selected']} additional (top {m} globally)")
    print()
    print(f"Target weights ready for perturbation experiments!")
    
    return candidate_path


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Run initial gradient analysis pipeline")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--useful_neurons_path", type=str, 
                       default="Neuron_Selection/ExperimentTooling/Definitions/useful_neurons.json",
                       help="Path to useful_neurons.json")
    parser.add_argument("--k", type=int, default=2,
                       help="Top k weights per neuron (default: 2)")
    parser.add_argument("--m", type=int, default=4,
                       help="Additional weights to extract globally (default: 4)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on (default: cpu)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: {agent_path}/gradient_analysis)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.agent_path):
        print(f"Error: Agent path does not exist: {args.agent_path}")
        sys.exit(1)
    
    # Check for cross_metric_summary.json in agent's patching_results directory
    cross_metric_path = os.path.join(args.agent_path, "patching_results", "cross_metric_summary.json")
    if not os.path.exists(cross_metric_path):
        print(f"Error: Cross metric file does not exist: {cross_metric_path}")
        print(f"Expected location: {{agent_path}}/patching_results/cross_metric_summary.json")
        sys.exit(1)
    
    if not os.path.exists(args.useful_neurons_path):
        print(f"Error: Useful neurons file does not exist: {args.useful_neurons_path}")
        sys.exit(1)
    
    # Run the pipeline
    try:
        candidate_path = run_initial_gradient_analysis(
            agent_path=args.agent_path,
            useful_neurons_path=args.useful_neurons_path,
            k=args.k,
            m=args.m,
            device=args.device,
            output_dir=args.output_dir
        )
        print(f"SUCCESS: Analysis complete. Results saved to: {candidate_path}")
        
    except Exception as e:
        print(f"ERROR: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 