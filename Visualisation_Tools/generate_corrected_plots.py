#!/usr/bin/env python3
"""
Corrected Plot Generator

Generates the correct plots for monotonic coalition building analysis:
1. Coalition verification plots (main logit progression plots)
2. Metric progression plots (separate raw scores and improvements)
3. Individual environment plots (showing coalition progression per environment)

All plots now correctly show coalition progression instead of per-trial results.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_coalition_verification_plots(agent_path: str, metrics=None):
    """Run the proper coalition verification plots."""
    print("="*60)
    print("GENERATING COALITION VERIFICATION PLOTS")
    print("="*60)
    print("These are the main logit progression plots showing how logits")
    print("change as neurons are added to the coalition.")
    print()
    
    cmd = [
        sys.executable, "-m", 
        "Neuron_Selection.CircuitTooling.MonotonicTooling.create_individual_plots",
        "--agent_path", agent_path
    ]
    
    if metrics:
        cmd.extend(["--metrics"] + metrics)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running coalition verification plots: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    return True


def run_metric_progression_plots(agent_path: str, analysis_type: str, metrics=None):
    """Run the separate metric progression plots."""
    print("="*60)
    print(f"GENERATING METRIC PROGRESSION PLOTS ({analysis_type.upper()})")
    print("="*60)
    print("These show how metric scores change during coalition building.")
    print("Each metric gets separate raw scores and improvements plots.")
    print()
    
    cmd = [
        sys.executable, 
        "Visualisation_Tools/monotonic_metric_progression.py",
        "--agent_path", agent_path,
        "--analysis_type", analysis_type
    ]
    
    if metrics:
        cmd.extend(["--metric"] + metrics)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running metric progression plots: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    return True


def run_individual_environment_plots(agent_path: str, analysis_type: str, metrics=None):
    """Run the individual environment plots showing coalition progression."""
    print("="*60)
    print(f"GENERATING INDIVIDUAL ENVIRONMENT PLOTS ({analysis_type.upper()})")
    print("="*60)
    print("These show coalition progression for each individual environment.")
    print("Each environment gets a separate plot showing logit evolution.")
    print()
    
    cmd = [
        sys.executable, 
        "Visualisation_Tools/individual_environment_plots.py",
        "--agent_path", agent_path,
        "--analysis_type", analysis_type
    ]
    
    if metrics and analysis_type == 'monotonic':
        for metric in metrics:
            metric_cmd = cmd + ["--metric", metric]
            try:
                result = subprocess.run(metric_cmd, check=True, capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print("Warnings/Errors:", result.stderr)
            except subprocess.CalledProcessError as e:
                print(f"Error running individual environment plots for {metric}: {e}")
                print("STDOUT:", e.stdout)
                print("STDERR:", e.stderr)
                return False
    else:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running individual environment plots: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate corrected plots for monotonic coalition analysis")
    parser.add_argument("--agent_path", type=str, required=True,
                       help="Path to the agent directory")
    parser.add_argument("--plots", type=str, choices=['coalition', 'progression', 'individual', 'all'], 
                       default='all', help="Type of plots to generate")
    parser.add_argument("--analysis_type", type=str, choices=['monotonic', 'descending', 'both'], 
                       default='monotonic', help="Analysis type for progression and individual plots")
    parser.add_argument("--metrics", type=str, nargs="*",
                       help="Specific metrics to process (default: all)")
    
    args = parser.parse_args()
    
    agent_path = Path(args.agent_path)
    
    if not agent_path.exists():
        print(f"Error: Agent path does not exist: {agent_path}")
        return
    
    print(f"Generating corrected plots for: {agent_path}")
    print()
    print("FIXED ISSUES:")
    print("- Individual environment plots now show coalition progression (not trials)")
    print("- Metric progression plots are generated separately (not combined)")
    print("- All plots correctly show logit evolution as coalition grows")
    print()
    
    success = True
    
    # Generate coalition verification plots
    if args.plots in ['coalition', 'all']:
        success &= run_coalition_verification_plots(str(agent_path), args.metrics)
        print()
    
    # Generate metric progression plots
    if args.plots in ['progression', 'all']:
        success &= run_metric_progression_plots(str(agent_path), args.analysis_type, args.metrics)
        print()
    
    # Generate individual environment plots
    if args.plots in ['individual', 'all']:
        success &= run_individual_environment_plots(str(agent_path), args.analysis_type, args.metrics)
        print()
    
    if success:
        print("="*60)
        print("PLOT GENERATION COMPLETE!")
        print("="*60)
        print()
        print("Generated plots:")
        if args.plots in ['coalition', 'all']:
            print("1. Coalition verification plots:")
            print("   Location: {agent_path}/circuit_verification/monotonic/{metric}/plots/")
            print("   Files: coalition_verification_{metric}.png")
            print()
        if args.plots in ['progression', 'all']:
            print("2. Metric progression plots:")
            print("   Location: {agent_path}/circuit_verification/monotonic/{metric}/metric_progression_plots/")
            print("   Files: metric_progression_{metric}_raw_scores.png")
            print("           metric_progression_{metric}_improvements.png")
            print()
        if args.plots in ['individual', 'all']:
            print("3. Individual environment plots:")
            print("   Location: {agent_path}/circuit_verification/monotonic/{metric}/individual_environment_plots/")
            print("   Files: environment_{env_name}_coalition_progression.png")
            print()
        print("All plots now correctly show coalition progression instead of per-trial results.")
    else:
        print("="*60)
        print("PLOT GENERATION FAILED!")
        print("="*60)
        print("Check the error messages above for details.")


if __name__ == "__main__":
    main() 