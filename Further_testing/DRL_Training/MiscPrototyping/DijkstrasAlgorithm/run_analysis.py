"""
Run Dijkstra's Algorithm Analysis on MiniGrid Environments

This script analyzes multiple MiniGrid environments using Dijkstra's algorithm
to find optimal paths from every cell and orientation to the goal.
"""

import os
import argparse
from minigrid_graph import generate_optimal_paths_for_env

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Dijkstra's algorithm analysis on MiniGrid environments")
    
    parser.add_argument('--envs', nargs='+', default=['MiniGrid-Empty-8x8-v0', 'MiniGrid-LavaCrossingS9N1-v0'],
                        help='List of environments to analyze')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 42, 12345],
                        help='List of seeds to use for each environment')
    parser.add_argument('--lava-modes', nargs='+', default=['normal', 'costly', 'blocked'],
                        help='List of lava modes to analyze')
    parser.add_argument('--output-dir', default='./results',
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis for each environment and seed
    results = {}
    for env_id in args.envs:
        env_results = {}
        for seed in args.seeds:
            print(f"Analyzing {env_id} with seed {seed}...")
            
            # Generate optimal paths for this environment and seed
            seed_results = generate_optimal_paths_for_env(
                env_id=env_id,
                seed=seed,
                lava_modes=args.lava_modes,
                output_dir=args.output_dir,
                visualize=args.visualize
            )
            
            env_results[seed] = seed_results
        
        results[env_id] = env_results
    
    print("\nAnalysis complete!")
    print(f"Results saved to {os.path.abspath(args.output_dir)}")
    
    # Print summary
    print("\nSummary:")
    for env_id, env_results in results.items():
        print(f"  {env_id}:")
        for seed, seed_results in env_results.items():
            print(f"    Seed {seed}:")
            for lava_mode, mode_results in seed_results.items():
                # Get the first (and only) key
                env_key = list(mode_results.keys())[0]
                # Count the number of cells analyzed
                height = len(mode_results[env_key]["grid"])
                width = len(mode_results[env_key]["grid"][0]) if height > 0 else 0
                total_cells = height * width
                num_walls = len(mode_results[env_key]["walls"])
                print(f"      {lava_mode}: {total_cells} total cells, {num_walls} walls")

if __name__ == "__main__":
    main() 