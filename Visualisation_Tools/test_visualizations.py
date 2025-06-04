#!/usr/bin/env python3
"""
Test script for the custom lavacrossing environment visualization.
"""

from env_plots import plot_lavacrossing_environment


def test_visualization():
    """Test the visualization function with different seeds."""
    
    # Test with the requested seed
    print("Testing with seed 81102...")
    plot_lavacrossing_environment(81102)
    
    # Test with a few additional seeds to show variety
    print("\nTesting with additional seeds for variety...")
    test_seeds = [81103, 81104, 12345]
    
    for seed in test_seeds:
        print(f"Generating visualization for seed {seed}...")
        plot_lavacrossing_environment(seed)
    
    print("\nAll visualizations completed!")
    print("Check the Agent_Evaluation/EnvVisualisations/ folder for the generated plots.")


if __name__ == "__main__":
    test_visualization() 