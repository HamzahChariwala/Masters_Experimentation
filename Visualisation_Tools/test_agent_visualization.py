#!/usr/bin/env python3
"""
Test script for the agent visualization function.
"""

from env_plots import plot_lavacrossing_with_agent


def test_agent_visualization():
    """Test the agent visualization with different positions and orientations."""
    
    # Test with seed 81102 at different positions and orientations
    test_seed = 81102
    
    # Test cases: (x, y, theta, description)
    test_cases = [
        (5, 5, 0, "center facing North"),
        (3, 7, 1, "left side facing East"), 
        (8, 3, 2, "right side facing South"),
        (2, 8, 3, "bottom facing West"),
        (1, 1, 1, "near start facing East")
    ]
    
    print(f"Testing agent visualization with seed {test_seed}")
    print()
    
    for i, (x, y, theta, description) in enumerate(test_cases):
        print(f"Test {i+1}/5: Agent at ({x},{y}) theta={theta} ({description})")
        plot_lavacrossing_with_agent(test_seed, x, y, theta)
        print()
    
    print("All agent visualization tests completed!")
    print("Check the partial/ folder for the generated visualizations.")


def generate_additional_visualizations():
    """Generate more interesting agent visualizations."""
    
    test_seed = 81102
    
    # Additional interesting test cases
    additional_cases = [
        (1, 9, 0, "bottom left corner facing North"),
        (9, 1, 2, "top right corner facing South"),
        (9, 9, 3, "bottom right corner facing West"),
        (1, 5, 0, "left edge center facing North"),
        (9, 5, 2, "right edge center facing South"),
        (5, 1, 1, "top edge center facing East"),
        (5, 9, 3, "bottom edge center facing West"),
        (7, 7, 1, "near bottom right facing East"),
        (3, 3, 3, "upper left area facing West"),
        (6, 2, 2, "upper right area facing South"),
        (2, 6, 0, "lower left area facing North"),
        (8, 8, 1, "bottom right area facing East")
    ]
    
    print(f"Generating additional interesting visualizations with seed {test_seed}")
    print()
    
    for i, (x, y, theta, description) in enumerate(additional_cases):
        print(f"Additional {i+1}/12: Agent at ({x},{y}) theta={theta} ({description})")
        plot_lavacrossing_with_agent(test_seed, x, y, theta)
        print()
    
    print("All additional visualizations completed!")
    print("Check the partial/ folder for all generated visualizations.")


if __name__ == "__main__":
    test_agent_visualization()
    print("\n" + "="*60 + "\n")
    generate_additional_visualizations() 