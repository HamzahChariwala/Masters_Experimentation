"""
Test script to demonstrate ASCII-based spawn distribution visualizations.
"""

# Define example parameters for each distribution type
test_params = [
    {
        "name": "Uniform Distribution",
        "type": "uniform"
    },
    {
        "name": "Poisson Distribution (near goal)",
        "type": "poisson_goal",
        "favor_near": True 
    },
    {
        "name": "Poisson Distribution (far from goal)",
        "type": "poisson_goal",
        "favor_near": False
    },
    {
        "name": "Gaussian Distribution (near goal)",
        "type": "gaussian_goal",
        "favor_near": True
    },
    {
        "name": "Gaussian Distribution (far from goal)",
        "type": "gaussian_goal",
        "favor_near": False
    },
    {
        "name": "Distance-based Distribution (near goal)",
        "type": "distance_goal",
        "favor_near": True
    },
    {
        "name": "Distance-based Distribution (far from goal)",
        "type": "distance_goal",
        "favor_near": False
    },
]

def print_simple_distribution_visual(dist_type, favor_near=True):
    """Print a simple ASCII visualization of the distribution."""
    size = 11  # Grid size (odd number to have a center)
    center = size // 2
    
    print("\n    Distribution Pattern (G = goal):")
    
    # Create a grid with empty spaces
    grid = [[' ' for _ in range(size)] for _ in range(size)]
    
    # Place a G in the center to represent the goal
    grid[center][center] = 'G'
    
    # Define a set of characters to represent density
    characters = ' .,:;+*#@'  # From low to high density
    
    # Fill the grid with density indicators
    for y in range(size):
        for x in range(size):
            if x == center and y == center:
                continue  # Skip the goal position
                
            # Calculate Manhattan distance from goal
            distance = abs(x - center) + abs(y - center)
            max_distance = 2 * center
            
            # Normalized distance [0, 1]
            norm_distance = distance / max_distance
            
            # Calculate density based on distribution type and favor_near
            density = 0
            if dist_type == "uniform":
                density = 0.5  # Uniform density
            elif favor_near:
                # Higher density near the goal
                if dist_type == "poisson_goal":
                    density = max(0, 1 - norm_distance * 1.5)
                elif dist_type == "gaussian_goal":
                    density = max(0, 1 - (norm_distance * 1.2) ** 2)
                elif dist_type == "distance_goal":
                    density = max(0, 1 - norm_distance ** 1.5)
            else:
                # Higher density far from the goal
                if dist_type == "poisson_goal":
                    density = min(1, norm_distance * 1.5)
                elif dist_type == "gaussian_goal":
                    density = min(1, (norm_distance * 1.2) ** 2)
                elif dist_type == "distance_goal":
                    density = min(1, norm_distance ** 1.5)
            
            # Convert density to character (skip the goal position)
            if not (x == center and y == center):
                # Map density [0,1] to character index
                char_idx = min(len(characters) - 1, int(density * len(characters)))
                grid[y][x] = characters[char_idx]
    
    # Print the grid
    for row in grid:
        print("    " + ''.join(row))
    print()

def print_transition_timeline(rate=1.0, total_timesteps=1_000_000):
    """Print a simple timeline showing the continuous transition."""
    width = 50  # Width of the timeline
    print("\n    Transition Timeline:")
    print("    Start " + "-" * width + " End")
    
    # Calculate marker positions for different percentages
    markers = []
    for pct in [0.25, 0.5, 0.75]:
        # Apply the rate function to calculate the actual position
        # rate > 1 means slow start, fast finish
        # rate < 1 means fast start, slow finish
        position = int(width * (pct ** rate))
        markers.append((position, f"{int(pct * 100)}%"))
    
    # Print the markers
    marker_line = "         "
    for pos, label in markers:
        space = " " * (pos - len(marker_line))
        marker_line += space + "↓"
    print(marker_line)
    
    # Print the percentages
    pct_line = "         "
    for pos, label in markers:
        space = " " * (pos - len(pct_line))
        pct_line += space + label
    print(pct_line)
    print()

def main():
    print("\n==== SPAWN DISTRIBUTION ASCII VISUALIZATIONS ====")
    print("Characters represent probability density (empty to @, low to high)")
    print("G represents the goal position\n")
    
    # Test each distribution type
    for params in test_params:
        print(f"Distribution: {params['name']}")
        print_simple_distribution_visual(params["type"], params.get("favor_near", True))
        print("-" * 50)
    
    # Print some timeline examples
    print("\nContinuous Transition Timeline Examples:")
    
    print("\nLinear transition (rate=1.0):")
    print_transition_timeline(rate=1.0)
    
    print("\nSlow start, fast finish (rate=2.0):")
    print_transition_timeline(rate=2.0)
    
    print("\nFast start, slow finish (rate=0.5):")
    print_transition_timeline(rate=0.5)
    
    print("\n=============================================")

if __name__ == "__main__":
    main() 