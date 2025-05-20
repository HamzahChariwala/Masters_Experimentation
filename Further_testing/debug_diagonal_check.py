import os
import sys
import numpy as np
import json

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the function we want to test
from Agent_Evaluation.AgentTooling.agent_functionality import check_risky_diagonal


def create_test_env_tensor(width, height):
    """Create a simple test environment tensor with a layout of floor cells and some lava"""
    # Initialize with all floor cells
    env_tensor = np.full((width, height), 'floor', dtype=object)
    
    # Add some lava cells for testing
    env_tensor[2, 4] = 'lava'  # Lava at (2,4)
    env_tensor[3, 1] = 'lava'  # Lava at (3,1)
    env_tensor[4, 3] = 'lava'  # Lava at (4,3)
    env_tensor[6, 2] = 'lava'  # Lava at (6,2)
    env_tensor[4, 6] = 'lava'  # Lava at (4,6)
    
    return env_tensor


def run_test_cases():
    """Run various test cases for the check_risky_diagonal function"""
    # Create a test environment
    env_tensor = create_test_env_tensor(10, 10)
    
    # Print the environment for reference
    print("Test Environment:")
    for y in range(8):  # Show a larger section
        row = []
        for x in range(8):
            cell = env_tensor[x, y]
            if cell == 'floor':
                row.append('.')
            elif cell == 'lava':
                row.append('L')
            else:
                row.append(cell[0])
        print(' '.join(row))
    print()
    
    # Test cases
    test_cases = [
        # Basic tests with forward orientation (theta=0, facing right)
        {
            "name": "Single position path, no action",
            "states": [(2, 3, 0)],
            "action_list": [],
            "expected": False
        },
        {
            "name": "Single position path with action",
            "states": [(2, 3, 0)],
            "action_list": [4],  # Diagonal right
            "expected": False  # Should return False as there's no complete path
        },
        {
            "name": "Path with diagonal right through lava (facing right)",
            "states": [(2, 3, 0), (3, 4, 0)],
            "action_list": [4],  # Diagonal right
            "expected": True  # Should detect lava at (2,4)
        },
        {
            "name": "Path with diagonal right, no lava (facing right)",
            "states": [(5, 5, 0), (6, 6, 0)],
            "action_list": [4],  # Diagonal right
            "expected": False
        },
        
        # Tests with different orientations
        {
            "name": "Diagonal right facing down (theta=1)",
            "states": [(3, 2, 1), (2, 3, 1)],
            "action_list": [4],  # Diagonal right when facing down
            "expected": True  # Should detect lava
        },
        {
            "name": "Diagonal left facing down (theta=1)",
            "states": [(3, 4, 1), (4, 5, 1)],
            "action_list": [3],  # Diagonal left when facing down
            "expected": True  # Should detect lava
        },
        {
            "name": "Diagonal right facing left (theta=2)",
            "states": [(5, 3, 2), (4, 2, 2)],
            "action_list": [4],  # Diagonal right when facing left
            "expected": True  # Should detect lava
        },
        {
            "name": "Diagonal left facing left (theta=2)",
            "states": [(5, 5, 2), (4, 6, 2)],
            "action_list": [3],  # Diagonal left when facing left
            "expected": True  # Should detect lava
        },
        {
            "name": "Diagonal right facing up (theta=3)",
            "states": [(5, 3, 3), (6, 2, 3)],
            "action_list": [4],  # Diagonal right when facing up
            "expected": True  # Should detect lava
        },
        {
            "name": "Diagonal left facing up (theta=3)",
            "states": [(3, 5, 3), (2, 4, 3)],
            "action_list": [3],  # Diagonal left when facing up
            "expected": True  # Should detect lava
        },
        
        # Safe paths with different orientations
        {
            "name": "Safe diagonal right facing down (theta=1)",
            "states": [(7, 3, 1), (6, 4, 1)],
            "action_list": [4],
            "expected": False
        },
        {
            "name": "Safe diagonal left facing left (theta=2)",
            "states": [(7, 7, 2), (6, 6, 2)],
            "action_list": [3],
            "expected": False
        }
    ]
    
    # Ensure specific lava cells for the tests
    env_tensor[2, 4] = 'lava'  # For test case 3 - facing right
    env_tensor[3, 3] = 'lava'  # For test case 5 - facing down, diagonal right
    env_tensor[4, 5] = 'lava'  # For test case 6 - facing down, diagonal left
    env_tensor[4, 2] = 'lava'  # For test case 7 - facing left, diagonal right
    env_tensor[4, 6] = 'lava'  # For test case 8 - facing left, diagonal left
    env_tensor[6, 2] = 'lava'  # For test case 9 - facing up, diagonal right
    env_tensor[2, 4] = 'lava'  # For test case 10 - facing up, diagonal left
    
    # Run the tests
    results = {}
    passed_count = 0
    failed_count = 0
    
    for i, test in enumerate(test_cases):
        name = test["name"]
        states = test["states"]
        action_list = test["action_list"]
        expected = test["expected"]
        
        # Run the function
        result = check_risky_diagonal(states, action_list, env_tensor)
        passed = result == expected
        
        if passed:
            passed_count += 1
        else:
            failed_count += 1
        
        print(f"Test {i+1}: {name}")
        print(f"  States: {states}")
        print(f"  Actions: {action_list}")
        print(f"  Expected: {expected}")
        print(f"  Result: {result}")
        print(f"  {'PASSED' if passed else 'FAILED'}")
        print()
        
        results[name] = {
            "states": [list(state) for state in states],  # Convert tuples to lists for JSON
            "action_list": action_list,
            "expected": expected,
            "result": result,
            "passed": passed
        }
    
    # Save results to JSON
    with open("debug_diagonal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Test summary: {passed_count} passed, {failed_count} failed")
    print(f"Saved test results to debug_diagonal_results.json")


if __name__ == "__main__":
    print("Testing check_risky_diagonal function...")
    run_test_cases() 