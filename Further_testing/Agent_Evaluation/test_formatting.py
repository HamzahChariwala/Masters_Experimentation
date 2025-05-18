"""
Test script to verify the JSON formatting for barrier_mask and lava_mask arrays.
"""

import os
import sys
import json
import glob
import numpy as np
import re

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from Agent_Evaluation.SummaryTooling import format_json_with_compact_arrays

def test_mask_formatting():
    """
    Test the formatting of barrier_mask and lava_mask arrays in JSON output.
    """
    print("Testing mask formatting in JSON output...")
    
    # Create a simple test data structure with layout, barrier_mask, and lava_mask
    # Include multiple states to ensure consistent formatting across all instances
    test_data = {
        "environment": {
            "layout": [
                ["wall", "wall", "wall", "wall"],
                ["wall", "empty", "empty", "wall"],
                ["wall", "empty", "goal", "wall"],
                ["wall", "wall", "wall", "wall"]
            ],
            "1,1,0": {
                "model_inputs": {
                    "barrier_mask": [
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1],
                        [0, 1, 1, 0]
                    ],
                    "lava_mask": [
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0]
                    ]
                }
            },
            "1,1,1": {
                "model_inputs": {
                    "barrier_mask": [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ],
                    "lava_mask": [
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0]
                    ]
                }
            },
            "1,2,0": {
                "model_inputs": {
                    "barrier_mask": [
                        [1, 1, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 1, 1, 1]
                    ],
                    "lava_mask": [
                        [0, 0, 0, 0],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [0, 0, 0, 0]
                    ]
                }
            }
        }
    }
    
    # Format the test data
    formatted_json = format_json_with_compact_arrays(test_data)
    
    # Print the formatted JSON
    print("\nFormatted JSON output:")
    print(formatted_json)
    
    # Save the formatted JSON to a file for inspection
    output_file = os.path.join(os.path.dirname(__file__), "test_formatting_output.json")
    with open(output_file, 'w') as f:
        f.write(formatted_json)
    
    print(f"\nSaved formatted JSON to: {output_file}")
    
    # Load the formatted JSON back to verify structure
    with open(output_file, 'r') as f:
        loaded_data = json.load(f)
    
    # Verify that the structure is preserved for all states
    print("\nVerification results:")
    
    # Check layout
    layout = loaded_data["environment"]["layout"]
    layout_ok = len(layout) == 4 and len(layout[0]) == 4
    print(f"- Layout dimensions: {len(layout)}x{len(layout[0])}")
    print(f"- Layout OK: {layout_ok}")
    
    # Check all states for consistent formatting
    states = ["1,1,0", "1,1,1", "1,2,0"]
    all_masks_ok = True
    
    for state in states:
        barrier_mask = loaded_data["environment"][state]["model_inputs"]["barrier_mask"]
        lava_mask = loaded_data["environment"][state]["model_inputs"]["lava_mask"]
        
        barrier_mask_ok = len(barrier_mask) == 4 and len(barrier_mask[0]) == 4
        lava_mask_ok = len(lava_mask) == 4 and len(lava_mask[0]) == 4
        
        # If any mask is malformed, set all_masks_ok to False
        if not barrier_mask_ok or not lava_mask_ok:
            all_masks_ok = False
        
        print(f"\nState {state}:")
        print(f"- Barrier mask dimensions: {len(barrier_mask)}x{len(barrier_mask[0])}")
        print(f"- Lava mask dimensions: {len(lava_mask)}x{len(lava_mask[0])}")
        print(f"- Barrier mask OK: {barrier_mask_ok}")
        print(f"- Lava mask OK: {lava_mask_ok}")
    
    # Check if json formatting looks correct - each mask should have multi-line formatting
    with open(output_file, 'r') as f:
        json_text = f.read()
    
    # Count occurrences of formatted barrier_mask and lava_mask arrays
    barrier_pattern = r'"barrier_mask":\s*\[\n\s+\['
    lava_pattern = r'"lava_mask":\s*\[\n\s+\['
    
    barrier_matches = len(re.findall(barrier_pattern, json_text))
    lava_matches = len(re.findall(lava_pattern, json_text))
    
    # We expect 3 states, so 3 of each mask type
    barrier_format_ok = barrier_matches == 3
    lava_format_ok = lava_matches == 3
    
    print(f"\nFormatting check:")
    print(f"- Barrier mask formatted arrays: {barrier_matches}/3 expected")
    print(f"- Lava mask formatted arrays: {lava_matches}/3 expected")
    print(f"- Barrier format OK: {barrier_format_ok}")
    print(f"- Lava format OK: {lava_format_ok}")
    
    # Overall result
    all_ok = layout_ok and all_masks_ok and barrier_format_ok and lava_format_ok
    if all_ok:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
    
    return all_ok

if __name__ == "__main__":
    test_mask_formatting() 