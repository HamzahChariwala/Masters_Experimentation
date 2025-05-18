"""
Test script to verify the JSON formatting on a real evaluation log file.
"""

import os
import sys
import json
import glob
import re

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from Agent_Evaluation.SummaryTooling import format_json_with_compact_arrays

def test_real_log_formatting():
    """
    Test the formatting of barrier_mask and lava_mask arrays in a real evaluation log.
    """
    print("Testing JSON formatting on real evaluation logs...")
    
    # Search for evaluation logs in the Agent_Storage directory
    agent_storage_dir = os.path.join(project_root, "Agent_Storage")
    log_files = []
    
    # Recursively find all JSON files in subdirectories
    for root, dirs, files in os.walk(agent_storage_dir):
        for file in files:
            if file.endswith(".json") and "evaluation_logs" in root:
                log_files.append(os.path.join(root, file))
    
    if not log_files:
        print("No evaluation log files found. Exiting.")
        return False
    
    # Select the first log file found for testing
    log_file = log_files[0]
    print(f"Using log file: {log_file}")
    
    # Load the log file
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    # Apply our JSON formatter
    formatted_json = format_json_with_compact_arrays(log_data)
    
    # Save the formatted JSON to a file for inspection
    output_file = os.path.join(os.path.dirname(__file__), "real_log_formatted.json")
    with open(output_file, 'w') as f:
        f.write(formatted_json)
    
    print(f"\nSaved formatted JSON to: {output_file}")
    
    # Verify that the formatted JSON is valid
    try:
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        print("✅ Formatted JSON is valid")
    except json.JSONDecodeError as e:
        print(f"❌ Formatted JSON is invalid: {e}")
        return False
    
    # Check if environment structure is preserved
    if "environment" not in loaded_data:
        print("❌ Environment data missing")
        return False
    
    env = loaded_data["environment"]
    
    # Check layout
    if "layout" not in env:
        print("❌ Layout data missing")
        return False
    
    print("✅ Layout preserved")
    
    # Check barrier_mask and lava_mask in state model_inputs
    barrier_mask_count = 0
    lava_mask_count = 0
    
    # Count states and model inputs with barrier_mask and lava_mask
    states_count = 0
    for key, value in env.items():
        if key != "layout" and isinstance(value, dict):
            states_count += 1
            if "model_inputs" in value and isinstance(value["model_inputs"], dict):
                if "barrier_mask" in value["model_inputs"]:
                    barrier_mask_count += 1
                if "lava_mask" in value["model_inputs"]:
                    lava_mask_count += 1
    
    print(f"Total states: {states_count}")
    print(f"States with barrier_mask: {barrier_mask_count}")
    print(f"States with lava_mask: {lava_mask_count}")
    
    # Check formatting with regex
    with open(output_file, 'r') as f:
        json_text = f.read()
    
    barrier_pattern = r'"barrier_mask":\s*\[\n\s+\['
    lava_pattern = r'"lava_mask":\s*\[\n\s+\['
    
    barrier_matches = len(re.findall(barrier_pattern, json_text))
    lava_matches = len(re.findall(lava_pattern, json_text))
    
    print(f"Detected formatted barrier_mask arrays: {barrier_matches}")
    print(f"Detected formatted lava_mask arrays: {lava_matches}")
    
    barrier_format_ok = barrier_matches == barrier_mask_count
    lava_format_ok = lava_matches == lava_mask_count
    
    print(f"Barrier format OK: {barrier_format_ok}")
    print(f"Lava format OK: {lava_format_ok}")
    
    # Overall result
    all_ok = barrier_format_ok and lava_format_ok
    if all_ok:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
    
    return all_ok

if __name__ == "__main__":
    test_real_log_formatting() 