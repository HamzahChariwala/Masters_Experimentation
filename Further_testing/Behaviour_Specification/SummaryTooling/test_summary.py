"""
Test script for evaluation_summary.py
"""

import os
import shutil
import json
from evaluation_summary import process_dijkstra_logs

def test_dijkstra_log_processing():
    """
    Test processing of Dijkstra logs.
    """
    print("Testing Dijkstra log processing...")
    
    # Set up test directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    test_output_dir = os.path.join(script_dir, "test_output")
    
    # Create test output directory if it doesn't exist
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Copy test data to output dir to preserve originals
    sample_file_path = os.path.join(script_dir, "test_data", "sample_dijkstra_dangerous_modes.json")
    test_file_path = os.path.join(test_output_dir, "test_dijkstra_all_modes.json")
    shutil.copy(sample_file_path, test_file_path)
    
    print(f"Using test output directory: {test_output_dir}")
    print(f"Copied sample file to: {test_file_path}")
    
    # Process just the test file with dangerous modes
    all_mode_summaries = process_dijkstra_logs(
        logs_dir=test_output_dir,  # Use the test output dir as the logs dir
        save_results=True,
        output_dir=test_output_dir
    )
    
    # Verify results
    print("\nProcessing Results:")
    modes_processed = []
    total_env_modes = 0
    for mode, environments in all_mode_summaries.items():
        if environments:
            modes_processed.append(mode)
            env_count = len(environments)
            total_env_modes += env_count
            print(f"  Mode '{mode}': {env_count} environments processed")
    
    print(f"\nTotal modes detected: {len(modes_processed)}")
    print(f"Total environment-mode combinations: {total_env_modes}")
    
    # Check if dangerous mode summaries were created
    summary_files = [
        f for f in os.listdir(test_output_dir) 
        if f.startswith("performance_summary_") and f.endswith(".json")
    ]
    print(f"Summary files created: {len(summary_files)}")
    for f in summary_files:
        print(f"  - {f}")
    
    # Check the processed dangerous mode file
    dangerous_file = os.path.join(test_output_dir, "test_dijkstra_all_modes.json")
    if os.path.exists(dangerous_file):
        print("\nVerifying dangerous mode processing...")
        try:
            with open(dangerous_file, 'r') as f:
                data = json.load(f)
                
            if "performance" in data:
                print("✅ Performance data added to test file")
                performance = data["performance"]
                detected_modes = list(performance.keys())
                print(f"  Detected modes: {', '.join(detected_modes)}")
                
                # Check if performance is first key
                first_key = list(data.keys())[0]
                if first_key == "performance":
                    print("✅ Performance is the first key in the file")
                else:
                    print(f"❌ Performance is not the first key. First key is: {first_key}")
                
                # Check if all dangerous modes are present
                dangerous_modes = ["dangerous_1", "dangerous_2", "dangerous_3", "dangerous_4", "dangerous_5"]
                missing_modes = [mode for mode in dangerous_modes if mode not in detected_modes]
                if not missing_modes:
                    print("✅ All dangerous modes were detected")
                else:
                    print(f"❌ Missing dangerous modes: {', '.join(missing_modes)}")
            else:
                print("❌ No performance data found in test file")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Error reading test file: {e}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_dijkstra_log_processing() 