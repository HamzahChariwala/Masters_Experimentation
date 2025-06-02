#!/usr/bin/env python3
"""
Simple script to run the corruption tool with agent data.
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run the State Corruption Tool')
    parser.add_argument('--path', required=True, help='Path to the agent directory with filtered_states.json')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for state selection')
    
    args = parser.parse_args()
    
    # Get directory of this script and path to corruption_tool.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corruption_tool_path = os.path.join(script_dir, 'corruption_tool.py')
    
    # Run the corruption tool
    python_exe = sys.executable
    cmd = [python_exe, corruption_tool_path, '--path', args.path]
    
    if args.seed:
        cmd.extend(['--seed', str(args.seed)])
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main()) 