#!/usr/bin/env python3
import os
import sys
import argparse
from typing import List, Optional

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import our summary function
from SummaryTooling.evaluation_summary import process_evaluation_logs


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the evaluation summary tool.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Convert agent_paths to full paths if specified
    agent_dirs = None
    if args.agent_paths:
        agent_dirs = []
        for path in args.agent_paths:
            # Check if path is absolute
            if os.path.isabs(path):
                agent_dirs.append(path)
            # Check if path already includes Agent_Storage
            elif path.startswith("Agent_Storage/"):
                agent_dirs.append(os.path.join(project_root, path))
            else:
                # Assume path is relative to Agent_Storage
                agent_dirs.append(os.path.join(project_root, "Agent_Storage", path))
    
    # Determine output directory
    output_dir = args.output_dir
    if output_dir:
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    # Process evaluation logs
    print(f"Processing evaluation logs for {len(agent_dirs) if agent_dirs else 'all'} agents...")
    summaries = process_evaluation_logs(
        agent_dirs=agent_dirs,
        save_results=not args.no_save,
        output_dir=output_dir
    )
    
    # Print summary of results
    total_agents = len(summaries)
    total_logs = sum(len(logs) for logs in summaries.values())
    print(f"\nSummary: Processed {total_logs} evaluation logs across {total_agents} agents")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate summary statistics for agent evaluation logs")
    parser.add_argument("--agent-paths", type=str, nargs="+", 
                        help="Paths to agent directories (relative to Agent_Storage or absolute)")
    parser.add_argument("--output-dir", type=str, 
                        help="Directory to save the results (defaults to agent's evaluation_logs directory)")
    parser.add_argument("--no-save", action="store_true", 
                        help="Don't save the results to a file (just print to console)")
    args = parser.parse_args()
    
    main(args) 