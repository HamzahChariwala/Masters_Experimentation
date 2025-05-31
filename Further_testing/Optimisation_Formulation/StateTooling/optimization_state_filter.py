#!/usr/bin/env python3
"""
Optimization State Filter - Filter states for optimization problems with preserve/alter categorization.

This script builds upon the existing state_filter.py functionality to create filtered state sets 
for optimization problems. It allows filtering states into 'preserve' and 'alter' categories
and saves them to the agent's optimization directory.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any, Callable, Union

# Add the Neuron_Selection/ObservationTooling directory to path to import state_filter functions
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
observation_tooling_dir = os.path.join(project_root, 'Neuron_Selection', 'ObservationTooling')
sys.path.insert(0, observation_tooling_dir)

# Import functions from the existing state_filter.py
from state_filter import (
    create_state_filter,
    collect_matching_states,
    DEFAULT_PROPERTIES,
    save_results
)

def create_optimization_directories(agent_path: str) -> Tuple[str, str]:
    """
    Create optimization directory structure in the agent directory.
    
    Args:
        agent_path: Path to the agent directory
        
    Returns:
        Tuple of (preserve_dir, alter_dir) paths
    """
    optimization_dir = os.path.join(agent_path, 'optimisation_states')
    preserve_dir = os.path.join(optimization_dir, 'preserve')
    alter_dir = os.path.join(optimization_dir, 'alter')
    
    # Create directories if they don't exist
    os.makedirs(preserve_dir, exist_ok=True)
    os.makedirs(alter_dir, exist_ok=True)
    
    return preserve_dir, alter_dir

def filter_states_for_optimization(
    agent_path: str,
    alter_criteria: List[str],
    preserve_criteria: List[str] = None,
    output_suffix: str = ""
) -> Dict[str, int]:
    """
    Filter states for optimization purposes into preserve and alter categories.
    
    Args:
        agent_path: Path to the agent directory
        alter_criteria: List of criteria strings for 'alter' states
        preserve_criteria: List of criteria strings for 'preserve' states (optional)
        output_suffix: Suffix to add to output filenames
        
    Returns:
        Dictionary with counts of filtered states
    """
    # Create optimization directories
    preserve_dir, alter_dir = create_optimization_directories(agent_path)
    
    # Create filter function for alter states
    alter_filter = create_state_filter(alter_criteria)
    
    # Collect alter states
    print(f"Filtering ALTER states with criteria: {alter_criteria}")
    alter_states = collect_matching_states(agent_path, alter_filter, DEFAULT_PROPERTIES)
    print(f"Found {len(alter_states)} ALTER states")
    
    # Save alter states
    alter_filename = f"states{output_suffix}.json" if output_suffix else "states.json"
    alter_output_path = os.path.join(alter_dir, alter_filename)
    save_results(alter_states, alter_output_path)
    print(f"ALTER states saved to: {alter_output_path}")
    
    # Handle preserve states
    preserve_count = 0
    if preserve_criteria is not None:
        # Use specific preserve criteria
        print(f"Filtering PRESERVE states with criteria: {preserve_criteria}")
        preserve_filter = create_state_filter(preserve_criteria)
        preserve_states = collect_matching_states(agent_path, preserve_filter, DEFAULT_PROPERTIES)
        preserve_count = len(preserve_states)
        print(f"Found {preserve_count} PRESERVE states")
        
        # Save preserve states
        preserve_filename = f"states{output_suffix}.json" if output_suffix else "states.json"
        preserve_output_path = os.path.join(preserve_dir, preserve_filename)
        save_results(preserve_states, preserve_output_path)
        print(f"PRESERVE states saved to: {preserve_output_path}")
    else:
        # Use complement of alter states (everything not in alter set)
        print("Using complement strategy for PRESERVE states (all states not in ALTER set)")
        
        # Get all states (no filter)
        all_states_filter = create_state_filter([])  # Empty criteria matches all states
        all_states = collect_matching_states(agent_path, all_states_filter, DEFAULT_PROPERTIES)
        
        # Create preserve states by excluding alter states
        alter_state_keys = set(alter_states.keys())
        preserve_states = {key: value for key, value in all_states.items() 
                          if key not in alter_state_keys}
        preserve_count = len(preserve_states)
        print(f"Found {preserve_count} PRESERVE states (complement of ALTER)")
        
        # Save preserve states
        preserve_filename = f"states{output_suffix}.json" if output_suffix else "states.json"
        preserve_output_path = os.path.join(preserve_dir, preserve_filename)
        save_results(preserve_states, preserve_output_path)
        print(f"PRESERVE states saved to: {preserve_output_path}")
    
    return {
        'alter_count': len(alter_states),
        'preserve_count': preserve_count,
        'total_count': len(alter_states) + preserve_count
    }

def main():
    """Main entry point for the optimization state filter."""
    parser = argparse.ArgumentParser(
        description='Filter states for optimization problems into preserve/alter categories.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter with specific alter criteria and complement preserve
  python optimization_state_filter.py --path /path/to/agent --alter "blocked:is:false" "next_cell_is_lava:is:true" "cell_type:eq:floor"
  
  # Filter with both alter and preserve criteria
  python optimization_state_filter.py --path /path/to/agent --alter "blocked:is:false" --preserve "reaches_goal:is:true" "lava_steps:eq:0"
  
  # Add suffix to output files
  python optimization_state_filter.py --path /path/to/agent --alter "blocked:is:false" --suffix "_test"
        """
    )
    
    parser.add_argument('--path', required=True, 
                       help='Path to the agent directory')
    parser.add_argument('--alter', nargs='+', required=True,
                       help='Criteria for ALTER states (space-separated list)')
    parser.add_argument('--preserve', nargs='*', default=None,
                       help='Criteria for PRESERVE states (space-separated list). If not provided, uses complement of ALTER states')
    parser.add_argument('--suffix', default="",
                       help='Suffix to add to output filenames (default: none)')
    
    args = parser.parse_args()
    
    # Validate agent path
    if not os.path.exists(args.path):
        print(f"Error: Agent path does not exist: {args.path}")
        return 1
    
    eval_logs_path = os.path.join(args.path, 'evaluation_logs')
    if not os.path.exists(eval_logs_path):
        print(f"Error: Evaluation logs directory not found at {eval_logs_path}")
        return 1
    
    try:
        # Filter states
        results = filter_states_for_optimization(
            agent_path=args.path,
            alter_criteria=args.alter,
            preserve_criteria=args.preserve,
            output_suffix=args.suffix
        )
        
        # Print summary
        print("\n" + "="*60)
        print("OPTIMIZATION STATE FILTERING SUMMARY")
        print("="*60)
        print(f"Agent path: {args.path}")
        print(f"ALTER criteria: {args.alter}")
        print(f"PRESERVE criteria: {args.preserve if args.preserve else 'Complement of ALTER'}")
        print(f"ALTER states found: {results['alter_count']}")
        print(f"PRESERVE states found: {results['preserve_count']}")
        print(f"Total states processed: {results['total_count']}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 