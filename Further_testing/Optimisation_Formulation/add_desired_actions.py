#!/usr/bin/env python3
"""
Add Desired Actions - Augment alter states with reference actions from dijkstra agents.

This script takes the alter states from optimization filtering and adds a 'desired_action' 
field by looking up what action a specified dijkstra agent (ruleset) took in the same 
environment and state.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and return contents of a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file with proper formatting."""
    with open(file_path, 'w') as f:
        formatted_json = "{\n"
        entries = []
        
        for key, value in data.items():
            entry = f'  "{key}": {json.dumps(value)}'
            entries.append(entry)
        
        formatted_json += ",\n".join(entries)
        formatted_json += "\n}"
        f.write(formatted_json)

def parse_alter_state_key(key: str) -> tuple:
    """
    Parse an alter state key to extract environment and state components.
    
    Args:
        key: Key like "MiniGrid-LavaCrossingS11N5-v0-81103-2,1,1-0001"
        
    Returns:
        Tuple of (environment, state_key) where environment is the env name 
        and state_key is the state coordinates
    """
    # Split by last dash to separate counter suffix
    base_key = '-'.join(key.split('-')[:-1])
    
    # Split by the last dash again to separate state from environment  
    parts = base_key.split('-')
    
    # The state key is the last part, environment is everything before
    state_key = parts[-1]
    environment = '-'.join(parts[:-1])
    
    return environment, state_key

def get_dijkstra_action(env_name: str, state_key: str, ruleset: str, 
                       dijkstra_base_path: str) -> Optional[int]:
    """
    Get the action taken by a dijkstra agent for a specific environment and state.
    
    Args:
        env_name: Environment name (e.g., "MiniGrid-LavaCrossingS11N5-v0-81103")
        state_key: State coordinates (e.g., "2,1,1") 
        ruleset: Dijkstra agent ruleset (e.g., "standard")
        dijkstra_base_path: Base path to dijkstra evaluation files
        
    Returns:
        Action number if found, None otherwise
    """
    dijkstra_file = os.path.join(dijkstra_base_path, f"{env_name}.json")
    
    if not os.path.exists(dijkstra_file):
        return None
    
    try:
        dijkstra_data = load_json_file(dijkstra_file)
        
        # Navigate to the performance data for the specified ruleset
        if 'performance' not in dijkstra_data or ruleset not in dijkstra_data['performance']:
            return None
            
        ruleset_data = dijkstra_data['performance'][ruleset]
        
        # Get the action for the specific state
        if state_key in ruleset_data:
            state_array = ruleset_data[state_key]
            # Action is at index 7 in the array
            action = state_array[7]
            return action if action is not None else None
        
        return None
        
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None

def add_desired_actions(alter_states_path: str, dijkstra_base_path: str, 
                       ruleset: str = "standard") -> Dict[str, int]:
    """
    Add desired actions to alter states based on dijkstra agent behavior.
    
    Args:
        alter_states_path: Path to the alter states JSON file
        dijkstra_base_path: Path to directory containing dijkstra evaluation files
        ruleset: Dijkstra agent ruleset to use as reference
        
    Returns:
        Dictionary with statistics about the operation
    """
    print(f"Loading alter states from: {alter_states_path}")
    alter_states = load_json_file(alter_states_path)
    
    stats = {
        'total_states': len(alter_states),
        'actions_found': 0,
        'actions_missing': 0,
        'environments_processed': set()
    }
    
    print(f"Processing {stats['total_states']} alter states with ruleset '{ruleset}'")
    
    for state_key, state_data in alter_states.items():
        # Parse the state key to get environment and coordinates
        environment, coordinates = parse_alter_state_key(state_key)
        stats['environments_processed'].add(environment)
        
        # Get the desired action from dijkstra agent
        desired_action = get_dijkstra_action(environment, coordinates, ruleset, dijkstra_base_path)
        
        if desired_action is not None:
            state_data['desired_action'] = desired_action
            stats['actions_found'] += 1
        else:
            stats['actions_missing'] += 1
            print(f"Warning: No action found for {environment} state {coordinates}")
    
    # Convert set to count for final stats
    stats['environments_processed'] = len(stats['environments_processed'])
    
    print(f"Saving updated alter states to: {alter_states_path}")
    save_json_file(alter_states, alter_states_path)
    
    return stats

def main():
    """Main entry point for adding desired actions."""
    parser = argparse.ArgumentParser(
        description='Add desired actions from dijkstra agents to alter states.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add standard dijkstra actions to alter states
  python add_desired_actions.py --alter_path ../Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6/optimisation_states/alter/states.json --ruleset standard
  
  # Use conservative dijkstra agent
  python add_desired_actions.py --alter_path ../Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6/optimisation_states/alter/states.json --ruleset conservative
  
  # Specify custom dijkstra path
  python add_desired_actions.py --alter_path path/to/alter/states.json --dijkstra_path path/to/dijkstra/files --ruleset standard
        """
    )
    
    parser.add_argument('--alter_path', required=True,
                       help='Path to the alter states JSON file')
    parser.add_argument('--dijkstra_path', default="../Behaviour_Specification/Evaluations",
                       help='Path to directory containing dijkstra evaluation files (default: ../Behaviour_Specification/Evaluations)')
    parser.add_argument('--ruleset', default="standard",
                       choices=['standard', 'conservative', 'dangerous_1', 'dangerous_2', 'dangerous_3', 'dangerous_4', 'dangerous_5'],
                       help='Dijkstra agent ruleset to use as reference (default: standard)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.alter_path):
        print(f"Error: Alter states file not found: {args.alter_path}")
        return 1
        
    if not os.path.exists(args.dijkstra_path):
        print(f"Error: Dijkstra directory not found: {args.dijkstra_path}")
        return 1
    
    try:
        # Add desired actions
        stats = add_desired_actions(args.alter_path, args.dijkstra_path, args.ruleset)
        
        # Print summary
        print("\n" + "="*60)
        print("ADD DESIRED ACTIONS SUMMARY")
        print("="*60)
        print(f"Alter states file: {args.alter_path}")
        print(f"Dijkstra ruleset: {args.ruleset}")
        print(f"Total states processed: {stats['total_states']}")
        print(f"Actions found: {stats['actions_found']}")
        print(f"Actions missing: {stats['actions_missing']}")
        print(f"Environments processed: {stats['environments_processed']}")
        print(f"Success rate: {stats['actions_found']/stats['total_states']*100:.1f}%")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 