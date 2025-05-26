#!/usr/bin/env python3
"""
Example script demonstrating different state filtering criteria combinations.
Run these examples by passing different example numbers to the script:

python state_filter_examples.py --example 1 --path /path/to/agent/directory

"""

import os
import argparse
import subprocess

# Define example criteria for different scenarios
EXAMPLES = {
    1: {
        "name": "Goal reached without lava steps",
        "criteria": ["reaches_goal:is:true", "lava_steps:eq:0"],
        "description": "Find states where the agent can reach the goal without stepping in lava."
    },
    2: {
        "name": "Floor cell with lava ahead",
        "criteria": ["cell_type:eq:floor", "next_cell_is_lava:is:true"],
        "description": "Find states where the agent is on a floor cell but the next cell is lava."
    },
    3: {
        "name": "Risky diagonal near lava",
        "criteria": ["risky_diagonal:is:true", "next_cell_is_lava:is:true"],
        "description": "Find states where the agent makes a risky diagonal move near lava."
    },
    4: {
        "name": "Goal reached despite lava steps",
        "criteria": ["reaches_goal:is:true", "lava_steps:gt:0"],
        "description": "Find states where the agent reaches the goal despite stepping in lava."
    },
    5: {
        "name": "Safe path to goal with lava ahead",
        "criteria": ["reaches_goal:is:true", "lava_steps:eq:0", "next_cell_is_lava:is:true", "cell_type:eq:floor"],
        "description": "Find states where the agent can reach the goal without stepping in lava, but the next step is lava."
    },
    6: {
        "name": "Short path to goal",
        "criteria": ["reaches_goal:is:true", "path_length:lt:15"],
        "description": "Find states with short paths to the goal."
    },
    7: {
        "name": "Forward action taken",
        "criteria": ["action_taken:eq:2"],  # Assuming 2 represents forward
        "description": "Find states where the agent took the forward action."
    },
    8: {
        "name": "Efficient lava navigation",
        "criteria": ["reaches_goal:is:true", "lava_steps:gt:0", "path_length:lt:20"],
        "description": "Find states where the agent reaches the goal efficiently despite stepping in lava."
    },
    9: {
        "name": "Safe diagonal moves",
        "criteria": ["risky_diagonal:is:true", "next_cell_is_lava:is:false"],
        "description": "Find states where the agent makes diagonal moves that aren't near lava."
    },
    10: {
        "name": "Long paths without lava",
        "criteria": ["reaches_goal:is:true", "lava_steps:eq:0", "path_length:gt:25"],
        "description": "Find states where the agent takes long paths to avoid lava completely."
    }
}

def list_examples():
    """Print a list of all available examples with descriptions."""
    print("Available examples:")
    print("-" * 80)
    for num, example in EXAMPLES.items():
        print(f"{num}. {example['name']}")
        print(f"   Description: {example['description']}")
        print(f"   Criteria: {example['criteria']}")
        print()

def run_state_filter(agent_path, criteria, output_suffix=None):
    """Run the state_filter.py script with the given criteria."""
    # Modify the state_filter.py file to use the specified criteria
    filter_path = os.path.join(os.path.dirname(__file__), "state_filter.py")
    
    # Read the original file
    with open(filter_path, 'r') as f:
        content = f.read()
    
    # Find the CRITERIA line and replace it
    criteria_str = repr(criteria)
    if "CRITERIA = " in content:
        import re
        new_content = re.sub(r'CRITERIA = \[.*?\]', f'CRITERIA = {criteria_str}', content, flags=re.DOTALL)
    else:
        print("Warning: Could not find CRITERIA line in state_filter.py")
        new_content = content
    
    # Write the modified file
    with open(filter_path, 'w') as f:
        f.write(new_content)
    
    # Create output path with suffix if provided
    if output_suffix:
        output_path = os.path.join(agent_path, f'filtered_states_{output_suffix}.json')
    else:
        output_path = os.path.join(agent_path, 'filtered_states.json')
    
    # Run the state_filter.py script
    cmd = ["python", filter_path, "--path", agent_path, "--output", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Successfully filtered states: {result.stdout}")
    else:
        print(f"Error filtering states: {result.stderr}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run state filtering examples.')
    parser.add_argument('--example', type=int, help='Example number to run (1-10)')
    parser.add_argument('--path', help='Path to the agent directory')
    parser.add_argument('--list', action='store_true', help='List all available examples')
    
    args = parser.parse_args()
    
    if args.list:
        list_examples()
        exit(0)
    
    if not args.example or not args.path:
        parser.print_help()
        exit(1)
    
    if args.example not in EXAMPLES:
        print(f"Error: Example {args.example} not found. Use --list to see available examples.")
        exit(1)
    
    example = EXAMPLES[args.example]
    print(f"Running example {args.example}: {example['name']}")
    print(f"Description: {example['description']}")
    print(f"Criteria: {example['criteria']}")
    
    output_path = run_state_filter(args.path, example['criteria'], f"example_{args.example}")
    print(f"Results saved to: {output_path}") 