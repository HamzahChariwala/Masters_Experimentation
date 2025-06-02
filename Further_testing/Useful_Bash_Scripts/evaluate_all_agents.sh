#!/bin/bash

# Script to evaluate all agents in Agent_Storage

echo "Starting evaluation for all agents..."

# Find all directories with config.yaml files
agent_dirs=$(find Agent_Storage -name "config.yaml" -type f -exec dirname {} \; | sort)

# Loop through each agent directory and run evaluation
for agent_dir in $agent_dirs; do
  # Don't extract the relative path, use the full agent_dir path
  echo "========================================================================================="
  echo "Evaluating agent: $agent_dir"
  echo "========================================================================================="
  
  # Run the evaluation command with the appropriate full path
  python Agent_Evaluation/generate_evaluations.py --path "$agent_dir"
  
  echo "Evaluation complete for $agent_dir"
  echo ""
done

echo "All agents have been evaluated!" 