#!/bin/bash

# Script for evaluating default agents

AGENT_GROUP="default"
AGENT_BASE_DIR="Agent_Storage/SpawnTests/default"

echo "Starting evaluations for ${AGENT_GROUP} agents..."

# Loop through each agent version (v1 to v10)
for version in {1..10}; do
  agent_dir="${AGENT_BASE_DIR}/${AGENT_GROUP}-v${version}"
  
  # Check if the agent directory exists
  if [ -d "$agent_dir" ]; then
    # Check if final_eval.json exists
    if [ ! -f "$agent_dir/final_eval.json" ]; then
      echo "========================================================================================="
      echo "Evaluating agent: $agent_dir"
      echo "========================================================================================="
      
      # Run the evaluation command
      python Agent_Evaluation/generate_evaluations.py --path "$agent_dir"
      
      # Run the comparison evaluation
      python Agent_Evaluation/SummaryTooling/comparison_evaluation.py --agent "$(basename "$agent_dir")"
      
      echo "Evaluation complete for $agent_dir"
      echo ""
    else
      echo "Skipping evaluation for $agent_dir - final_eval.json exists"
    fi
  else
    echo "Agent directory does not exist: $agent_dir"
  fi
done

echo "Evaluations complete for ${AGENT_GROUP} agents!"
