#!/bin/bash

# Script to train and evaluate all non-v1 agents in Agent_Storage

echo "Starting characterisation (training + evaluation) for all non-v1 agents..."

# Find all directories with config.yaml files, excluding those ending in -v1
agent_dirs=$(find Agent_Storage -name "config.yaml" -type f -exec dirname {} \; | grep -v "\-v1$" | sort)

# Loop through each agent directory for training
echo "=== Starting Training Phase ==="
for agent_dir in $agent_dirs; do
  # Extract relative path from Agent_Storage
  relative_path=${agent_dir#"Agent_Storage/"}
  
  echo "========================================================================================="
  echo "Training agent: $relative_path"
  echo "========================================================================================="
  
  # Run the training command with the appropriate path
  python Agent_Training/train.py --path $relative_path
  
  # Add a small delay between trainings to allow for proper initialization
  sleep 2
done

echo "Training phase complete!"
echo ""

# Loop through each agent directory for evaluation
echo "=== Starting Evaluation Phase ==="
for agent_dir in $agent_dirs; do
  echo "========================================================================================="
  echo "Evaluating agent: $agent_dir"
  echo "========================================================================================="
  
  # Run the evaluation command with the appropriate full path
  python Agent_Evaluation/generate_evaluations.py --path "$agent_dir"
  
  echo "Evaluation complete for $agent_dir"
  echo ""
done

echo "All agents have been trained and evaluated!" 