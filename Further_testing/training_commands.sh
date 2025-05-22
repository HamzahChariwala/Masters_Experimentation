#!/bin/bash

# Script to train all agents in Agent_Storage

echo "Starting training for all agents..."

# Find all directories with config.yaml files
agent_dirs=$(find Agent_Storage -name "config.yaml" -type f -exec dirname {} \; | sort)

# Loop through each agent directory and queue training
for agent_dir in $agent_dirs; do
  # Extract relative path from Agent_Storage
  relative_path=${agent_dir#"Agent_Storage/"}
  
  echo "Queueing training for $relative_path"
  
  # Run the training command with the appropriate path
  echo "python Agent_Training/train.py --path $relative_path"
  python Agent_Training/train.py --path $relative_path
  
  # Add a small delay between trainings to allow for proper initialization
  sleep 2
done

echo "All training jobs have been queued!"