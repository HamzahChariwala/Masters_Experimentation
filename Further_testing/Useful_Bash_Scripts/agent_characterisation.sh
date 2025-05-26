#!/bin/bash

# Script to train and evaluate agents in Agent_Storage that need processing

echo "Starting characterisation (training + evaluation) for agents..."

# Find all directories with config.yaml files, excluding those ending in -v1
agent_dirs=$(find Agent_Storage -name "config.yaml" -type f -exec dirname {} \; | grep -v "\-v1$" | sort)

# Loop through each agent directory for training
echo "=== Starting Training Phase ==="
for agent_dir in $agent_dirs; do
  # Extract relative path from Agent_Storage
  relative_path=${agent_dir#"Agent_Storage/"}
  
  # Check if agent.zip exists
  if [ ! -f "$agent_dir/agent.zip" ]; then
    echo "========================================================================================="
    echo "Training agent: $relative_path (no agent.zip found)"
    echo "========================================================================================="
    
    # Run the training command with the appropriate path
    python Agent_Training/train.py --path $relative_path
    
    # Add a small delay between trainings to allow for proper initialization
    sleep 2
  else
    echo "Skipping training for $relative_path - agent.zip exists"
  fi
done

echo "Training phase complete!"
echo ""

# Loop through each agent directory for evaluation
echo "=== Starting Evaluation Phase ==="
for agent_dir in $agent_dirs; do
  # Check if final_eval.json exists
  if [ ! -f "$agent_dir/final_eval.json" ]; then
    echo "========================================================================================="
    echo "Evaluating agent: $agent_dir (no final_eval.json found)"
    echo "========================================================================================="
    
    # Run the evaluation command with the appropriate full path
    python Agent_Evaluation/generate_evaluations.py --path "$agent_dir"
    
    # Run the comparison evaluation
    python Agent_Evaluation/SummaryTooling/comparison_evaluation.py --agent "$(basename "$agent_dir")"
    
    echo "Evaluation complete for $agent_dir"
    echo ""
  else
    echo "Skipping evaluation for $agent_dir - final_eval.json exists"
  fi
done

# Loop through each agent directory for reachable-path evaluation
echo "=== Starting Reachable-Path Evaluation Phase ==="
for agent_dir in $agent_dirs; do
  # Check if evaluation_logs directory exists and performance_reachable_paths.json doesn't exist
  if [ -d "$agent_dir/evaluation_logs" ] && [ ! -f "$agent_dir/evaluation_summary/performance_reachable_paths.json" ]; then
    echo "========================================================================================="
    echo "Generating reachable-path evaluation for: $agent_dir"
    echo "========================================================================================="
    
    # Run the reachable-path evaluation command
    python Agent_Evaluation/SummaryTooling/reachable_path_evaluation.py --agent "$(basename "$agent_dir")"
    
    echo "Reachable-path evaluation complete for $agent_dir"
    echo ""
  else
    if [ ! -d "$agent_dir/evaluation_logs" ]; then
      echo "Skipping reachable-path evaluation for $agent_dir - no evaluation_logs directory"
    else
      echo "Skipping reachable-path evaluation for $agent_dir - performance_reachable_paths.json exists"
    fi
  fi
done

echo "All necessary agents have been trained and evaluated!" 