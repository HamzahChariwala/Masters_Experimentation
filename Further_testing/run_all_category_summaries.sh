#!/bin/bash

# Script to run category_summaries.py on all agent categories
# This processes all agent types and generates category summaries

echo "Running category summaries for all agent groups..."

# Read the all_agent_dirs.txt file to get all agent groups
while IFS= read -r line; do
  # Extract the agent group and base directory
  AGENT_BASE_DIR=$(dirname "$line")
  AGENT_GROUP=$(basename "$line")
  
  # Display which group we're processing
  if [[ "$AGENT_BASE_DIR" == *"EpisodeEnd/Exponential"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP} Exponential"
  elif [[ "$AGENT_BASE_DIR" == *"EpisodeEnd/Linear"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP} Linear"
  elif [[ "$AGENT_BASE_DIR" == *"EpisodeEnd/Sigmoid"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP} Sigmoid"
  elif [[ "$AGENT_BASE_DIR" == *"NoDeath"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP} NoDeath"
  elif [[ "$AGENT_BASE_DIR" == *"WindowSizing"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP}"
  else
    DISPLAY_NAME="${AGENT_GROUP}"
  fi
  
  echo "========================================================================================="
  echo "Processing category summaries for: ${DISPLAY_NAME}"
  echo "========================================================================================="
  
  # Run the category_summaries.py script for this agent group
  python Agent_Evaluation/SummaryTooling/category_summaries.py --agent_type "${AGENT_GROUP}"
  
  echo "Completed category summaries for ${DISPLAY_NAME}"
  echo ""
  
done < all_agent_dirs.txt

echo "All category summaries have been generated!" 