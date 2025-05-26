#!/bin/bash

# Script to run category_summaries.py on all agent categories
# This processes all agent types and generates category summaries

echo "Running category summaries for all agent groups..."

# Read the all_agent_dirs.txt file to get all agent groups
while IFS= read -r line; do
  # Extract the agent group and path components
  FULL_PATH=$(dirname "$line")
  AGENT_GROUP=$(basename "$line")
  
  # Example line: Agent_Storage/LavaTests/Standard
  # We need to identify the main category (LavaTests) and the agent type (Standard)
  
  # Extract parts of the path
  CATEGORY=$(echo "$FULL_PATH" | cut -d'/' -f2)
  
  # For deeper paths like LavaTests/EpisodeEnd/Exponential/1_penalty
  # we need to combine the subcategories
  if [[ "$FULL_PATH" == *"EpisodeEnd/Exponential"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP} Exponential"
    BASE_DIR="LavaTests/EpisodeEnd/Exponential"
  elif [[ "$FULL_PATH" == *"EpisodeEnd/Linear"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP} Linear"
    BASE_DIR="LavaTests/EpisodeEnd/Linear"
  elif [[ "$FULL_PATH" == *"EpisodeEnd/Sigmoid"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP} Sigmoid"
    BASE_DIR="LavaTests/EpisodeEnd/Sigmoid"
  elif [[ "$FULL_PATH" == *"NoDeath"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP} NoDeath"
    BASE_DIR="LavaTests/NoDeath"
  elif [[ "$FULL_PATH" == *"LavaTests/Standard"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP}"
    BASE_DIR="LavaTests"
  elif [[ "$FULL_PATH" == *"LavaTests/Standard2"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP}"
    BASE_DIR="LavaTests"
  elif [[ "$FULL_PATH" == *"LavaTests/Standard3"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP}"
    BASE_DIR="LavaTests"
  elif [[ "$FULL_PATH" == *"SpawnTests"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP}"
    BASE_DIR="SpawnTests"
  elif [[ "$FULL_PATH" == *"WindowSizing"* ]]; then
    DISPLAY_NAME="${AGENT_GROUP}"
    BASE_DIR="WindowSizing"
  else
    DISPLAY_NAME="${AGENT_GROUP}"
    BASE_DIR="$CATEGORY"
  fi
  
  echo "========================================================================================="
  echo "Processing category summaries for: ${DISPLAY_NAME}"
  echo "Base directory: ${BASE_DIR}"
  echo "Agent type: ${AGENT_GROUP}"
  echo "========================================================================================="
  
  # Run the category_summaries.py script for this agent group
  python ./Agent_Evaluation/category_summaries.py --base "${BASE_DIR}" --agent "${AGENT_GROUP}"
  
  echo "Completed category summaries for ${DISPLAY_NAME}"
  echo ""
  
done < all_agent_dirs.txt

echo "All category summaries have been generated!" 